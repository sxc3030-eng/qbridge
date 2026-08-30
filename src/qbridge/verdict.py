"""Verdicts de conformite d'un rejeu.

L'echelle est graduee parce que l'egalite bit-pour-bit est le mauvais critere
des qu'on quitte un simulateur deterministe. Un backend materiel ne pourra
jamais faire mieux que STATISTICALLY_COMPATIBLE — c'est une propriete de la
physique, pas un defaut du harnais.

Le chi2 est implemente ici plutot que via scipy : une dependance de moins pour
un harnais cense rester lisible et executable dans dix ans.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional

import numpy as np

INFIDELITY_TOLERANCE = 1e-4
"""Seuil d'infidelite sous lequel deux vecteurs d'etat sont juges equivalents.
Calibre au-dessus de l'ecart mesure entre max_fused_gate_size=2 et 4
(infidelite ~1.4e-5) et bien au-dessus du bruit d'arrondi de complex64."""

CHI2_ALPHA = 0.001
"""Seuil de p-value. Volontairement bas : on veut detecter une vraie
divergence, pas signaler du bruit d'echantillonnage normal."""

MIN_EXPECTED_COUNT = 5.0
"""Effectif attendu minimal par categorie pour que le chi2 soit valide.

Regle classique. Sans elle le test est un piege : a 20 qubits avec 200 tirages,
presque chaque bitstring est unique, chaque categorie contribue 1.0 au chi2 avec
ddl = categories-1, donc p tend vers 0.5 QUELLES QUE SOIENT les donnees. Mesure :
deux jeux de tirages totalement independants a 20 qubits ressortaient
'compatibles' avec p=0.476, et l'ecart mesure entre cpu_threads=1 et 4 en mode
midcircuit ressortait 'compatible' avec p=1.0.

Ce n'est pas un defaut du chi2 : avec 200 tirages sur 2^20 issues possibles,
AUCUN test ne peut distinguer deux distributions. C'est une limite de complexite
d'echantillonnage. La seule reponse honnete est de refuser de conclure."""


class Verdict(IntEnum):
    """Du plus fort au plus faible.

    `INDETERMINATE` est place APRES `DIVERGENT` pour que les comparaisons du
    type `verdict <= NUMERICALLY_EQUIVALENT` ne l'acceptent jamais : ne pas
    pouvoir conclure n'est pas une reussite.
    """

    BIT_EXACT = 0
    NUMERICALLY_EQUIVALENT = 1
    STATISTICALLY_COMPATIBLE = 2
    DIVERGENT = 3
    INDETERMINATE = 4


@dataclass(frozen=True)
class ComparisonResult:
    verdict: Verdict
    detail: str
    infidelity: Optional[float] = None
    p_value: Optional[float] = None
    max_abs_delta: Optional[float] = None
    """Ecart maximal amplitude par amplitude.

    Indispensable a cote de l'infidelite, qui SATURE : la fidelite est
    quadratique pres de 1, donc un ecart de 4e-8 par amplitude produit une
    infidelite de ~1.6e-15, sous la resolution du float64 — elle s'affiche
    alors comme 0.0 alors que les vecteurs different reellement. Mesure :
    qsim vs cirq sur 12 qubits donne infidelite 0.0 mais max|delta| 4e-8 sur
    4093 des 4096 amplitudes."""


def compare_state_vectors(a: np.ndarray, b: np.ndarray) -> ComparisonResult:
    """Compare deux vecteurs d'etat."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return ComparisonResult(
            Verdict.DIVERGENT, f"formes incompatibles : {a.shape} vs {b.shape}"
        )
    if a.dtype == b.dtype and a.tobytes() == b.tobytes():
        return ComparisonResult(
            Verdict.BIT_EXACT, "octets identiques", 0.0, max_abs_delta=0.0
        )

    infidelite = abs(1.0 - float(abs(np.vdot(a, b)) ** 2))
    ecart_max = float(np.abs(a - b).max())
    if infidelite <= INFIDELITY_TOLERANCE:
        return ComparisonResult(
            Verdict.NUMERICALLY_EQUIVALENT,
            f"infidelite {infidelite:.3e} <= {INFIDELITY_TOLERANCE:.0e}, "
            f"max|delta| {ecart_max:.3e}",
            infidelite,
            max_abs_delta=ecart_max,
        )
    return ComparisonResult(
        Verdict.DIVERGENT,
        f"infidelite {infidelite:.3e} > {INFIDELITY_TOLERANCE:.0e}, "
        f"max|delta| {ecart_max:.3e}",
        infidelite,
        max_abs_delta=ecart_max,
    )


def bitstring_counts(samples: np.ndarray) -> Dict[int, int]:
    """Convertit un tableau (repetitions, n_qubits) en comptage par entier."""
    arr = np.asarray(samples)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    # dtype=int64 explicite : sous numpy 1.x sur Windows, arange rend de l'int32
    # et le decalage deborde silencieusement au-dela de 31 qubits, corrompant
    # les comptages et donc tous les verdicts qui en dependent.
    poids = np.int64(1) << np.arange(arr.shape[1] - 1, -1, -1, dtype=np.int64)
    valeurs = (arr.astype(np.int64) * poids).sum(axis=1)
    uniques, comptes = np.unique(valeurs, return_counts=True)
    return {int(u): int(c) for u, c in zip(uniques, comptes)}


def _regularized_gamma_p(a: float, x: float) -> float:
    """Fonction gamma incomplete inferieure regularisee P(a, x)."""
    if x <= 0 or a <= 0:
        return 0.0
    if x < a + 1.0:
        terme = 1.0 / a
        somme = terme
        n = a
        for _ in range(1000):
            n += 1.0
            terme *= x / n
            somme += terme
            if abs(terme) < abs(somme) * 1e-15:
                break
        return somme * math.exp(-x + a * math.log(x) - math.lgamma(a))
    minuscule = 1e-300
    b = x + 1.0 - a
    c = 1.0 / minuscule
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < minuscule:
            d = minuscule
        c = b + an / c
        if abs(c) < minuscule:
            c = minuscule
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return 1.0 - h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _chi2_survival(x: float, k: int) -> float:
    """P(X > x) pour X ~ chi2(k)."""
    if x <= 0:
        return 1.0
    return 1.0 - _regularized_gamma_p(k / 2.0, x / 2.0)


def chi2_homogeneity_pvalue(
    counts_a: Dict[int, int], counts_b: Dict[int, int]
) -> float:
    """p-value d'un test du chi2 d'homogeneite entre deux echantillons."""
    n_a = sum(counts_a.values())
    n_b = sum(counts_b.values())
    if n_a == 0 or n_b == 0:
        return 0.0

    chi2 = 0.0
    categories = 0
    for k in sorted(set(counts_a) | set(counts_b)):
        o_a = counts_a.get(k, 0)
        o_b = counts_b.get(k, 0)
        total = o_a + o_b
        if total == 0:
            continue
        e_a = total * n_a / (n_a + n_b)
        e_b = total * n_b / (n_a + n_b)
        if e_a > 0:
            chi2 += (o_a - e_a) ** 2 / e_a
        if e_b > 0:
            chi2 += (o_b - e_b) ** 2 / e_b
        categories += 1
    return _chi2_survival(chi2, max(categories - 1, 1))


def compare_samples(
    a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]
) -> ComparisonResult:
    """Compare deux jeux de mesures."""
    if set(a) != set(b):
        return ComparisonResult(
            Verdict.DIVERGENT,
            f"cles de mesure differentes : {sorted(a)} vs {sorted(b)}",
        )
    if all(
        a[k].shape == b[k].shape
        and a[k].dtype == b[k].dtype
        and a[k].tobytes() == b[k].tobytes()
        for k in a
    ):
        return ComparisonResult(Verdict.BIT_EXACT, "echantillons identiques")

    # Garde du regime clairseme. A verifier AVANT de conclure quoi que ce soit :
    # sans elle le chi2 certifie 'compatibles' deux jeux sans aucun rapport.
    for k in sorted(a):
        comptes_a = bitstring_counts(a[k])
        comptes_b = bitstring_counts(b[k])
        categories = len(set(comptes_a) | set(comptes_b))
        tirages = min(sum(comptes_a.values()), sum(comptes_b.values()))
        if categories == 0:
            continue
        attendu_moyen = tirages / categories
        if attendu_moyen < MIN_EXPECTED_COUNT:
            requis = int(categories * MIN_EXPECTED_COUNT)
            return ComparisonResult(
                Verdict.INDETERMINATE,
                f"cle {k!r} : {categories} bitstrings distincts pour {tirages} "
                f"tirages (effectif attendu {attendu_moyen:.2f} < "
                f"{MIN_EXPECTED_COUNT:g}). Aucun test statistique ne peut "
                f"conclure dans ce regime ; il faudrait au moins {requis} "
                "tirages. Les echantillons ne sont PAS identiques.",
            )

    p_min, cle_min = 2.0, sorted(a)[0] if a else ""
    for k in sorted(a):
        p = chi2_homogeneity_pvalue(bitstring_counts(a[k]), bitstring_counts(b[k]))
        if p < p_min:
            p_min, cle_min = p, k
    if p_min >= CHI2_ALPHA:
        return ComparisonResult(
            Verdict.STATISTICALLY_COMPATIBLE,
            f"chi2 p={p_min:.4f} >= {CHI2_ALPHA} (cle la plus faible : {cle_min!r})",
            p_value=p_min,
        )
    return ComparisonResult(
        Verdict.DIVERGENT,
        f"chi2 p={p_min:.2e} < {CHI2_ALPHA} sur la cle {cle_min!r}",
        p_value=p_min,
    )
