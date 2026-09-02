"""Quatrieme verdict : le resultat est-il PLAUSIBLE pour la machine declaree ?

CE QUE LES TROIS AUTRES VERDICTS NE FONT PAS. Bit-exact prouve que deux
executions donnent les memes octets. Statistique prouve que deux jeux de
tirages viennent de la meme loi. Archivistique prouve que les octets archives
sont bien ceux qui ont ete scelles. Aucun des trois ne regarde la PHYSIQUE :
tous les trois acceptent sans broncher une archive fabriquee de toutes pieces,
scellee proprement, signee dans les regles, et annoncant un GHZ parfait sur une
machine dont la calibration scellee dit qu'elle ne peut pas faire mieux que
97 %.

CE QUE CELUI-CI FAIT. Il confronte les tirages scelles a ce que l'etat
d'appareil scelle PREDIT. C'est un controle de coherence interne a l'archive :
elle porte a la fois l'affirmation et de quoi la refuter.

ZERO RESSOURCE QUANTIQUE. Comme la garantie archivistique, et pour la meme
raison : une preuve qui exige une machine quantique n'est pas une preuve qu'on
peut opposer dans dix ans. Il faut un simulateur pour la distribution ideale,
rien de plus.

MESURE FONDATRICE, sur `ibm_marrakesh` le 2026-09-01 :

    predit par l'etat declare : 97.38 %
    observe sur la machine    : 97.27 % +/- 0.51 %
    ecart                     : 0.2 sigma

SON DOMAINE DE VALIDITE, MESURE ET NON SUPPOSE. Le modele est multiplicatif :
chaque porte reussit independamment. Il ignore les erreurs COHERENTES, qui
s'accumulent en amplitude et croissent en n^2. Sur ibm_marrakesh, il accuse a
tort des 10 portes a deux qubits. Au-dela de 3 % d'infidelite predite, le
verdict rend donc INDETERMINE plutot que de risquer une fausse accusation —
seule la BORNE reste opposable, et elle l'est a toute profondeur.

SEULE LA BORNE ACCUSE. Un resultat MOINS bon que predit ne rend jamais
IMPLAUSIBLE : la calibration publiee est une limite optimiste, et tout ce
qu'elle ne modelise pas ne peut que degrader. Seul un resultat qui DEPASSE ce
que la machine declaree peut produire est impossible. Cette asymetrie vient de
la physique, pas d'un seuil ajuste — et elle a coute deux fausses accusations
avant d'etre comprise.

CE QUE CE VERDICT N'EST PAS. Ce n'est pas une preuve d'authenticite. Un
faussaire qui connait la calibration peut fabriquer des tirages plausibles :
rien ici ne l'en empeche, et la signature detachee reste le seul mecanisme
d'opposabilite. Ce verdict attrape l'incoherence, pas la malveillance
competente. Le dire autrement serait mentir.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MIN_SHOTS = 100
"""En dessous, l'incertitude statistique noie tout ecart interessant."""

POUVOIR_DISCRIMINANT_MIN = 0.5
"""Si le support ideal couvre plus de la moitie des bitstrings possibles, un
resultat totalement depolarise tomberait deja dedans la moitie du temps : le
controle ne distinguerait plus une bonne execution d'un tirage au hasard."""

INFIDELITE_DOMAINE_MAX = 0.03
"""Au-dela, le modele multiplicatif ne peut plus CONFIRMER une coherence.

MESURE SUR ibm_marrakesh, GHZ suivi de paires CNOT.CNOT (l'identite, donc
l'etat ideal ne bouge pas) :

      cz | predit | observe | verdict du modele
       2 | 97.38% |  96.78% | 1.2 sigma  -> correct
      10 | 94.09% |  88.77% | 7.2 sigma  -> FAUSSE ACCUSATION
      34 | 84.87% |  39.26% | 40.7 sigma -> FAUSSE ACCUSATION

Les distributions disent pourquoi. A 34 cz, les etats dominants sont `010`
(33.5 %) et `101` (23.2 %) : tous deux un basculement du qubit 1, celui que les
paires CNOT repetees martelent. C'est une erreur COHERENTE qui s'accumule en
amplitude — elle croit en n^2, pas en n — et `111` oscille (44.9 -> 25.4 -> 5.0
-> 33.9 %), signature d'une rotation qui tourne. Un modele d'erreurs
independantes ne peut pas capturer cela.

Le seuil est EMPIRIQUE et encadre par deux points seulement : 2.6 %
d'infidelite ou le modele tient, 5.9 % ou il a deja lache. Plus de mesures
l'affineraient. En attendant, il est place du cote prudent.
"""

SEUIL_SUPPORT = 1e-9
"""Probabilite ideale en dessous de laquelle un bitstring n'est pas dans le
support. Pas zero : l'arithmetique flottante produit des 1e-17 parasites."""


class Plausibility(IntEnum):
    """Ordonne du plus fort au plus faible, comme `Verdict`.

    `INDETERMINE` est place APRES `IMPLAUSIBLE` pour qu'un test
    `<= TENSION` ne puisse jamais accepter une absence de conclusion.
    """

    PLAUSIBLE = 0
    TENSION = 1
    IMPLAUSIBLE = 2
    INDETERMINE = 3


@dataclass(frozen=True)
class PlausibilityReport:
    verdict: Plausibility
    predicted_fidelity: Optional[float] = None
    observed_weight: Optional[float] = None
    sigma: Optional[float] = None
    upper_bound: Optional[float] = None
    """Poids maximal atteignable sur le support : `F + (1-F) * |support|/2^n`.

    Avec probabilite F le calcul reussit ; sinon le resultat est brouille et
    tombe dans le support par hasard. VALABLE A TOUTE PROFONDEUR, parce que le
    bruit non modelise ne peut que degrader — jamais faire mieux que les
    erreurs declarees. Les six mesures d'`effondrement_en_profondeur.py`, de 2 a
    258 portes cz, la respectent toutes."""
    within_domain: Optional[bool] = None
    calibration_age_hours: Optional[float] = None
    """Age de la mesure de calibration la plus vieille au moment de l'execution.

    AUCUN SEUIL N'EST APPLIQUE, et c'est delibere : la vitesse de derive d'un
    QPU entre deux calibrations n'a pas ete mesuree ici. Inventer un seuil
    reviendrait a fabriquer le chiffre que tout le reste de ce projet refuse de
    fabriquer. La valeur est rapportee pour que le lecteur juge."""
    support_size: Optional[int] = None
    total_bitstrings: Optional[int] = None
    shots: Optional[int] = None
    error_budget: Dict[str, float] = field(default_factory=dict)
    """Part de chaque famille de portes dans le budget d'erreur. Mesure sur
    `ibm_marrakesh` : la LECTURE pesait 73.8 %, contre 20 % pour les portes a
    deux qubits."""
    reason: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.name,
            "predicted_fidelity": self.predicted_fidelity,
            "observed_weight": self.observed_weight,
            "sigma": self.sigma,
            "upper_bound": self.upper_bound,
            "within_domain": self.within_domain,
            "calibration_age_hours": self.calibration_age_hours,
            "support_size": self.support_size,
            "total_bitstrings": self.total_bitstrings,
            "shots": self.shots,
            "error_budget": self.error_budget,
            "reason": self.reason,
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------
# fidelite predite par l'etat d'appareil scelle
# --------------------------------------------------------------------------


def _erreur_moyenne(snapshot, nom_porte: str) -> Optional[float]:
    """Erreur moyenne des portes scellees portant ce nom.

    POURQUOI UNE MOYENNE. La provenance scelle des COMPTES par nom de porte
    (`{"cz": 2}`), pas la liste des paires concernees. La calibration est deja
    restreinte aux qubits reellement utilises, donc la moyenne porte sur les
    bonnes portes. C'est une approximation, elle est declaree comme telle, et
    elle est exacte quand toutes les portes de ce nom ont servi.
    """
    valeurs = [
        params["gate_error"].value
        for cle, params in snapshot.gates.items()
        if cle.split(":")[0] == nom_porte and "gate_error" in params
    ]
    return sum(valeurs) / len(valeurs) if valeurs else None


def _erreur_de_lecture(snapshot) -> Optional[float]:
    """Erreur de lecture moyenne.

    ATTENTION AU DOUBLE COMPTAGE. Sur une vraie machine IBM, `measure` est une
    porte du `target` ET `readout_error` existe par qubit — avec la MEME valeur
    (9.521e-03 pour q(0) de `ibm_marrakesh`). Sur les instantanes factices,
    seul `readout_error` existe. On prend donc la porte si elle est scellee,
    sinon le parametre de qubit, jamais les deux.
    """
    depuis_portes = _erreur_moyenne(snapshot, "measure")
    if depuis_portes is not None:
        return depuis_portes

    valeurs = [
        params["readout_error"].value
        for params in snapshot.qubits.values()
        if "readout_error" in params
    ]
    return sum(valeurs) / len(valeurs) if valeurs else None


def fidelite_predite(
    snapshot,
    gate_counts: Dict[str, int],
    operations: Optional[Dict[str, int]] = None,
) -> Tuple[float, Dict[str, float], List[str]]:
    """Probabilite qu'AUCUNE erreur ne survienne, selon l'etat scelle.

    Rend `(fidelite, budget_par_famille, avertissements)`.

    Modele multiplicatif : chaque operation reussit independamment. C'est une
    approximation — les erreurs correlees existent — mais elle a predit 97.38 %
    la ou la machine a rendu 97.27 %.

    DEUX PRECISIONS POSSIBLES. Avec `operations` — les operations exactes par
    qubit physique, scellees dans la provenance — chaque erreur est lue sur la
    porte qui a REELLEMENT servi. Sans, on retombe sur la moyenne par nom de
    porte, ce qui est une approximation declaree : sur `ibm_marrakesh` les
    paires cz vont de 1.65e-3 a 3.63e-3, un facteur 2.2.
    """
    if operations:
        return _fidelite_exacte(snapshot, operations)

    fidelite = 1.0
    budget: Dict[str, float] = {}
    avertissements: List[str] = [
        "operations exactes absentes de la provenance : erreurs moyennees par "
        "nom de porte, ce qui est APPROXIMATIF quand les qubits n'ont pas tous "
        "la meme qualite"
    ]

    for nom, compte in sorted(gate_counts.items()):
        compte = int(compte)
        if compte <= 0:
            continue
        erreur = (
            _erreur_de_lecture(snapshot)
            if nom == "measure"
            else _erreur_moyenne(snapshot, nom)
        )
        if erreur is None:
            avertissements.append(
                f"porte {nom!r} executee {compte} fois mais ABSENTE de la "
                "calibration scellee : son erreur est comptee comme nulle, "
                "donc la fidelite predite est SUREVALUEE"
            )
            continue
        if erreur > 0:
            budget[nom] = erreur * compte
        fidelite *= (1.0 - erreur) ** compte

    total = sum(budget.values())
    if total > 0:
        budget = {k: v / total for k, v in budget.items()}
    return fidelite, budget, avertissements


def _fidelite_exacte(
    snapshot, operations: Dict[str, int]
) -> Tuple[float, Dict[str, float], List[str]]:
    """Fidelite lue sur les portes qui ont REELLEMENT servi.

    Chaque cle de `operations` est deja au format des cles de calibration
    (`"cz:q(0),q(1)"`) : le lookup est direct.
    """
    fidelite = 1.0
    budget: Dict[str, float] = {}
    avertissements: List[str] = []

    for cle, compte in sorted(operations.items()):
        compte = int(compte)
        if compte <= 0:
            continue
        nom = cle.split(":")[0]

        params = snapshot.gates.get(cle)
        erreur = params.get("gate_error").value if params and "gate_error" in params else None

        if erreur is None and nom == "measure":
            # Les instantanes factices ne scellent pas `measure` comme porte :
            # l'erreur de lecture vit alors dans le parametre de qubit.
            qubit = cle.split(":", 1)[1] if ":" in cle else None
            par_qubit = snapshot.qubits.get(qubit or "", {})
            if "readout_error" in par_qubit:
                erreur = par_qubit["readout_error"].value

        if erreur is None:
            erreur = _erreur_moyenne(snapshot, nom)
            if erreur is None:
                avertissements.append(
                    f"operation {cle!r} executee {compte} fois mais ABSENTE de "
                    "la calibration scellee : son erreur est comptee comme "
                    "nulle, donc la fidelite predite est SUREVALUEE"
                )
                continue
            avertissements.append(
                f"{cle!r} absente de la calibration : erreur moyenne des "
                f"portes {nom!r} utilisee a la place"
            )

        if erreur > 0:
            budget[nom] = budget.get(nom, 0.0) + erreur * compte
        fidelite *= (1.0 - erreur) ** compte

    total = sum(budget.values())
    if total > 0:
        budget = {k: v / total for k, v in budget.items()}
    return fidelite, budget, avertissements


# --------------------------------------------------------------------------
# distribution ideale du circuit scelle
# --------------------------------------------------------------------------


def _probabilites_ideales(circuit, cle_mesure: str) -> Optional[np.ndarray]:
    """Loi exacte des bitstrings, sans aucun bruit.

    Le circuit scelle est simule SANS ses mesures ; les qubits non mesures sont
    marginalises. Le vecteur rendu est indexe comme les bitstrings de qbridge :
    le premier qubit mesure est le bit de poids fort.
    """
    import cirq

    mesures = [
        op
        for moment in circuit
        for op in moment
        if cirq.is_measurement(op) and cirq.measurement_key_name(op) == cle_mesure
    ]
    if len(mesures) != 1:
        return None
    qubits_mesures = list(mesures[0].qubits)

    sans_mesure = cirq.Circuit(
        op
        for moment in circuit
        for op in moment
        if not cirq.is_measurement(op)
    )
    autres = [q for q in sorted(circuit.all_qubits()) if q not in qubits_mesures]

    try:
        resultat = cirq.Simulator(seed=0).simulate(
            sans_mesure, qubit_order=qubits_mesures + autres
        )
    except Exception:
        return None

    probas = np.abs(np.asarray(resultat.final_state_vector)) ** 2
    if autres:
        probas = probas.reshape(2 ** len(qubits_mesures), 2 ** len(autres)).sum(axis=1)
    somme = probas.sum()
    return probas / somme if somme > 0 else None


# --------------------------------------------------------------------------
# le verdict
# --------------------------------------------------------------------------


def verify_physical_plausibility(record, *, measurement_key: Optional[str] = None):
    """Les tirages scelles collent-ils a l'etat d'appareil scelle ?

    Ne demande aucune ressource quantique : un simulateur suffit pour la
    distribution ideale, et tout le reste est deja dans l'archive.
    """
    from qbridge.calibration import CalibrationSnapshot

    manifeste = record.manifest
    avertissements: List[str] = []

    def indetermine(raison: str) -> PlausibilityReport:
        return PlausibilityReport(
            verdict=Plausibility.INDETERMINE,
            reason=raison,
            warnings=avertissements,
        )

    if manifeste.calibration_json is None:
        return indetermine(
            "aucun etat d'appareil scelle : rien ne permet de dire ce que "
            "cette machine POUVAIT produire"
        )
    if record.samples is None or not record.samples:
        return indetermine("aucun tirage archive : il n'y a rien a confronter")

    if manifeste.device_provenance_json is None:
        return indetermine(
            "aucune provenance de transpilation scellee : on ignore quelles "
            "portes ont reellement ete executees"
        )
    provenance = json.loads(manifeste.device_provenance_json)
    gate_counts = provenance.get("gate_counts")
    if not gate_counts:
        return indetermine("la provenance scellee ne compte aucune porte")

    cle = measurement_key or sorted(record.samples)[0]
    if len(record.samples) > 1 and measurement_key is None:
        avertissements.append(
            f"plusieurs cles de mesure ({sorted(record.samples)}) : "
            f"controle effectue sur {cle!r} seulement"
        )
    tirages = np.asarray(record.samples[cle])
    n_shots, n_qubits = tirages.shape
    if n_shots < MIN_SHOTS:
        return indetermine(
            f"{n_shots} tirages seulement (minimum {MIN_SHOTS}) : "
            "l'incertitude statistique noierait tout ecart"
        )

    instantane = CalibrationSnapshot.from_json(manifeste.calibration_json)

    # DEFAUT 25. La prediction sort de ces valeurs ; rien ne disait de quand
    # elles dataient. Sur l'archive reelle d'ibm_marrakesh, la plus vieille
    # precedait l'execution de 38.5 heures.
    age_h = None
    age_s = instantane.age_seconds(manifeste.created_at)
    if age_s is not None:
        age_h = age_s / 3600.0
        if age_h < 0:
            avertissements.append(
                f"calibration POSTERIEURE a l'execution de {-age_h:.1f} h : "
                "elle ne peut pas decrire la machine au moment du run"
            )
        elif age_h > 1:
            avertissements.append(
                f"la mesure de calibration la plus vieille precede l'execution "
                f"de {age_h:.1f} h ; aucun seuil n'est applique car la vitesse "
                "de derive de cet appareil n'a pas ete mesuree"
            )
    etalement_h = instantane.temporal_spread_seconds() / 3600.0
    if etalement_h > 1:
        avertissements.append(
            f"les mesures de calibration s'etalent sur {etalement_h:.1f} h : "
            "ce n'est PAS l'etat de l'appareil a un instant"
        )

    predite, budget, avert_fid = fidelite_predite(
        instantane, gate_counts, provenance.get("operations")
    )
    avertissements.extend(avert_fid)

    import cirq

    circuit = cirq.read_json(json_text=manifeste.circuit_json)
    probas = _probabilites_ideales(circuit, cle)
    if probas is None or len(probas) != 2**n_qubits:
        return indetermine(
            "distribution ideale incalculable pour ce circuit : le controle "
            "de plausibilite ne s'applique pas"
        )

    support = np.flatnonzero(probas > SEUIL_SUPPORT)
    total_bitstrings = 2**n_qubits
    pouvoir = 1.0 - len(support) / total_bitstrings
    if pouvoir < POUVOIR_DISCRIMINANT_MIN:
        return indetermine(
            f"le support ideal couvre {len(support)}/{total_bitstrings} "
            "bitstrings : un resultat totalement depolarise y tomberait "
            "presque toujours, le controle ne discriminerait rien"
        )

    poids = 1 << np.arange(n_qubits - 1, -1, -1, dtype=np.int64)
    valeurs = (tirages.astype(np.int64) * poids).sum(axis=1)
    dans_le_support = int(np.isin(valeurs, support).sum())
    observe = dans_le_support / n_shots

    # La variance se calcule sous l'HYPOTHESE NULLE — « la machine declaree a
    # produit ces tirages » — donc sur la fidelite PREDITE, jamais sur la
    # proportion observee. Sur l'observee, un faux annoncant 100 % donnait un
    # ecart-type nul et un sigma de 838 142 : un artefact, pas une mesure.
    p = min(max(predite, 1e-6), 1 - 1e-6)
    ecart_type = math.sqrt(p * (1 - p) / n_shots)
    sigma = abs(predite - observe) / ecart_type

    # LA BORNE, valable a TOUTE profondeur. Avec probabilite F le calcul
    # reussit ; sinon le resultat est brouille et tombe dans le support par
    # hasard. Le bruit non modelise ne peut que degrader : depasser cette borne
    # est impossible, quelle que soit la profondeur.
    hasard = len(support) / total_bitstrings
    borne = predite + (1.0 - predite) * hasard
    depassement = (observe - borne) / ecart_type

    dans_le_domaine = (1.0 - predite) <= INFIDELITE_DOMAINE_MAX

    if depassement > 3:
        verdict, raison = (
            Plausibility.IMPLAUSIBLE,
            f"le resultat archive DEPASSE ce que la machine declaree peut "
            f"produire au mieux ({100 * borne:.2f} %) : impossible, quelle que "
            "soit la profondeur du circuit",
        )
    elif not dans_le_domaine:
        verdict, raison = (
            Plausibility.INDETERMINE,
            f"infidelite predite de {100 * (1 - predite):.1f} %, au-dela du "
            f"domaine ou ce modele est fiable ({100 * INFIDELITE_DOMAINE_MAX:.0f} %). "
            "Mesure sur ibm_marrakesh : des 10 portes a deux qubits, les "
            "erreurs coherentes s'accumulent et le modele accuse a tort. "
            "Seule l'impossibilite a pu etre ecartee, pas la coherence",
        )
    elif sigma < 2:
        verdict, raison = (
            Plausibility.PLAUSIBLE,
            "le resultat archive est coherent avec l'etat d'appareil scelle",
        )
    else:
        # SEULE LA BORNE PEUT ACCUSER. Un resultat MOINS bon que predit n'est
        # jamais une preuve de faux : la calibration publiee est une limite
        # OPTIMISTE, et tout ce qu'elle ne modelise pas — erreurs coherentes,
        # diaphonie, derive depuis la derniere mesure — ne peut que degrader.
        #
        # DEFAUT 26, vecu deux fois en une journee. D'abord en profondeur : le
        # verdict accusait a 40 sigma des archives honnetes. Puis en surface,
        # DANS le domaine declare fiable : IBM a rafraichi les erreurs de
        # lecture d'ibm_marrakesh (0.952 % -> 0.378 % sur q(0), soit 2.5x
        # mieux) SANS re-mesurer T1 et T2, vieux de 43.4 h. La prediction est
        # montee a 98.15 % ; la machine a rendu 96.19 %. Le verdict a crie
        # IMPLAUSIBLE a 4.7 sigma sur une archive produite quatre minutes plus
        # tot par moi-meme.
        #
        # Accuser sur « moins bon que promis » etait structurellement faux, pas
        # mal calibre. Un seuil ne repare pas cela.
        verdict, raison = (
            Plausibility.TENSION,
            f"le resultat archive est nettement en dessous des "
            f"{100 * predite:.2f} % que l'etat scelle laissait esperer "
            f"({sigma:.1f} sigma). Ce n'est PAS une accusation : la calibration "
            "publiee est une limite optimiste, et tout ce qu'elle ne modelise "
            "pas ne peut que degrader",
        )

    return PlausibilityReport(
        verdict=verdict,
        predicted_fidelity=predite,
        observed_weight=observe,
        sigma=sigma,
        upper_bound=borne,
        within_domain=dans_le_domaine,
        calibration_age_hours=age_h,
        support_size=len(support),
        total_bitstrings=total_bitstrings,
        shots=n_shots,
        error_budget=budget,
        reason=raison,
        warnings=avertissements,
    )
