"""Classification (option x mode) -> niveau d'influence sur le resultat.

CETTE TABLE EST EMPIRIQUE. Mesuree le 2026-08-30 avec cirq 1.7.0 /
qsimcirq 0.22.0 sur qsim_avx2, OpenMP actif (x1.86 a 25 qubits).
`tests/test_determinism_boundary.py` la re-mesure et echouera si qsim change.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Set

from qbridge.modes import ExecutionMode


class Tier(str, Enum):
    SEMANTIC = "semantic"
    """Change le resultat qualitativement. Identique obligatoire pour tout rejeu."""

    NUMERIC = "numeric"
    """Ne change le resultat qu'au niveau de l'arrondi flottant. Identique
    obligatoire pour un rejeu BIT_EXACT ; peut varier pour NUMERICALLY_EQUIVALENT."""

    PERFORMANCE = "performance"
    """Sans effet mesurable sur le resultat. Libre de varier d'une machine a l'autre."""


_TOUS_LES_MODES = tuple(ExecutionMode)

# Chaque entree : nom d'option -> {mode: niveau}.
OPTION_TIERS: Dict[str, Dict[ExecutionMode, Tier]] = {
    # MESURE : bit-identique t=1..16 en vecteur d'etat (jusqu'a 25 qubits) et en
    # echantillonnage terminal. MESURE DIFFERENT en midcircuit a 20 qubits :
    # VirtualMeasure lit un vecteur de normes partielles de longueur num_threads.
    # En mode expectation, RunReduce partitionne par thread : NUMERIC par prudence
    # (empiriquement stable a 20 qubits, mais le source montre la dependance).
    "cpu_threads": {
        ExecutionMode.STATE_VECTOR: Tier.PERFORMANCE,
        ExecutionMode.TERMINAL_SAMPLING: Tier.PERFORMANCE,
        ExecutionMode.MIDCIRCUIT_SAMPLING: Tier.SEMANTIC,
        ExecutionMode.EXPECTATION: Tier.NUMERIC,
    },
    # MESURE : f>=3 change le vecteur d'etat (max|delta| 3.2e-9, infidelite 1.4e-5).
    "max_fused_gate_size": {m: Tier.NUMERIC for m in _TOUS_LES_MODES},
    # FTZ/DAZ : sans effet sur les circuits testes, mais modifie la semantique
    # flottante en presence de denormaux. NUMERIC par prudence.
    "denormals_are_zeros": {m: Tier.NUMERIC for m in _TOUS_LES_MODES},
    # Noyaux CUDA entierement distincts des noyaux AVX.
    "use_gpu": {m: Tier.NUMERIC for m in _TOUS_LES_MODES},
    "gpu_mode": {m: Tier.NUMERIC for m in _TOUS_LES_MODES},
    # Reductions CUDA non tracees : ne pas supposer neutres.
    "gpu_state_threads": {m: Tier.NUMERIC for m in _TOUS_LES_MODES},
    "gpu_data_blocks": {m: Tier.NUMERIC for m in _TOUS_LES_MODES},
    # Nombre de trajectoires moyennees : change la valeur estimee.
    "ev_noisy_repetitions": {m: Tier.SEMANTIC for m in _TOUS_LES_MODES},
    # Journalisation seulement. La seule option franchement neutre.
    "verbosity": {m: Tier.PERFORMANCE for m in _TOUS_LES_MODES},
}


def known_options() -> Set[str]:
    """Noms d'options couverts par la table."""
    return set(OPTION_TIERS)


def tier_of(option_name: str, mode: ExecutionMode) -> Tier:
    """Niveau d'une option pour un mode donne. Leve KeyError si inconnue."""
    try:
        par_mode = OPTION_TIERS[option_name]
    except KeyError:
        raise KeyError(
            f"Option qsim inconnue : {option_name!r}. "
            f"Options classees : {sorted(OPTION_TIERS)}"
        ) from None
    return par_mode[mode]


def split_options(
    options: Dict[str, Any], mode: ExecutionMode
) -> Dict[Tier, Dict[str, Any]]:
    """Repartit un dict d'options en trois dicts, un par niveau, pour ce mode."""
    parts: Dict[Tier, Dict[str, Any]] = {t: {} for t in Tier}
    for name, value in options.items():
        parts[tier_of(name, mode)][name] = value
    return parts

def option_types() -> Dict[str, type]:
    """Type attendu de chaque option, lu depuis `QSimOptions` a l'execution.

    Derive de la dataclass plutot qu'ecrit a la main : une option dont le type
    change chez qsim ne peut pas se desynchroniser d'une table locale.
    """
    import dataclasses

    import qsimcirq

    types: Dict[str, type] = {}
    for champ in dataclasses.fields(qsimcirq.QSimOptions):
        annotation = champ.type
        if isinstance(annotation, str):
            annotation = {"int": int, "bool": bool, "float": float}.get(annotation)
        if isinstance(annotation, type):
            types[champ.name] = annotation
    return types


def check_option_value(name: str, value: Any) -> None:
    """Refuse une valeur du mauvais type pour cette option.

    `--option cpu_threads=off` produisait `False`, scelle tel quel dans le
    manifeste et passe a qsim comme zero thread. Le nom etait valide, la valeur
    ne l'etait pas, et rien ne la regardait. Le principe affiche du projet — une
    option acceptee en silence rend un rejeu faussement rassurant — vaut pour
    les valeurs autant que pour les noms.

    Note : `bool` est sous-classe de `int` en Python, donc un booleen passerait
    un test naif `isinstance(value, int)`. On le rejette explicitement.
    """
    attendu = option_types().get(name)
    if attendu is None:
        return
    if attendu is bool:
        if not isinstance(value, bool):
            raise ValueError(
                f"L'option {name!r} attend un booleen, recu {value!r} "
                f"({type(value).__name__})."
            )
        return
    if isinstance(value, bool) or not isinstance(value, attendu):
        raise ValueError(
            f"L'option {name!r} attend un {attendu.__name__}, recu {value!r} "
            f"({type(value).__name__})."
        )
    if attendu is int and value < 0:
        raise ValueError(
            f"L'option {name!r} attend un entier positif, recu {value!r}."
        )
