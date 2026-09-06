"""Modes d'execution.

Le mode n'est pas cosmetique : qsim emprunte des chemins C++ differents selon
le mode, et la neutralite d'une option comme `cpu_threads` en depend
directement. Voir `tiers.py`.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import cirq


class ExecutionMode(str, Enum):
    STATE_VECTOR = "state_vector"
    """simulate() : vecteur d'etat complet, aucune mesure echantillonnee."""

    TERMINAL_SAMPLING = "terminal_sampling"
    """run() avec toutes les mesures terminales et repetitions > 1.
    qsim evolue l'etat une fois puis echantillonne l'etat final."""

    MIDCIRCUIT_SAMPLING = "midcircuit_sampling"
    """run() avec au moins une mesure non terminale, OU repetitions == 1.
    qsim re-execute le circuit par repetition et appelle VirtualMeasure."""

    EXPECTATION = "expectation"
    """simulate_expectation_values() : reduction sur tout le vecteur d'etat."""


def _a_un_controle_classique(circuit: "cirq.Circuit") -> bool:
    """Une porte dont l'execution depend d'une mesure ANTERIEURE ?

    DEFAUT 33. `are_all_measurements_terminal()` de cirq rend True pour un
    circuit a controle classique : chaque mesure EST bien la derniere operation
    sur SON qubit. Mais le resultat de la premiere pilote une porte ulterieure
    sur un AUTRE qubit — c'est une mesure en cours de circuit, quoi qu'en dise
    la question posee a cirq.

    CE QUE CETTE ERREUR COUTAIT. Le mode pilote la table des niveaux :

        cpu_threads en terminal_sampling   -> PERFORMANCE
        cpu_threads en midcircuit_sampling -> SEMANTIC

    Or il est MESURE que `cpu_threads` change les bitstrings en mode
    midcircuit. Classe PERFORMANCE, il sort du hash semantique : deux
    executions rendant des resultats DIFFERENTS auraient porte le meme hash.

    C'est exactement ce que le modele de niveaux existe pour empecher.

    Trouve en faisant tourner un vrai branchement conditionnel sur
    `ibm_marrakesh` : le manifeste scellait `terminal_sampling` pour un circuit
    dont toute la raison d'etre est de mesurer au milieu.
    """
    for moment in circuit:
        for operation in moment:
            if getattr(operation, "classical_controls", None):
                return True
    return False


def detect_mode(circuit: cirq.Circuit, *, repetitions: Optional[int]) -> ExecutionMode:
    """Determine le mode d'execution a partir du circuit et des repetitions."""
    if repetitions is None:
        # `simulate()` sur un circuit qui mesure ECHANTILLONNE bel et bien ces
        # mesures et effondre l'etat : classer ce cas en STATE_VECTOR
        # rendrait `cpu_threads` surchargeable sur un resultat qui depend de
        # tirages. On retombe sur le mode le plus strict.
        if circuit.has_measurements():
            return ExecutionMode.MIDCIRCUIT_SAMPLING
        return ExecutionMode.STATE_VECTOR
    if repetitions < 1:
        raise ValueError(f"repetitions doit valoir au moins 1, recu {repetitions}")
    # repetitions == 1 bascule qsim sur la boucle par repetition (chemin B),
    # meme quand toutes les mesures sont terminales. On aligne le mode dessus.
    if (
        repetitions == 1
        or not circuit.are_all_measurements_terminal()
        or _a_un_controle_classique(circuit)
    ):
        return ExecutionMode.MIDCIRCUIT_SAMPLING
    return ExecutionMode.TERMINAL_SAMPLING
