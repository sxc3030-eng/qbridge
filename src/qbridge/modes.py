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


def detect_mode(circuit: cirq.Circuit, *, repetitions: Optional[int]) -> ExecutionMode:
    """Determine le mode d'execution a partir du circuit et des repetitions."""
    if repetitions is None:
        return ExecutionMode.STATE_VECTOR
    if repetitions < 1:
        raise ValueError(f"repetitions doit valoir au moins 1, recu {repetitions}")
    # repetitions == 1 bascule qsim sur la boucle par repetition (chemin B),
    # meme quand toutes les mesures sont terminales. On aligne le mode dessus.
    if repetitions == 1 or not circuit.are_all_measurements_terminal():
        return ExecutionMode.MIDCIRCUIT_SAMPLING
    return ExecutionMode.TERMINAL_SAMPLING
