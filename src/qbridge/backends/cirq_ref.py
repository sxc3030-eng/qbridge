"""Backend de reference : le simulateur natif de Cirq.

Sert d'oracle independant. Il est lent, mais il ne partage aucune ligne de code
avec qsim : si les deux concordent, la probabilite d'un bug commun est faible.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import cirq
import numpy as np


class CirqReferenceBackend:
    """Enveloppe `cirq.Simulator`. Instance fraiche a chaque appel."""

    name = "cirq-reference"
    BIT_EXACT_REPLAYABLE = True
    """Attribut de CLASSE : lisible sans construire d'instance.

    Le plafond de verdict doit pouvoir interroger le backend de CAPTURE,
    qu'on ne peut pas toujours instancier (le backend materiel exige une
    calibration). Construire un temoin dans un try/except revenait a
    desactiver le plafond en silence des que la construction echouait."""

    def __init__(self) -> None:
        self.version = cirq.__version__

    @staticmethod
    def _rejeter_les_options(options: Dict[str, Any]) -> None:
        if options:
            raise ValueError(
                f"Le backend {CirqReferenceBackend.name} n'accepte aucune option "
                f"d'execution ; recu : {sorted(options)}"
            )

    def simulate(
        self,
        circuit: cirq.Circuit,
        *,
        seed: int,
        options: Dict[str, Any],
        noise: Optional[cirq.NoiseModel] = None,
    ) -> np.ndarray:
        self._rejeter_les_options(options)
        sim = cirq.Simulator(seed=seed, noise=noise)
        return sim.simulate(circuit).state_vector()

    def sample(
        self,
        circuit: cirq.Circuit,
        *,
        repetitions: int,
        seed: int,
        options: Dict[str, Any],
        noise: Optional[cirq.NoiseModel] = None,
    ) -> Dict[str, np.ndarray]:
        self._rejeter_les_options(options)
        sim = cirq.Simulator(seed=seed, noise=noise)
        return dict(sim.run(circuit, repetitions=repetitions).measurements)

    def is_bit_exact_replayable(self) -> bool:
        # Lit l'attribut de classe : instance et classe ne peuvent pas
        # diverger, il n'y a qu'une source de verite.
        return type(self).BIT_EXACT_REPLAYABLE

