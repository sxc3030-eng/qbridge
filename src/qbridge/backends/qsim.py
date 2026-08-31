"""Backend qsim.

Deux exigences non negociables :

1. Une instance `QSimSimulator` FRAICHE par execution. Son `_prng` avance a
   chaque appel — qsim le teste explicitement en amont
   (`test_sampling_nondeterminism`). Reutiliser une instance casserait le rejeu.
2. Toute option est validee contre la table des niveaux. Une option inconnue
   leve KeyError plutot que d'etre silencieusement ignoree : une option ignoree
   est exactement le genre de derive qui rend un rejeu faussement rassurant.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import cirq
import numpy as np
import qsimcirq

from qbridge.tiers import known_options


class QsimBackend:
    """Enveloppe `qsimcirq.QSimSimulator`."""

    name = "qsim"
    BIT_EXACT_REPLAYABLE = True
    """Attribut de CLASSE : lisible sans construire d'instance.

    Le plafond de verdict doit pouvoir interroger le backend de CAPTURE,
    qu'on ne peut pas toujours instancier (le backend materiel exige une
    calibration). Construire un temoin dans un try/except revenait a
    desactiver le plafond en silence des que la construction echouait."""

    def __init__(self) -> None:
        self.version = qsimcirq.__version__

    @staticmethod
    def _valider(options: Dict[str, Any]) -> qsimcirq.QSimOptions:
        connues = known_options()
        for cle in options:
            if cle not in connues:
                raise KeyError(
                    f"Option qsim inconnue : {cle!r}. "
                    f"Options classees : {sorted(connues)}"
                )
        return qsimcirq.QSimOptions(**options)

    def _simulateur(
        self,
        options: Dict[str, Any],
        seed: int,
        noise: Optional[cirq.NoiseModel],
    ) -> qsimcirq.QSimSimulator:
        """Toujours une instance neuve — voir la note 1 du module."""
        return qsimcirq.QSimSimulator(self._valider(options), seed=seed, noise=noise)

    def simulate(
        self,
        circuit: cirq.Circuit,
        *,
        seed: int,
        options: Dict[str, Any],
        noise: Optional[cirq.NoiseModel] = None,
    ) -> np.ndarray:
        sim = self._simulateur(options, seed, noise)
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
        sim = self._simulateur(options, seed, noise)
        return dict(sim.run(circuit, repetitions=repetitions).measurements)

    def is_bit_exact_replayable(self) -> bool:
        # Lit l'attribut de classe : instance et classe ne peuvent pas
        # diverger, il n'y a qu'une source de verite.
        return type(self).BIT_EXACT_REPLAYABLE

