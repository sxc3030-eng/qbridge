"""Le protocole `Backend` : la frontiere destinee a durer.

Aujourd'hui elle est implementee par des simulateurs. Le jour ou une machine
reelle est disponible, un `HardwareBackend` implemente le meme protocole et le
manifeste ne change pas de forme — seul le verdict de rejeu se degrade de
BIT_EXACT vers STATISTICALLY_COMPATIBLE.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable

import cirq
import numpy as np


@runtime_checkable
class Backend(Protocol):
    """Executeur de circuit. Doit etre sans etat entre deux appels."""

    name: str
    version: str

    def simulate(
        self,
        circuit: cirq.Circuit,
        *,
        seed: int,
        options: Dict[str, Any],
        noise: Optional[cirq.NoiseModel] = None,
    ) -> np.ndarray:
        """Renvoie le vecteur d'etat final complet.

        Un backend materiel levera `NotImplementedError` : on ne peut pas lire
        le vecteur d'etat d'une machine reelle (mesure destructive, no-cloning).
        """
        ...

    def sample(
        self,
        circuit: cirq.Circuit,
        *,
        repetitions: int,
        seed: int,
        options: Dict[str, Any],
        noise: Optional[cirq.NoiseModel] = None,
    ) -> Dict[str, np.ndarray]:
        """Renvoie les mesures, indexees par cle de mesure."""
        ...

    def is_bit_exact_replayable(self) -> bool:
        """Vrai si ce backend peut reproduire un resultat bit-pour-bit.
        Faux pour tout materiel reel : le bruit physique n'est pas rejouable."""
        ...
