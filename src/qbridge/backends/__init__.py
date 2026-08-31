"""Registre des backends.

`make_backend` existe parce que tous les backends ne se construisent pas
pareil : les simulateurs n'ont besoin de rien, le backend materiel exige un
instantane de calibration. Appeler `BACKENDS[nom]()` partout obligerait chaque
appelant a connaitre cette difference — le registre l'absorbe.
"""

from typing import Any, Optional

from qbridge.backends.base import Backend
from qbridge.backends.cirq_ref import CirqReferenceBackend
from qbridge.backends.hardware import SimulatedHardwareBackend
from qbridge.backends.qsim import QsimBackend

BACKENDS = {
    QsimBackend.name: QsimBackend,
    CirqReferenceBackend.name: CirqReferenceBackend,
    SimulatedHardwareBackend.name: SimulatedHardwareBackend,
}

NEEDS_CALIBRATION = frozenset({SimulatedHardwareBackend.name})
"""Backends qui exigent un instantane de calibration pour executer."""


def make_backend(name: str, calibration: Optional[Any] = None) -> Any:
    """Construit un backend par son nom.

    Une calibration passee a un backend qui n'en veut pas est REFUSEE plutot
    qu'ignoree : l'ignorer laisserait croire qu'elle a servi.
    """
    if name not in BACKENDS:
        raise KeyError(f"Backend inconnu : {name!r}. Disponibles : {sorted(BACKENDS)}")
    if name in NEEDS_CALIBRATION:
        return BACKENDS[name](calibration)
    if calibration is not None:
        raise ValueError(
            f"Le backend {name!r} n'utilise pas d'instantane de calibration ; "
            "en fournir un laisserait croire qu'il a influence le resultat."
        )
    return BACKENDS[name]()


__all__ = [
    "Backend",
    "CirqReferenceBackend",
    "QsimBackend",
    "SimulatedHardwareBackend",
    "BACKENDS",
    "NEEDS_CALIBRATION",
    "make_backend",
]
