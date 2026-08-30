from qbridge.backends.base import Backend
from qbridge.backends.cirq_ref import CirqReferenceBackend
from qbridge.backends.qsim import QsimBackend

BACKENDS = {
    QsimBackend.name: QsimBackend,
    CirqReferenceBackend.name: CirqReferenceBackend,
}

__all__ = ["Backend", "CirqReferenceBackend", "QsimBackend", "BACKENDS"]
