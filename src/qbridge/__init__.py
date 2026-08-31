"""qbridge — harnais de capture/replay pour executions de circuits quantiques."""

from qbridge.backends import SimulatedHardwareBackend, make_backend
from qbridge.calibration import (
    CalibrationSnapshot,
    DatedValue,
    synthetic_snapshot,
)
from qbridge.capture import CaptureRun, capture
from qbridge.classical import (
    CallableCapture,
    ClassicalContext,
    Evidence,
    SourceDriftReport,
    capture_classical,
    verify_source_unchanged,
)
from qbridge.manifest import Manifest
from qbridge.modes import ExecutionMode
from qbridge.record import RunRecord
from qbridge.replay import (
    ArchivalReport,
    ReplayReport,
    replay,
    replay_record,
    verify_archival,
)
from qbridge.signing import (
    Ed25519Signer,
    Ed25519Verifier,
    HmacSigner,
    Signature,
    SignatureAlgorithm,
    SignatureReport,
    sign_manifest,
    verify_manifest_signature,
)
from qbridge.tiers import Tier
from qbridge.verdict import Verdict

__version__ = "0.8.1"
__all__ = [
    "capture",
    "replay",
    "replay_record",
    "verify_archival",
    "Manifest",
    "RunRecord",
    "CaptureRun",
    "ReplayReport",
    "ArchivalReport",
    "Verdict",
    "Tier",
    "ExecutionMode",
    "capture_classical",
    "verify_source_unchanged",
    "ClassicalContext",
    "CallableCapture",
    "SourceDriftReport",
    "Evidence",
    "sign_manifest",
    "verify_manifest_signature",
    "Signature",
    "SignatureAlgorithm",
    "SignatureReport",
    "HmacSigner",
    "Ed25519Signer",
    "Ed25519Verifier",
    "CalibrationSnapshot",
    "DatedValue",
    "synthetic_snapshot",
    "SimulatedHardwareBackend",
    "make_backend",
    "__version__",
]
