"""qbridge — harnais de capture/replay pour executions de circuits quantiques."""

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
from qbridge.tiers import Tier
from qbridge.verdict import Verdict

__version__ = "0.4.0"
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
    "__version__",
]
