"""qbridge — harnais de capture/replay pour executions de circuits quantiques.

IMPORTS EAGER, ET POURQUOI PAS PARESSEUX. Une version de ce fichier utilisait
`__getattr__` (PEP 562) pour n'importer qu'a la demande, afin que
`qbridge --help` puisse fonctionner sans cirq ni qsimcirq.

Elle etait FAUSSE, et le piege merite d'etre ecrit ici pour que personne ne le
retente : plusieurs noms publics collisionnent avec des sous-modules —
`qbridge.capture` est a la fois la fonction et le module, idem pour `replay`,
`signing`, `record`, `calibration`. Avec des imports paresseux, un
`import qbridge.replay` fait quelque part dans l'arbre REMPLACE la fonction mise
en cache par le module, et le gagnant depend de l'ORDRE DES IMPORTS. Symptome
observe : `TypeError: 'module' object is not callable` sur 21 tests, alors que
chacun passait isole.

En eager, l'import de la fonction s'execute en dernier et gagne toujours. Le
comportement redevient deterministe.

CONSEQUENCE ASSUMEE : `import qbridge` tire cirq et qsimcirq, donc
`qbridge --help` en a besoin, contrairement a ce que le docstring de `cli.py` a
affirme un temps. Le vrai correctif serait de renommer les modules
d'implementation (`_replay.py`, `_capture.py`...) pour supprimer la collision.
C'est un renommage large, et il n'a pas ete fait ici.
"""

from qbridge.backends import SimulatedHardwareBackend, make_backend
from qbridge.calibration import CalibrationSnapshot, DatedValue, synthetic_snapshot
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
from qbridge.journal import Journal, JournalEntry, JournalReport
from qbridge.plausibility import (
    Plausibility,
    PlausibilityReport,
    verify_physical_plausibility,
)
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
    SignatureScope,
    sign_manifest,
    sign_record,
    verify_manifest_signature,
    verify_record_signature,
)
from qbridge.tiers import Tier
from qbridge.verdict import Verdict

__version__ = "0.12.0"
__all__ = [
    "capture",
    "replay",
    "replay_record",
    "verify_archival",
    "verify_physical_plausibility",
    "Journal",
    "JournalEntry",
    "JournalReport",
    "Plausibility",
    "PlausibilityReport",
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
    "sign_record",
    "verify_manifest_signature",
    "verify_record_signature",
    "Signature",
    "SignatureAlgorithm",
    "SignatureScope",
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
