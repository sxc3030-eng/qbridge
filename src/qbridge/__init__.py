"""qbridge — harnais de capture/replay pour executions de circuits quantiques.

IMPORTS PARESSEUX, ET POURQUOI. Ce module ne charge rien au moment de
`import qbridge`. Tout passe par `__getattr__` (PEP 562), qui importe le
sous-module concerne au premier acces.

La raison n'est pas la performance. `cli.py` prend soin de ne faire aucun import
du domaine au niveau du module, precisement pour que `qbridge --help` puisse
fonctionner sans cirq ni qsimcirq. Mais le point d'entree passe par le PAQUET :
`import qbridge.cli` importait d'abord `qbridge/__init__.py`, qui tirait
`backends`, donc `cirq`, donc `qsimcirq`. La precaution prise dans `cli.py`
etait annulee par ce fichier, et son docstring affirmait le contraire de ce qui
se passait reellement.

`from qbridge import capture` continue de fonctionner : `from X import Y`
consulte `__getattr__` comme n'importe quel acces d'attribut.
"""

from typing import Any

__version__ = "0.8.3"

# nom public -> (sous-module, nom dans ce sous-module)
_EXPORTS = {
    "capture": ("qbridge.capture", "capture"),
    "CaptureRun": ("qbridge.capture", "CaptureRun"),
    "replay": ("qbridge.replay", "replay"),
    "replay_record": ("qbridge.replay", "replay_record"),
    "verify_archival": ("qbridge.replay", "verify_archival"),
    "ReplayReport": ("qbridge.replay", "ReplayReport"),
    "ArchivalReport": ("qbridge.replay", "ArchivalReport"),
    "Manifest": ("qbridge.manifest", "Manifest"),
    "RunRecord": ("qbridge.record", "RunRecord"),
    "Verdict": ("qbridge.verdict", "Verdict"),
    "Tier": ("qbridge.tiers", "Tier"),
    "ExecutionMode": ("qbridge.modes", "ExecutionMode"),
    "capture_classical": ("qbridge.classical", "capture_classical"),
    "verify_source_unchanged": ("qbridge.classical", "verify_source_unchanged"),
    "ClassicalContext": ("qbridge.classical", "ClassicalContext"),
    "CallableCapture": ("qbridge.classical", "CallableCapture"),
    "SourceDriftReport": ("qbridge.classical", "SourceDriftReport"),
    "Evidence": ("qbridge.classical", "Evidence"),
    "sign_manifest": ("qbridge.signing", "sign_manifest"),
    "sign_record": ("qbridge.signing", "sign_record"),
    "verify_manifest_signature": ("qbridge.signing", "verify_manifest_signature"),
    "verify_record_signature": ("qbridge.signing", "verify_record_signature"),
    "Signature": ("qbridge.signing", "Signature"),
    "SignatureAlgorithm": ("qbridge.signing", "SignatureAlgorithm"),
    "SignatureScope": ("qbridge.signing", "SignatureScope"),
    "SignatureReport": ("qbridge.signing", "SignatureReport"),
    "HmacSigner": ("qbridge.signing", "HmacSigner"),
    "Ed25519Signer": ("qbridge.signing", "Ed25519Signer"),
    "Ed25519Verifier": ("qbridge.signing", "Ed25519Verifier"),
    "CalibrationSnapshot": ("qbridge.calibration", "CalibrationSnapshot"),
    "DatedValue": ("qbridge.calibration", "DatedValue"),
    "synthetic_snapshot": ("qbridge.calibration", "synthetic_snapshot"),
    "SimulatedHardwareBackend": ("qbridge.backends", "SimulatedHardwareBackend"),
    "make_backend": ("qbridge.backends", "make_backend"),
}

__all__ = [*sorted(_EXPORTS), "__version__"]


def __getattr__(name: str) -> Any:
    """Importe a la demande. Voir la note du module."""
    cible = _EXPORTS.get(name)
    if cible is None:
        raise AttributeError(f"module {__name__!r} n'a pas d'attribut {name!r}")
    import importlib

    module, attribut = cible
    valeur = getattr(importlib.import_module(module), attribut)
    globals()[name] = valeur  # memorise : un seul import par nom
    return valeur


def __dir__() -> list:
    return sorted(__all__)
