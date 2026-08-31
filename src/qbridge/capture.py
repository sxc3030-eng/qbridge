"""capture() : executer un circuit et sceller tout ce qu'il faut pour le rejouer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import cirq
import numpy as np

from qbridge.backends import BACKENDS
from qbridge.digest import canonical_json, sha256_of_array, sha256_of_text
from qbridge.manifest import Manifest


@dataclass(frozen=True)
class CaptureRun:
    """Le manifeste plus les resultats obtenus lors de la capture initiale."""

    manifest: Manifest
    state_vector: Optional[np.ndarray]
    samples: Optional[Dict[str, np.ndarray]]
    result_hash: str


def hash_samples(samples: Dict[str, np.ndarray]) -> str:
    return sha256_of_text(
        "".join(f"{k}:{sha256_of_array(samples[k])}" for k in sorted(samples))
    )


def capture(
    circuit: cirq.Circuit,
    *,
    backend: str = "qsim",
    seed: int,
    repetitions: Optional[int] = None,
    options: Optional[Dict[str, Any]] = None,
    noise: Optional[cirq.NoiseModel] = None,
    classical: Optional[Any] = None,
) -> CaptureRun:
    """Execute `circuit` et renvoie un `CaptureRun` scelle.

    Si `repetitions` est fourni on echantillonne ; sinon on calcule le vecteur
    d'etat complet.

    `classical` accepte un `ClassicalContext` (voir `capture_classical`) : le
    code qui a bati le circuit et celui qui reduira les tirages, plus
    l'environnement Python epingle. C'est ce qui complete la garantie
    archivistique — sans lui, on archive des bitstrings que plus personne ne
    saura interpreter.
    """
    if backend not in BACKENDS:
        raise KeyError(
            f"Backend inconnu : {backend!r}. Disponibles : {sorted(BACKENDS)}"
        )
    options = dict(options or {})
    impl = BACKENDS[backend]()

    manifest = Manifest.build(
        circuit=circuit,
        backend_name=impl.name,
        backend_version=impl.version,
        seed=seed,
        repetitions=repetitions,
        options=options,
        noise_json=cirq.to_json(noise) if noise is not None else None,
        classical_json=(
            canonical_json(classical.to_dict()) if classical is not None else None
        ),
    )

    if repetitions is None:
        sv = impl.simulate(circuit, seed=seed, options=options, noise=noise)
        return CaptureRun(manifest, sv, None, sha256_of_array(sv))

    samples = impl.sample(
        circuit, repetitions=repetitions, seed=seed, options=options, noise=noise
    )
    return CaptureRun(manifest, None, samples, hash_samples(samples))
