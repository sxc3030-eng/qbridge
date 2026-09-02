"""capture() : executer un circuit et sceller tout ce qu'il faut pour le rejouer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import cirq
import numpy as np

from qbridge.backends import BACKENDS, make_backend
from qbridge.digest import canonical_json, sha256_of, sha256_of_array
from qbridge.manifest import Manifest


@dataclass(frozen=True)
class CaptureRun:
    """Le manifeste plus les resultats obtenus lors de la capture initiale."""

    manifest: Manifest
    state_vector: Optional[np.ndarray]
    samples: Optional[Dict[str, np.ndarray]]
    result_hash: str


def hash_samples(samples: Dict[str, np.ndarray]) -> str:
    """Empreinte d'un jeu de mesures, indexee par cle.

    On hache le JSON canonique d'un dictionnaire, PAS une concatenation de
    chaines. La concatenation `f"{k}:{hash}"` sans delimiteur n'etait pas
    injective, et la collision se construit en trois lignes :

        A = {"m": a, "x": b}          ->  "m:" + H1 + "x:" + H2
        B = {f"m:{H1}x": b}           ->  "m:H1x" + ":" + H2

    Les deux produisaient exactement la meme chaine, donc le meme result_hash
    pour deux archives de contenus differents. Les cles de mesure sont des
    chaines arbitraires : rien n'interdisait d'en fabriquer une contenant `:`
    et de l'hexadecimal. Le JSON canonique echappe et delimite, ce qui rend la
    representation non ambigue.
    """
    return sha256_of({k: sha256_of_array(v) for k, v in samples.items()})


def capture(
    circuit: cirq.Circuit,
    *,
    backend: Any = "qsim",
    seed: int,
    repetitions: Optional[int] = None,
    options: Optional[Dict[str, Any]] = None,
    noise: Optional[cirq.NoiseModel] = None,
    classical: Optional[Any] = None,
    calibration: Optional[Any] = None,
) -> CaptureRun:
    """Execute `circuit` et renvoie un `CaptureRun` scelle.

    Si `repetitions` est fourni on echantillonne ; sinon on calcule le vecteur
    d'etat complet.

    `classical` accepte un `ClassicalContext` (voir `capture_classical`) : le
    code qui a bati le circuit et celui qui reduira les tirages, plus
    l'environnement Python epingle. C'est ce qui complete la garantie
    archivistique — sans lui, on archive des bitstrings que plus personne ne
    saura interpreter.

    `calibration` accepte un `CalibrationSnapshot` : l'etat DATE de l'appareil.
    Obligatoire pour un backend materiel, refuse pour les simulateurs qui ne
    s'en servent pas.
    """
    options = dict(options or {})
    # Un backend qui exige des identifiants ou une connexion ne peut pas etre
    # construit depuis une simple chaine : on accepte alors l'objet deja
    # construit. C'est le cas d'IbmRuntimeBackend, volontairement absent du
    # registre pour cette raison.
    if isinstance(backend, str):
        impl = make_backend(backend, calibration)
    else:
        impl = backend
        if calibration is not None:
            raise ValueError(
                f"Le backend {getattr(impl, 'name', impl)!r} porte son propre "
                "etat d'appareil ; lui passer une calibration separee "
                "laisserait croire qu'elle a influence le resultat."
            )

    # L'EXECUTION D'ABORD, LE MANIFESTE ENSUITE. Le placement des qubits
    # logiques vers les qubits physiques n'existe qu'une fois la transpilation
    # faite ; l'etat d'appareil a sceller en depend. Construire le manifeste
    # avant reviendrait a sceller un appareil dont on ignore quelle partie a
    # servi.
    vecteur = None
    samples = None
    if repetitions is None:
        vecteur = impl.simulate(circuit, seed=seed, options=options, noise=noise)
    else:
        samples = impl.sample(
            circuit, repetitions=repetitions, seed=seed, options=options, noise=noise
        )

    avertissements: list = []
    if calibration is None:
        obtenir = getattr(impl, "device_calibration", None)
        if callable(obtenir):
            calibration, avertissements = obtenir()

    provenance = None
    decrire = getattr(impl, "device_provenance", None)
    if callable(decrire):
        trace = decrire()
        if trace is not None:
            provenance = canonical_json(trace)

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
        calibration_json=(
            calibration.to_json() if calibration is not None else None
        ),
        # DEFAUT 27 : ces avertissements etaient calcules puis jetes. Ils
        # disent ce qui a ete converti, restreint, ou carrement RATE — dont
        # « etat d'appareil NON scelle » quand l'extraction echoue.
        calibration_warnings=avertissements,
        # `capture()` scelle ce qui SORT de la machine. Un post-traitement se
        # fait apres, sur l'archive, et doit alors le declarer.
        samples_are_raw=True,
        device_provenance_json=provenance,
        kernel=None if getattr(impl, "USES_QSIM_KERNEL", True) else {},
    )

    if repetitions is None:
        return CaptureRun(manifest, vecteur, None, sha256_of_array(vecteur))
    return CaptureRun(manifest, None, samples, hash_samples(samples))
