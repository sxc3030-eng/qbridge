"""L'enregistrement d'execution : le manifeste PLUS ce qui en est sorti.

Distinction volontaire avec `Manifest` :

- Le `Manifest` est la RECETTE. Il ne contient aucun resultat. C'est lui qu'on
  transmet pour demander a quelqu'un de refaire l'experience.
- Le `RunRecord` est la recette PLUS le resultat obtenu. C'est lui qu'on archive.

Les bitstrings bruts sont la seule donnee non regenerable de toute la chaine :
c'est le seul enregistrement physique de l'evenement quantique. Tout le reste
du dossier existe pour les rendre interpretables. On les stocke donc tels
quels, en entier, jamais agreges — les agregats se recalculent, les tirages
non.

Le vecteur d'etat, lui, n'est PAS stocke : 2^n * 8 octets devient absurde des
30 qubits (8.6 Go). On en garde le hash, ce qui suffit a detecter une derive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from qbridge.capture import CaptureRun, hash_samples
from qbridge.digest import sha256_of_array
from qbridge.manifest import Manifest

RECORD_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class RunRecord:
    """Un manifeste et le resultat qu'il a produit."""

    schema_version: str
    manifest: Manifest
    result_hash: str
    samples: Optional[Dict[str, np.ndarray]]
    state_vector_hash: Optional[str]

    @classmethod
    def from_capture(cls, run: CaptureRun) -> "RunRecord":
        return cls(
            schema_version=RECORD_SCHEMA_VERSION,
            manifest=run.manifest,
            result_hash=run.result_hash,
            samples=run.samples,
            state_vector_hash=(
                sha256_of_array(run.state_vector)
                if run.state_vector is not None
                else None
            ),
        )

    def verify_integrity(self) -> None:
        """Verifie que le dossier est coherent avec lui-meme.

        Ne consomme AUCUNE ressource quantique : on recalcule le hash a partir
        des octets stockes et on le compare a celui qui a ete scelle.
        """
        attendu = self.manifest._compute_semantic_hash()
        if attendu != self.manifest.semantic_hash:
            raise ValueError(
                "Echec du controle d'integrite du manifeste : il a ete modifie. "
                f"hash stocke={self.manifest.semantic_hash[:16]}... "
                f"hash recalcule={attendu[:16]}..."
            )
        if self.samples is not None:
            recalcule = hash_samples(self.samples)
            if recalcule != self.result_hash:
                raise ValueError(
                    "Echec du controle d'integrite des resultats : les bitstrings "
                    f"stockes ne correspondent pas au hash scelle. "
                    f"stocke={self.result_hash[:16]}... "
                    f"recalcule={recalcule[:16]}..."
                )

    def bitstring_counts(self, key: str) -> Dict[int, int]:
        """Comptage des bitstrings pour une cle de mesure.

        C'est un agregat DERIVE : il se recalcule toujours depuis `samples`, il
        n'est jamais stocke. Stocker un agregat a cote de sa source, c'est
        creer deux verites qui peuvent diverger.
        """
        if self.samples is None:
            raise ValueError(
                "Cet enregistrement ne contient pas de bitstrings "
                f"(mode {self.manifest.mode}) : rien a compter."
            )
        if key not in self.samples:
            raise KeyError(
                f"Cle de mesure inconnue : {key!r}. "
                f"Disponibles : {sorted(self.samples)}"
            )
        from qbridge.verdict import bitstring_counts

        return bitstring_counts(self.samples[key])

    # ---------- persistance ----------

    def save(self, directory: str | Path) -> Path:
        """Ecrit le dossier : manifest.json + samples.npz + record.json."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.manifest.save(d / "manifest.json")

        entete: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "result_hash": self.result_hash,
            "state_vector_hash": self.state_vector_hash,
            "has_samples": self.samples is not None,
            "measurement_keys": sorted(self.samples) if self.samples else [],
        }
        (d / "record.json").write_text(
            json.dumps(entete, indent=2, sort_keys=True), encoding="utf-8"
        )
        if self.samples is not None:
            np.savez_compressed(d / "samples.npz", **self.samples)
        return d

    @classmethod
    def load(cls, directory: str | Path) -> "RunRecord":
        d = Path(directory)
        entete = json.loads((d / "record.json").read_text(encoding="utf-8"))
        samples: Optional[Dict[str, np.ndarray]] = None
        if entete["has_samples"]:
            with np.load(d / "samples.npz") as z:
                samples = {k: z[k] for k in z.files}
        return cls(
            schema_version=entete["schema_version"],
            manifest=Manifest.load(d / "manifest.json"),
            result_hash=entete["result_hash"],
            samples=samples,
            state_vector_hash=entete["state_vector_hash"],
        )
