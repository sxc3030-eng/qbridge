"""Le manifeste : la recette scellee d'une execution.

Le manifeste ne contient JAMAIS d'etat quantique. Le no-cloning l'interdit sur
materiel reel, et accepter un etat depuis un simulateur creerait une API qui ne
peut pas survivre au passage au materiel. On ne scelle que la recette.

Le `semantic_hash` couvre : le circuit, le seed, le mode, les options de niveau
SEMANTIC et NUMERIC pour ce mode, et le noyau SIMD. Il EXCLUT les options de
niveau PERFORMANCE et le reste de l'environnement — parce qu'il est mesure
qu'elles ne changent pas le resultat.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import cirq

from qbridge.digest import sha256_of, sha256_of_text
from qbridge.fingerprint import environment_fingerprint, kernel_fingerprint
from qbridge.modes import ExecutionMode, detect_mode
from qbridge.tiers import Tier, split_options

MANIFEST_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class Manifest:
    """Description complete et rejouable d'une execution de circuit."""

    schema_version: str
    created_at: str
    circuit_json: str
    circuit_hash: str
    backend_name: str
    backend_version: str
    mode: str
    seed: int
    repetitions: Optional[int]
    noise_json: Optional[str]
    semantic_options: Dict[str, Any]
    numeric_options: Dict[str, Any]
    performance_options: Dict[str, Any]
    kernel: Dict[str, Any]
    environment: Dict[str, Any]
    semantic_hash: str = field(default="")

    @classmethod
    def build(
        cls,
        *,
        circuit: cirq.Circuit,
        backend_name: str,
        backend_version: str,
        seed: Optional[int],
        repetitions: Optional[int],
        options: Dict[str, Any],
        noise_json: Optional[str],
    ) -> "Manifest":
        if seed is None:
            raise ValueError(
                "Un seed explicite est obligatoire : sans lui l'execution n'est "
                "pas reproductible et le manifeste serait mensonger."
            )
        mode = detect_mode(circuit, repetitions=repetitions)
        parts = split_options(options, mode)
        circuit_json = cirq.to_json(circuit)
        brut = cls(
            schema_version=MANIFEST_SCHEMA_VERSION,
            created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
            circuit_json=circuit_json,
            circuit_hash=sha256_of_text(circuit_json),
            backend_name=backend_name,
            backend_version=backend_version,
            mode=mode.value,
            seed=int(seed),
            repetitions=repetitions,
            noise_json=noise_json,
            semantic_options=parts[Tier.SEMANTIC],
            numeric_options=parts[Tier.NUMERIC],
            performance_options=parts[Tier.PERFORMANCE],
            kernel=kernel_fingerprint(),
            environment=environment_fingerprint(),
        )
        return cls(**{**brut.__dict__, "semantic_hash": brut._compute_semantic_hash()})

    def _compute_semantic_hash(self) -> str:
        """Hash de tout ce qui influence le resultat."""
        return sha256_of(
            {
                "schema_version": self.schema_version,
                "circuit_hash": self.circuit_hash,
                "backend_name": self.backend_name,
                "mode": self.mode,
                "seed": self.seed,
                "repetitions": self.repetitions,
                "noise_json": self.noise_json,
                "semantic_options": self.semantic_options,
                "numeric_options": self.numeric_options,
                "kernel": self.kernel,
            }
        )

    def execution_mode(self) -> ExecutionMode:
        return ExecutionMode(self.mode)

    def circuit(self) -> cirq.Circuit:
        """Reconstruit le circuit depuis le JSON scelle."""
        return cirq.read_json(json_text=self.circuit_json)

    def noise(self) -> Optional[cirq.NoiseModel]:
        if self.noise_json is None:
            return None
        return cirq.read_json(json_text=self.noise_json)

    def all_options(self) -> Dict[str, Any]:
        return {
            **self.semantic_options,
            **self.numeric_options,
            **self.performance_options,
        }

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Manifest":
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
