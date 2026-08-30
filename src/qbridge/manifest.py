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

import copy
import datetime as _dt
import json
from dataclasses import dataclass, field, fields
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

    def verify_self(self) -> None:
        """Verifie la coherence interne du manifeste. Leve ValueError sinon.

        Trois controles, chacun couvrant une faille reelle :

        1. `circuit_hash` doit correspondre a `circuit_json`. Sans ce controle,
           `circuit_hash` n'est que decoratif : on peut remplacer le circuit en
           entier, le hash semantique reste valide (il ne couvre que
           `circuit_hash`, pas le JSON), et le rejeu certifie BIT_EXACT un
           circuit qui n'a rien a voir.
        2. Chaque option doit se trouver dans le bon seau pour le mode declare.
           Sans ce controle, `performance_options` — exclu du hash — devient une
           porte derobee : y ecrire `cpu_threads` en mode midcircuit contourne
           silencieusement le verrouillage que `override_performance` refuse
           bruyamment.
        3. Le hash semantique doit correspondre au contenu.
        """
        recalcule = sha256_of_text(self.circuit_json)
        if recalcule != self.circuit_hash:
            raise ValueError(
                "Echec du controle d'integrite du circuit : le circuit serialise "
                "ne correspond pas a son empreinte. "
                f"circuit_hash={self.circuit_hash[:16]}... "
                f"recalcule={recalcule[:16]}..."
            )

        mode = ExecutionMode(self.mode)
        attendu = split_options(self.all_options(), mode)
        for niveau, observe in (
            (Tier.SEMANTIC, self.semantic_options),
            (Tier.NUMERIC, self.numeric_options),
            (Tier.PERFORMANCE, self.performance_options),
        ):
            if dict(observe) != attendu[niveau]:
                raise ValueError(
                    f"Echec du controle d'integrite des options : en mode "
                    f"{self.mode}, le seau {niveau.value} devrait contenir "
                    f"{sorted(attendu[niveau])} mais contient {sorted(observe)}. "
                    "Une option a ete deplacee de seau, ce qui contournerait le "
                    "verrouillage par niveau."
                )

        attendu_hash = self._compute_semantic_hash()
        if attendu_hash != self.semantic_hash:
            raise ValueError(
                "Echec du controle d'integrite du manifeste : il a ete modifie. "
                f"hash stocke={self.semantic_hash[:16]}... "
                f"hash recalcule={attendu_hash[:16]}..."
            )

    def execution_mode(self) -> ExecutionMode:
        mode = ExecutionMode(self.mode)
        if mode is ExecutionMode.EXPECTATION:
            raise NotImplementedError(
                "Le mode EXPECTATION est classe dans OPTION_TIERS mais n'est pas "
                "implemente dans capture()/replay(). Un manifeste qui le declare "
                "ne peut pas etre rejoue correctement : replay l'executerait par "
                "le chemin d'echantillonnage tout en lui appliquant les niveaux "
                "d'EXPECTATION."
            )
        return mode

    def circuit(self) -> cirq.Circuit:
        """Reconstruit le circuit depuis le JSON scelle."""
        return cirq.read_json(json_text=self.circuit_json)

    def noise(self) -> Optional[cirq.NoiseModel]:
        if self.noise_json is None:
            return None
        return cirq.read_json(json_text=self.noise_json)

    def all_options(self) -> Dict[str, Any]:
        """Reunit les trois seaux.

        Ordre de precedence : PERFORMANCE d'abord, donc SEMANTIC et NUMERIC
        l'emportent. L'inverse permettait a une option ecrite dans le seau
        PERFORMANCE — le seul qui ne soit pas couvert par le hash — d'ecraser
        sa valeur verrouillee.
        """
        return {
            **self.performance_options,
            **self.numeric_options,
            **self.semantic_options,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Copie PROFONDE. Une copie de surface laisserait muter les dicts
        internes d'une dataclass pourtant `frozen`, ce qui invalide le hash
        depuis l'exterieur sans que rien ne le signale."""
        return copy.deepcopy(dict(self.__dict__))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Manifest":
        version = data.get("schema_version")
        if version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Version de schema de manifeste incompatible : {version!r} "
                f"(attendu {MANIFEST_SCHEMA_VERSION!r})."
            )
        connus = {f.name for f in fields(cls)}
        inconnus = set(data) - connus
        if inconnus:
            raise ValueError(
                f"Champs inconnus dans le manifeste : {sorted(inconnus)}"
            )
        manquants = connus - set(data) - {"semantic_hash"}
        if manquants:
            raise ValueError(
                f"Champs absents du manifeste : {sorted(manquants)}"
            )
        return cls(**copy.deepcopy(data))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
