"""JSON canonique et empreintes SHA-256. Aucune dependance au domaine quantique."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def canonical_json(obj: Any) -> str:
    """Serialise en JSON deterministe : cles triees, sans espaces, ASCII.

    `allow_nan=False` fait echouer NaN/Infinity, qui ne sont pas du JSON valide
    et casseraient la stabilite du hash entre implementations.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def sha256_of(obj: Any) -> str:
    """SHA-256 hexadecimal de la forme canonique de `obj`."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_of_text(text: str) -> str:
    """SHA-256 d'une chaine deja serialisee (ex. sortie de cirq.to_json)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_of_array(array: np.ndarray) -> str:
    """SHA-256 des octets bruts d'un tableau, dtype et forme inclus.

    On hache les octets et non une representation texte : c'est la seule facon
    de detecter une difference bit-pour-bit dans un vecteur d'etat complex64.
    """
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("utf-8"))
    h.update(str(a.shape).encode("utf-8"))
    h.update(a.tobytes())
    return h.hexdigest()
