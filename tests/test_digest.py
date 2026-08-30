import numpy as np
import pytest

from qbridge.digest import canonical_json, sha256_of, sha256_of_array, sha256_of_text


def test_canonical_json_insensible_a_l_ordre_des_cles():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_canonical_json_compact_et_trie():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_sha256_stable():
    assert sha256_of({"a": 1}) == sha256_of({"a": 1})


def test_sha256_distingue():
    assert sha256_of({"a": 1}) != sha256_of({"a": 2})


def test_sha256_renvoie_64_hex():
    h = sha256_of({"a": 1})
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)


def test_floats_non_finis_refuses():
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_sha256_of_array_distingue_le_dtype():
    a = np.array([1, 0], dtype=np.complex64)
    b = np.array([1, 0], dtype=np.complex128)
    assert sha256_of_array(a) != sha256_of_array(b)


def test_sha256_of_array_distingue_la_forme():
    a = np.zeros((2, 2), dtype=np.uint8)
    b = np.zeros((4,), dtype=np.uint8)
    assert sha256_of_array(a) != sha256_of_array(b)


def test_sha256_of_text_est_stable():
    assert sha256_of_text("abc") == sha256_of_text("abc")
