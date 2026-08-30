import cirq
import pytest

from qbridge.manifest import MANIFEST_SCHEMA_VERSION, Manifest
from qbridge.modes import ExecutionMode


def _bell():
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1))


def _mid():
    """Circuit a mesure VRAIMENT intermediaire.

    Piege : `H(q0), measure(q0), H(q1), measure(q1)` n'en est PAS un — cirq
    empaquette les deux H dans un moment et les deux mesures dans le suivant,
    donc les deux mesures sont terminales. Il faut une operation qui suive une
    mesure SUR LE MEME QUBIT.
    """
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit(
        cirq.H(q[0]),
        cirq.CX(q[0], q[1]),
        cirq.measure(q[0], key="a"),
        cirq.X(q[0]) ** 0.5,
        cirq.measure(q[0], q[1], key="b"),
    )
    assert not c.are_all_measurements_terminal()
    return c


def _build(circuit=None, seed=7, repetitions=None, options=None):
    return Manifest.build(
        circuit=circuit if circuit is not None else _bell(),
        backend_name="qsim",
        backend_version="0.22.0",
        seed=seed,
        repetitions=repetitions,
        options=options or {},
        noise_json=None,
    )


def test_porte_une_version_de_schema():
    assert _build().schema_version == MANIFEST_SCHEMA_VERSION


def test_enregistre_le_mode_d_execution():
    assert _build().mode == ExecutionMode.STATE_VECTOR.value
    q0, q1 = cirq.LineQubit.range(2)
    terminal = cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1), cirq.measure(q0, q1, key="m"))
    assert (
        _build(circuit=terminal, repetitions=100).mode
        == ExecutionMode.TERMINAL_SAMPLING.value
    )


def test_options_reparties_par_niveau_en_mode_vecteur_detat():
    m = _build(options={"cpu_threads": 8, "max_fused_gate_size": 3})
    assert m.performance_options == {"cpu_threads": 8}
    assert m.numeric_options == {"max_fused_gate_size": 3}


def test_cpu_threads_devient_semantique_en_midcircuit():
    m = _build(circuit=_mid(), repetitions=100, options={"cpu_threads": 8})
    assert m.semantic_options == {"cpu_threads": 8}
    assert m.performance_options == {}


def test_hash_semantique_ignore_les_options_de_performance():
    a = _build(options={"cpu_threads": 8, "max_fused_gate_size": 3})
    b = _build(options={"cpu_threads": 1, "max_fused_gate_size": 3})
    assert a.semantic_hash == b.semantic_hash


def test_hash_semantique_prend_en_compte_cpu_threads_en_midcircuit():
    a = _build(circuit=_mid(), repetitions=100, options={"cpu_threads": 8})
    b = _build(circuit=_mid(), repetitions=100, options={"cpu_threads": 1})
    assert a.semantic_hash != b.semantic_hash


def test_hash_semantique_change_avec_les_options_numeriques():
    assert (
        _build(options={"max_fused_gate_size": 2}).semantic_hash
        != _build(options={"max_fused_gate_size": 3}).semantic_hash
    )


def test_hash_semantique_change_avec_le_seed():
    assert _build(seed=7).semantic_hash != _build(seed=8).semantic_hash


def test_hash_semantique_inclut_le_noyau_simd():
    m = _build()
    altere = Manifest.from_dict(
        {**m.to_dict(), "kernel": {**m.kernel, "qsim_instruction_set": 3}}
    )
    assert altere._compute_semantic_hash() != m.semantic_hash


def test_round_trip_json(tmp_path):
    m = _build(options={"cpu_threads": 4})
    chemin = tmp_path / "m.json"
    m.save(chemin)
    relu = Manifest.load(chemin)
    assert relu.semantic_hash == m.semantic_hash
    assert relu.circuit() == m.circuit()
    assert relu.environment == m.environment
    assert relu.kernel == m.kernel


def test_circuit_reconstruit_a_l_identique():
    m = _build()
    assert Manifest.from_dict(m.to_dict()).circuit() == m.circuit()


def test_seed_nul_refuse():
    with pytest.raises(ValueError, match="seed"):
        _build(seed=None)
