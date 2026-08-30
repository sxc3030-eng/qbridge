import cirq
import numpy as np
import pytest

from qbridge.backends.cirq_ref import CirqReferenceBackend


@pytest.fixture
def bell():
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1))


@pytest.fixture
def bell_mesure():
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1), cirq.measure(q0, q1, key="m"))


def test_nom_et_version():
    b = CirqReferenceBackend()
    assert b.name == "cirq-reference"
    assert b.version


def test_simulate_renvoie_un_vecteur_detat(bell):
    sv = CirqReferenceBackend().simulate(bell, seed=7, options={})
    assert sv.shape == (4,)
    assert np.isclose(abs(sv[0]) ** 2, 0.5) and np.isclose(abs(sv[3]) ** 2, 0.5)


def test_simulate_deterministe_a_seed_fixe(bell):
    b = CirqReferenceBackend()
    assert (
        b.simulate(bell, seed=7, options={}).tobytes()
        == b.simulate(bell, seed=7, options={}).tobytes()
    )


def test_sample_deterministe_a_seed_fixe(bell_mesure):
    b = CirqReferenceBackend()
    s1 = b.sample(bell_mesure, repetitions=50, seed=7, options={})
    s2 = b.sample(bell_mesure, repetitions=50, seed=7, options={})
    assert s1["m"].tobytes() == s2["m"].tobytes()


def test_sample_change_avec_le_seed(bell_mesure):
    b = CirqReferenceBackend()
    s1 = b.sample(bell_mesure, repetitions=200, seed=7, options={})
    s2 = b.sample(bell_mesure, repetitions=200, seed=8, options={})
    assert s1["m"].tobytes() != s2["m"].tobytes()


def test_refuse_toute_option(bell):
    with pytest.raises(ValueError, match="n'accepte aucune option"):
        CirqReferenceBackend().simulate(bell, seed=7, options={"cpu_threads": 4})


def test_est_rejouable_bit_pour_bit():
    assert CirqReferenceBackend().is_bit_exact_replayable() is True
