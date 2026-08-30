import cirq
import numpy as np
import pytest

from qbridge.backends.cirq_ref import CirqReferenceBackend
from qbridge.backends.qsim import QsimBackend


@pytest.fixture
def bell():
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1))


@pytest.fixture
def bell_mesure():
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1), cirq.measure(q0, q1, key="m"))


def test_nom_et_version():
    b = QsimBackend()
    assert b.name == "qsim" and b.version


def test_concorde_avec_l_oracle_cirq(bell):
    a = QsimBackend().simulate(bell, seed=7, options={})
    b = CirqReferenceBackend().simulate(bell, seed=7, options={})
    assert np.allclose(a, b, atol=1e-6)


def test_simulate_deterministe(bell):
    b = QsimBackend()
    o = {"cpu_threads": 4, "max_fused_gate_size": 2}
    assert (
        b.simulate(bell, seed=7, options=o).tobytes()
        == b.simulate(bell, seed=7, options=o).tobytes()
    )


def test_appels_repetes_sur_la_meme_instance_restent_reproductibles(bell_mesure):
    # QSimSimulator est a etat : son _prng avance. Le backend doit construire
    # une instance fraiche par appel, sinon deux appels identiques divergent.
    b = QsimBackend()
    s1 = b.sample(bell_mesure, repetitions=100, seed=7, options={})
    s2 = b.sample(bell_mesure, repetitions=100, seed=7, options={})
    assert (
        s1["m"].tobytes() == s2["m"].tobytes()
    ), "Le backend reutilise un simulateur a etat : le rejeu est casse."


def test_sample_change_avec_le_seed(bell_mesure):
    b = QsimBackend()
    s1 = b.sample(bell_mesure, repetitions=200, seed=7, options={})
    s2 = b.sample(bell_mesure, repetitions=200, seed=8, options={})
    assert s1["m"].tobytes() != s2["m"].tobytes()


def test_refuse_une_option_inconnue(bell):
    with pytest.raises(KeyError, match="inconnue"):
        QsimBackend().simulate(bell, seed=7, options={"pas_une_option": 1})


def test_accepte_un_modele_de_bruit(bell_mesure):
    b = QsimBackend()
    r = b.sample(
        bell_mesure, repetitions=20, seed=7, options={}, noise=cirq.depolarize(0.05)
    )
    assert r["m"].shape == (20, 2)


def test_le_bruit_change_le_resultat(bell_mesure):
    b = QsimBackend()
    propre = b.sample(bell_mesure, repetitions=400, seed=7, options={})
    bruite = b.sample(
        bell_mesure, repetitions=400, seed=7, options={}, noise=cirq.depolarize(0.2)
    )
    assert propre["m"].tobytes() != bruite["m"].tobytes()


def test_est_rejouable_bit_pour_bit():
    assert QsimBackend().is_bit_exact_replayable() is True
