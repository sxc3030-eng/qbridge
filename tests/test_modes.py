import cirq
import pytest

from qbridge.modes import ExecutionMode, detect_mode


def _q(n):
    return cirq.LineQubit.range(n)


def test_sans_repetitions_c_est_le_vecteur_detat():
    c = cirq.Circuit(cirq.H(_q(1)[0]))
    assert detect_mode(c, repetitions=None) is ExecutionMode.STATE_VECTOR


def test_mesures_terminales_c_est_l_echantillonnage_terminal():
    q = _q(2)
    c = cirq.Circuit(cirq.H(q[0]), cirq.CX(q[0], q[1]), cirq.measure(*q, key="m"))
    assert detect_mode(c, repetitions=100) is ExecutionMode.TERMINAL_SAMPLING


def test_mesures_intermediaires_c_est_le_mode_midcircuit():
    q = _q(1)
    c = cirq.Circuit(
        cirq.H(q[0]),
        cirq.measure(q[0], key="a"),
        cirq.H(q[0]),
        cirq.measure(q[0], key="b"),
    )
    assert detect_mode(c, repetitions=100) is ExecutionMode.MIDCIRCUIT_SAMPLING


def test_une_seule_repetition_bascule_en_midcircuit():
    # qsim change de chemin C++ quand repetitions == 1 : on refuse de le traiter
    # comme un echantillonnage terminal, meme si les mesures le sont.
    q = _q(2)
    c = cirq.Circuit(cirq.H(q[0]), cirq.measure(*q, key="m"))
    assert detect_mode(c, repetitions=1) is ExecutionMode.MIDCIRCUIT_SAMPLING


def test_repetitions_nulle_ou_negative_est_refusee():
    q = _q(1)
    c = cirq.Circuit(cirq.H(q[0]), cirq.measure(q[0], key="m"))
    with pytest.raises(ValueError, match="repetitions"):
        detect_mode(c, repetitions=0)


def test_les_modes_sont_serialisables_en_chaine():
    assert ExecutionMode.STATE_VECTOR.value == "state_vector"
    assert ExecutionMode("state_vector") is ExecutionMode.STATE_VECTOR
