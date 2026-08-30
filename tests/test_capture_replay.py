import json

import cirq
import pytest

from qbridge import capture, replay
from qbridge.manifest import Manifest
from qbridge.verdict import Verdict


def _bell():
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1))


def _bell_mesure():
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1), cirq.measure(q0, q1, key="m"))


def _mid():
    """Circuit a mesure VRAIMENT intermediaire — voir la note dans test_manifest."""
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


def test_capture_puis_replay_donne_bit_exact():
    run = capture(_bell(), backend="qsim", seed=7)
    assert replay(run.manifest).verdict is Verdict.BIT_EXACT


def test_capture_avec_mesures_donne_bit_exact():
    run = capture(_bell_mesure(), backend="qsim", seed=7, repetitions=100)
    assert replay(run.manifest).verdict is Verdict.BIT_EXACT


def test_replay_survit_a_un_changement_de_threads_en_vecteur_detat():
    run = capture(_bell(), backend="qsim", seed=7, options={"cpu_threads": 1})
    assert (
        replay(run.manifest, override_performance={"cpu_threads": 8}).verdict
        is Verdict.BIT_EXACT
    )


def test_replay_refuse_de_changer_les_threads_en_midcircuit():
    run = capture(
        _mid(), backend="qsim", seed=7, repetitions=50, options={"cpu_threads": 1}
    )
    with pytest.raises(ValueError, match="PERFORMANCE"):
        replay(run.manifest, override_performance={"cpu_threads": 8})


def test_replay_sur_l_oracle_cirq_reste_au_moins_numeriquement_equivalent():
    run = capture(_bell(), backend="qsim", seed=7)
    assert (
        replay(run.manifest, backend="cirq-reference").verdict
        <= Verdict.NUMERICALLY_EQUIVALENT
    )


def test_replay_detecte_un_manifeste_altere(tmp_path):
    run = capture(_bell(), backend="qsim", seed=7)
    chemin = tmp_path / "m.json"
    run.manifest.save(chemin)
    data = json.loads(chemin.read_text(encoding="utf-8"))
    data["seed"] = 999
    chemin.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="integrite"):
        replay(Manifest.load(chemin))


def test_replay_depuis_un_fichier(tmp_path):
    run = capture(_bell(), backend="qsim", seed=7)
    chemin = tmp_path / "m.json"
    run.manifest.save(chemin)
    assert replay(Manifest.load(chemin)).verdict is Verdict.BIT_EXACT


def test_capture_refuse_un_backend_inconnu():
    with pytest.raises(KeyError, match="inconnu"):
        capture(_bell(), backend="backend_imaginaire", seed=7)


def test_capture_expose_un_hash_de_resultat():
    a = capture(_bell(), backend="qsim", seed=7)
    b = capture(_bell(), backend="qsim", seed=7)
    assert a.result_hash == b.result_hash


def test_le_bruit_est_scelle_et_reellement_applique():
    # Regression : capture() acceptait `noise` et l'inscrivait au manifeste sans
    # jamais l'appliquer au backend. Un manifeste mensonger est pire que pas de
    # manifeste.
    propre = capture(_bell_mesure(), backend="qsim", seed=7, repetitions=400)
    bruite = capture(
        _bell_mesure(),
        backend="qsim",
        seed=7,
        repetitions=400,
        noise=cirq.depolarize(0.2),
    )
    assert bruite.manifest.noise_json is not None
    assert bruite.result_hash != propre.result_hash
    assert bruite.manifest.semantic_hash != propre.manifest.semantic_hash


def test_replay_reapplique_le_bruit_scelle():
    run = capture(
        _bell_mesure(),
        backend="qsim",
        seed=7,
        repetitions=200,
        noise=cirq.depolarize(0.1),
    )
    assert replay(run.manifest).verdict is Verdict.BIT_EXACT
