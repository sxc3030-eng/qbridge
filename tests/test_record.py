import json

import cirq
import numpy as np
import pytest

from qbridge import capture, replay_record, verify_archival
from qbridge.record import RunRecord
from qbridge.verdict import Verdict


def _bell_mesure():
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1), cirq.measure(q0, q1, key="m"))


def _bell():
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1))


def _record(**kw):
    return RunRecord.from_capture(capture(_bell_mesure(), seed=7, repetitions=200, **kw))


def test_from_capture_conserve_les_bitstrings():
    r = _record()
    assert r.samples is not None and r.samples["m"].shape == (200, 2)
    assert r.state_vector_hash is None


def test_mode_vecteur_detat_ne_garde_que_le_hash():
    # 2^n * 8 octets : stocker le vecteur devient absurde des 30 qubits.
    r = RunRecord.from_capture(capture(_bell(), seed=7))
    assert r.samples is None
    assert r.state_vector_hash is not None and len(r.state_vector_hash) == 64


def test_round_trip_sur_disque(tmp_path):
    r = _record()
    r.save(tmp_path / "run")
    relu = RunRecord.load(tmp_path / "run")
    assert relu.result_hash == r.result_hash
    assert relu.manifest.semantic_hash == r.manifest.semantic_hash
    assert relu.samples["m"].tobytes() == r.samples["m"].tobytes()


def test_verify_integrity_accepte_un_dossier_sain():
    _record().verify_integrity()  # ne doit pas lever


def test_verify_integrity_detecte_des_bitstrings_falsifies():
    r = _record()
    falsifie = {k: v.copy() for k, v in r.samples.items()}
    falsifie["m"][0][0] ^= 1  # un seul bit change
    altere = RunRecord(
        schema_version=r.schema_version,
        manifest=r.manifest,
        result_hash=r.result_hash,
        samples=falsifie,
        state_vector_hash=r.state_vector_hash,
    )
    with pytest.raises(ValueError, match="bitstrings"):
        altere.verify_integrity()


def test_verify_integrity_detecte_un_manifeste_falsifie(tmp_path):
    r = _record()
    r.save(tmp_path / "run")
    chemin = tmp_path / "run" / "manifest.json"
    data = json.loads(chemin.read_text(encoding="utf-8"))
    data["seed"] = 999
    chemin.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="manifeste"):
        RunRecord.load(tmp_path / "run").verify_integrity()


def test_les_comptages_sont_derives_jamais_stockes():
    r = _record()
    comptes = r.bitstring_counts("m")
    assert sum(comptes.values()) == 200
    # Etat de Bell : seuls |00> et |11> apparaissent.
    assert set(comptes) <= {0b00, 0b11}


def test_comptage_sur_une_cle_inconnue_leve():
    with pytest.raises(KeyError, match="inconnue"):
        _record().bitstring_counts("pas_une_cle")


def test_comptage_impossible_sans_bitstrings():
    r = RunRecord.from_capture(capture(_bell(), seed=7))
    with pytest.raises(ValueError, match="bitstrings"):
        r.bitstring_counts("m")


# ---------- rejeu archivistique : zero ressource quantique ----------


def test_verify_archival_valide_un_dossier_sain(tmp_path):
    r = _record()
    r.save(tmp_path / "run")
    rapport = verify_archival(RunRecord.load(tmp_path / "run"))
    assert rapport.manifest_intact and rapport.results_intact
    assert rapport.measurement_keys == ["m"]
    assert rapport.total_shots == 200


def test_verify_archival_signale_une_falsification_sans_lever():
    r = _record()
    altere = RunRecord(
        schema_version=r.schema_version,
        manifest=r.manifest,
        result_hash="0" * 64,
        samples=r.samples,
        state_vector_hash=r.state_vector_hash,
    )
    rapport = verify_archival(altere)
    assert rapport.manifest_intact is False and rapport.results_intact is False


def test_verify_archival_n_execute_aucun_circuit(monkeypatch, tmp_path):
    """La garantie archivistique doit tenir SANS simulateur.

    On casse volontairement les deux backends : si `verify_archival` en
    touchait un, le test exploserait.
    """
    r = _record()
    r.save(tmp_path / "run")

    import qbridge.backends.cirq_ref as cr
    import qbridge.backends.qsim as qs

    def interdit(*a, **k):
        raise AssertionError("verify_archival a execute un circuit — il ne doit pas")

    monkeypatch.setattr(qs.QsimBackend, "sample", interdit)
    monkeypatch.setattr(qs.QsimBackend, "simulate", interdit)
    monkeypatch.setattr(cr.CirqReferenceBackend, "sample", interdit)
    monkeypatch.setattr(cr.CirqReferenceBackend, "simulate", interdit)

    rapport = verify_archival(RunRecord.load(tmp_path / "run"))
    assert rapport.results_intact


# ---------- rejeu compare a l'archive, pas a une re-execution ----------


def test_replay_record_compare_a_l_archive(tmp_path):
    r = _record()
    r.save(tmp_path / "run")
    assert replay_record(RunRecord.load(tmp_path / "run")).verdict is Verdict.BIT_EXACT


def test_replay_record_detecte_une_archive_divergente():
    """Le point de `replay_record` : la reference vient du disque, donc un
    resultat archive different est detecte — ce que `replay(manifest)` ne peut
    pas faire puisqu'il re-execute sa propre reference."""
    r = _record()
    rng = np.random.default_rng(0)
    faux = {"m": rng.integers(0, 2, size=r.samples["m"].shape, dtype=r.samples["m"].dtype)}
    from qbridge.capture import hash_samples

    divergent = RunRecord(
        schema_version=r.schema_version,
        manifest=r.manifest,
        result_hash=hash_samples(faux),  # coherent avec les faux tirages
        samples=faux,
        state_vector_hash=r.state_vector_hash,
    )
    assert replay_record(divergent).verdict is Verdict.DIVERGENT


def test_replay_record_en_mode_vecteur_detat_compare_les_hashes():
    r = RunRecord.from_capture(capture(_bell(), seed=7))
    assert replay_record(r).verdict is Verdict.BIT_EXACT


def test_replay_record_refuse_une_archive_falsifiee():
    r = _record()
    altere = RunRecord(
        schema_version=r.schema_version,
        manifest=r.manifest,
        result_hash="0" * 64,
        samples=r.samples,
        state_vector_hash=r.state_vector_hash,
    )
    with pytest.raises(ValueError, match="bitstrings"):
        replay_record(altere)
