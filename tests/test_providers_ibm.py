"""Ingestion de calibrations IBM REELLES.

Le `fake_provider` de qiskit-ibm-runtime embarque 69 instantanes de vrais
appareils. Contrairement a Google, IBM date CHAQUE parametre — ce qui permet de
verifier pour de vrai la fonctionnalite pour laquelle `DatedValue` a ete concu.
"""

from __future__ import annotations

import pytest

cirq = pytest.importorskip("cirq")
pytest.importorskip("qiskit_ibm_runtime")

from qbridge import capture, replay  # noqa: E402
from qbridge.calibration import CalibrationSnapshot  # noqa: E402
from qbridge.providers import backends_disponibles, from_ibm_backend  # noqa: E402
from qbridge.record import RunRecord  # noqa: E402
from qbridge.verdict import Verdict, bitstring_counts  # noqa: E402


@pytest.fixture(scope="module")
def fez():
    return from_ibm_backend("FakeFez", qubits=range(6))


def test_les_69_backends_sont_listes():
    noms = backends_disponibles()
    assert len(noms) >= 60
    assert "FakeFez" in noms


def test_un_backend_inconnu_est_refuse():
    with pytest.raises(ValueError, match="Backend IBM inconnu"):
        from_ibm_backend("FakeMachineImaginaire")


def test_le_sous_ensemble_de_qubits_est_respecte(fez):
    instantane, _ = fez
    assert len(instantane.qubits) == 6
    assert set(instantane.qubits) == {f"q({i})" for i in range(6)}


def test_restreindre_les_qubits_est_SIGNALE(fez):
    """Sceller une partie de l'appareil est legitime, mais le taire laisserait
    croire que l'instantane decrit la machine entiere."""
    _, avertissements = fez
    joint = " ".join(avertissements)
    assert "restreint aux qubits" in joint
    assert "ne sont PAS scelles" in joint


def test_les_portes_hors_sous_ensemble_sont_comptees(fez):
    _, avertissements = fez
    assert any("portes ecartees" in a for a in avertissements)


# ---------- les conversions d'unites, qui se trompent en silence ----------


def test_T1_est_converti_en_microsecondes(fez):
    """IBM publie T1 en SECONDES : 4.88e-05 signifie 48.8 us. Une erreur de
    facteur produirait un modele de bruit absurde sans rien signaler."""
    instantane, avertissements = fez
    for nom, params in instantane.qubits.items():
        t1 = params["t1_us"].value
        assert 1.0 < t1 < 5000.0, f"{nom} : T1={t1} us, unite probablement fausse"
    assert any("convertis de secondes" in a for a in avertissements)


def test_les_erreurs_restent_des_probabilites(fez):
    instantane, _ = fez
    for nom, params in instantane.qubits.items():
        for cle in ("readout_error", "prob_meas0_prep1", "prob_meas1_prep0"):
            if cle in params:
                v = params[cle].value
                assert 0.0 <= v <= 1.0, f"{nom}.{cle} = {v}"


def test_les_durees_de_porte_viennent_bien_d_IBM(fez):
    """Contrairement a Google, IBM PUBLIE les durees de porte. Elles ne doivent
    donc pas porter la marque « fourni par l'appelant »."""
    instantane, _ = fez
    durees = [
        p["gate_length_ns"]
        for p in instantane.gates.values()
        if "gate_length_ns" in p
    ]
    assert durees, "aucune duree de porte lue"
    for d in durees:
        assert "ABSENT de la calibration" not in d.unit
        assert d.value >= 0
    # Une duree NULLE est une donnee legitime : `rz` est une rotation virtuelle
    # sur ce materiel, implementee comme un changement de phase de reference.
    assert any(d.value > 0 for d in durees)


def test_la_duree_utilisee_est_celle_de_la_PORTE(fez):
    """Mesure sur ibm_fez : x/sx/rx durent 24 ns, cz 84 ns, rz 0 ns, et reset
    1584 ns. La moyenne globale vaut 210 ns — ecrasee par le reset. L'utiliser
    pour convertir T1 en relaxation appliquait NEUF FOIS trop de bruit a une
    porte X."""
    instantane, _ = fez
    q = cirq.LineQubit.range(2)

    x = instantane.gate_length_for(cirq.X(q[0]))
    cz = instantane.gate_length_for(cirq.CZ(q[0], q[1]))
    moyenne = instantane.mean_gate_length_ns()

    assert x == pytest.approx(24.0), f"X dure {x} ns"
    assert cz == pytest.approx(84.0), f"CZ dure {cz} ns"
    assert x < moyenne / 5, (
        "la moyenne globale n'est pas representative d'une porte : c'est "
        "precisement pourquoi elle ne doit plus servir"
    )


def test_une_duree_nulle_ne_produit_aucune_relaxation(fez):
    """`rz` virtuelle : duree nulle, donc aucune relaxation. Un repli sur la
    moyenne lui en aurait applique."""
    instantane, _ = fez
    modele = instantane.noise_model()
    q0 = cirq.LineQubit(0)
    assert modele._damping(q0, 0.0) == 0.0
    assert modele._damping(q0, 24.0) > 0.0


# ---------- LE point : les dates par parametre ----------


def test_chaque_parametre_porte_une_date_DIFFERENTE(fez):
    """C'est la raison d'etre de `DatedValue`. Sur ibm_fez, T1 est mesure le
    26 fevrier et readout_error le 24 : deux jours d'ecart dans un meme
    « instantane »."""
    instantane, _ = fez
    params = instantane.qubits["q(0)"]
    assert params["t1_us"].date != params["readout_error"].date


def test_l_etalement_temporel_est_reel_et_expose(fez):
    """Un instantane IBM n'est PAS l'etat de l'appareil a un instant."""
    instantane, avertissements = fez
    etalement = instantane.temporal_spread_seconds()
    assert etalement > 3600, "aucun etalement : les dates ont ete perdues"
    assert any("s'etalent sur" in a for a in avertissements)


def test_l_etalement_croit_avec_le_nombre_de_qubits_scelles():
    """Verification que l'etalement mesure ce qu'il pretend : plus on scelle de
    qubits, plus on capture de dates differentes. Sur l'appareil entier
    (156 qubits) l'ecart atteint deux mois."""
    petit, _ = from_ibm_backend("FakeFez", qubits=range(4))
    grand, _ = from_ibm_backend("FakeFez", qubits=range(60))
    assert grand.temporal_spread_seconds() > petit.temporal_spread_seconds()


def test_google_et_IBM_different_bien_sur_ce_point():
    """Google publie un horodatage unique, IBM date chaque parametre. Les deux
    adaptateurs doivent refleter cette difference plutot que de l'uniformiser.
    """
    pytest.importorskip("cirq_google")
    from qbridge.providers import from_google_calibration

    google, _ = from_google_calibration("rainbow")
    ibm, _ = from_ibm_backend("FakeFez", qubits=range(6))

    assert google.temporal_spread_seconds() == 0.0
    assert ibm.temporal_spread_seconds() > 0.0


# ---------- integrite et cycle complet ----------


def test_l_instantane_ibm_se_verifie(fez):
    instantane, _ = fez
    instantane.verify_self()


def test_aller_retour_json(fez):
    instantane, _ = fez
    relu = CalibrationSnapshot.from_json(instantane.to_json())
    relu.verify_self()
    assert relu.snapshot_hash == instantane.snapshot_hash


def _bell_ibm():
    q = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.H(q[0]), cirq.CX(q[0], q[1]), cirq.measure(*q, key="m"))


def test_le_cycle_complet_tourne_sur_une_calibration_ibm(fez, tmp_path):
    instantane, _ = fez
    run = capture(
        _bell_ibm(),
        backend="hardware-sim",
        seed=7,
        repetitions=600,
        calibration=instantane,
    )
    assert run.manifest.calibration().device_id == "ibm:ibm_fez"

    record = RunRecord.from_capture(run)
    record.save(tmp_path / "ibm")
    RunRecord.load(tmp_path / "ibm").verify_integrity()

    from qbridge import verify_archival

    rapport = verify_archival(RunRecord.load(tmp_path / "ibm"))
    assert rapport.manifest_intact and rapport.results_intact


def test_le_bruit_ibm_degrade_sans_detruire(fez):
    instantane, _ = fez
    run = capture(
        _bell_ibm(),
        backend="hardware-sim",
        seed=7,
        repetitions=600,
        calibration=instantane,
    )
    comptes = bitstring_counts(run.samples["m"])
    domine = comptes.get(0b00, 0) + comptes.get(0b11, 0)
    fidelite = domine / sum(comptes.values())
    assert 0.75 < fidelite < 1.0, f"fidelite {fidelite:.3f} invraisemblable"


def test_le_plafond_tient_sur_une_calibration_ibm(fez):
    instantane, _ = fez
    run = capture(
        _bell_ibm(),
        backend="hardware-sim",
        seed=7,
        repetitions=300,
        calibration=instantane,
    )
    assert replay(run.manifest).verdict is not Verdict.BIT_EXACT


def test_deux_appareils_IBM_ne_sont_pas_la_meme_experience():
    a, _ = from_ibm_backend("FakeFez", qubits=range(2))
    b, _ = from_ibm_backend("FakeManilaV2", qubits=range(2))
    assert a.snapshot_hash != b.snapshot_hash

    circuit = _bell_ibm()
    ma = capture(
        circuit, backend="hardware-sim", seed=7, repetitions=50, calibration=a
    ).manifest
    mb = capture(
        circuit, backend="hardware-sim", seed=7, repetitions=50, calibration=b
    ).manifest
    assert ma.semantic_hash != mb.semantic_hash
