"""Le backend materiel met le protocole a l'epreuve.

Ces tests ne verifient pas une physique : ils verifient que le manifeste
survit a un backend qui viole les trois hypotheses confortables du simulateur —
pas de vecteur d'etat lisible, pas de reproductibilite bit-a-bit, et un etat
d'appareil date dont depend le resultat.

Si le protocole tenait uniquement avec des simulateurs, il ne tiendrait pas le
jour ou une vraie machine arrive. C'est ce fichier qui repond a la question.
"""

from __future__ import annotations

import cirq
import pytest

from qbridge import Verdict, capture, replay, replay_record
from qbridge.backends import NEEDS_CALIBRATION, make_backend
from qbridge.backends.hardware import SimulatedHardwareBackend
from qbridge.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    CalibrationSnapshot,
    DatedValue,
    synthetic_snapshot,
)
from qbridge.manifest import Manifest
from qbridge.record import RunRecord
from qbridge.verdict import bitstring_counts


def _qubits():
    return tuple(cirq.LineQubit.range(3))


def _ghz():
    q = _qubits()
    return cirq.Circuit(
        [cirq.H(q[0]), cirq.CX(q[0], q[1]), cirq.CX(q[1], q[2]), cirq.measure(*q, key="m")]
    )


@pytest.fixture
def snapshot():
    return synthetic_snapshot(_qubits())


@pytest.fixture
def run(snapshot):
    return capture(
        _ghz(), backend="hardware-sim", seed=7, repetitions=400, calibration=snapshot
    )


# ---------- contrainte 1 : pas de vecteur d'etat ----------


def test_simulate_est_impossible_sur_materiel(snapshot):
    """Le no-cloning interdit de copier un etat et la mesure est destructive.
    Rendre un tableau approximatif serait pire que refuser."""
    backend = SimulatedHardwareBackend(snapshot)
    with pytest.raises(NotImplementedError, match="no-cloning"):
        backend.simulate(_ghz(), seed=7, options={})


def test_capture_sans_repetitions_echoue_sur_materiel(snapshot):
    with pytest.raises(NotImplementedError, match="vecteur d'etat"):
        capture(_ghz(), backend="hardware-sim", seed=7, calibration=snapshot)


# ---------- contrainte 2 : pas de reproductibilite bit-a-bit ----------


def test_le_backend_declare_ne_pas_etre_bit_exact(snapshot):
    assert SimulatedHardwareBackend(snapshot).is_bit_exact_replayable() is False


def test_le_verdict_est_plafonne_MEME_quand_les_octets_coincident(run):
    """LE test du protocole.

    Sous le capot qsim est deterministe a seed fixe, donc les tirages sont
    identiques. Le harnais doit malgre tout refuser de dire BIT_EXACT : le
    plafond est un CONTRAT du backend, pas une deduction de ce qu'on observe.
    Deduire le verdict de l'observation donnerait, sur une vraie machine, une
    conclusion que la physique ne permet pas.
    """
    rapport = replay(run.manifest)
    assert rapport.verdict is Verdict.STATISTICALLY_COMPATIBLE
    assert "plafonne" in rapport.detail
    assert "identiques" in rapport.detail, (
        "les octets coincident bel et bien : c'est ce qui rend le plafond "
        "significatif plutot que trivial"
    )


def test_le_plafond_s_applique_aussi_au_rejeu_sur_archive(run, tmp_path):
    RunRecord.from_capture(run).save(tmp_path / "archive")
    rapport = replay_record(RunRecord.load(tmp_path / "archive"))
    assert rapport.verdict is Verdict.STATISTICALLY_COMPATIBLE
    assert "plafonne" in rapport.detail


def test_un_simulateur_n_est_PAS_plafonne():
    """Controle : sans lui, le test precedent ne distinguerait pas « plafonne »
    de « le harnais ne dit jamais BIT_EXACT »."""
    propre = capture(_ghz(), backend="qsim", seed=7, repetitions=400)
    assert replay(propre.manifest).verdict is Verdict.BIT_EXACT


# ---------- contrainte 3 : l'etat d'appareil doit etre scelle ----------


def test_le_materiel_exige_une_calibration():
    with pytest.raises(ValueError, match="calibration"):
        capture(_ghz(), backend="hardware-sim", seed=7, repetitions=10)


def test_une_calibration_donnee_a_un_simulateur_est_REFUSEE(snapshot):
    """L'ignorer laisserait croire qu'elle a influence le resultat."""
    with pytest.raises(ValueError, match="n'utilise pas"):
        capture(_ghz(), backend="qsim", seed=7, repetitions=10, calibration=snapshot)


def test_la_calibration_est_scellee_dans_le_manifeste(run, snapshot):
    assert run.manifest.calibration_json is not None
    relu = run.manifest.calibration()
    assert relu.snapshot_hash == snapshot.snapshot_hash
    assert relu.device_id == snapshot.device_id


def test_la_calibration_entre_dans_le_hash_SEMANTIQUE(snapshot):
    """Elle DETERMINE le resultat — c'est ce qui la distingue du contexte
    classique, volontairement exclu du hash semantique."""
    autre = synthetic_snapshot(_qubits(), base_date="2026-08-29T06:56:41+00:00")
    a = capture(
        _ghz(), backend="hardware-sim", seed=7, repetitions=50, calibration=snapshot
    ).manifest
    b = capture(
        _ghz(), backend="hardware-sim", seed=7, repetitions=50, calibration=autre
    ).manifest
    assert a.semantic_hash != b.semantic_hash


def test_une_calibration_falsifiee_est_detectee(run):
    altere = Manifest.from_dict(
        {**run.manifest.to_dict(), "calibration_json": '{"schema_version": "1.0"}'}
    )
    with pytest.raises(ValueError, match="contenu|semantique|manifeste"):
        altere.verify_self()


def test_le_backend_refuse_un_bruit_qui_contredirait_la_calibration(snapshot):
    """Accepter un modele de bruit externe permettrait de contredire l'etat
    d'appareil scelle sans que rien ne le signale."""
    backend = SimulatedHardwareBackend(snapshot)
    with pytest.raises(ValueError, match="contredirait"):
        backend.sample(
            _ghz(), repetitions=10, seed=7, options={}, noise=cirq.depolarize(0.1)
        )


# ---------- l'instantane dit la verite sur ses dates ----------


def test_l_instantane_a_un_etalement_temporel_non_nul(snapshot):
    """Un instantane n'est pas l'etat d'un appareil a un instant : c'est un sac
    de mesures datees separement. Chez IBM, T1 et readout_error d'un meme
    « instantane » sont mesures a deux jours d'ecart."""
    assert snapshot.temporal_spread_seconds() == pytest.approx(48 * 3600)
    assert len(set(snapshot.all_dates())) > 1


def test_chaque_parametre_porte_sa_propre_date(snapshot):
    params = snapshot.qubits[str(cirq.LineQubit(0))]
    assert params["t1_us"].date != params["readout_error"].date


def test_l_instantane_se_verifie(snapshot):
    snapshot.verify_self()


def test_une_valeur_modifiee_casse_le_hash_de_l_instantane(snapshot):
    donnees = snapshot.to_dict()
    donnees["qubits"][str(cirq.LineQubit(0))]["t1_us"]["value"] = 999.0
    with pytest.raises(ValueError, match="calibration"):
        CalibrationSnapshot.from_dict(donnees).verify_self()


def test_aller_retour_json_de_l_instantane(snapshot):
    relu = CalibrationSnapshot.from_json(snapshot.to_json())
    assert relu.snapshot_hash == snapshot.snapshot_hash
    relu.verify_self()


def test_le_schema_de_calibration_est_verifie(snapshot):
    donnees = {**snapshot.to_dict(), "schema_version": "99.0"}
    with pytest.raises(ValueError, match="schema"):
        CalibrationSnapshot.from_dict(donnees)


def test_la_version_de_schema_est_celle_attendue(snapshot):
    assert snapshot.schema_version == CALIBRATION_SCHEMA_VERSION


# ---------- le bruit derive fait vraiment quelque chose ----------


def test_le_bruit_de_calibration_degrade_reellement_le_resultat(run):
    """Un GHZ parfait ne donne que |000> et |111>. Sous bruit, d'autres
    bitstrings apparaissent : sans ca, le backend ne testerait rien."""
    bruite = bitstring_counts(run.samples["m"])
    propre = bitstring_counts(
        capture(_ghz(), backend="qsim", seed=7, repetitions=400).samples["m"]
    )
    assert set(propre) == {0b000, 0b111}
    assert len(bruite) > len(propre)


def test_le_ghz_reste_domine_par_000_et_111_malgre_le_bruit(run):
    """Le bruit doit degrader, pas detruire : sinon le modele serait absurde."""
    comptes = bitstring_counts(run.samples["m"])
    domine = comptes.get(0b000, 0) + comptes.get(0b111, 0)
    assert domine / sum(comptes.values()) > 0.85


def test_une_calibration_plus_bruyante_degrade_davantage():
    q = _qubits()
    propre = synthetic_snapshot(q)
    sale = CalibrationSnapshot.build(
        device_id="sale",
        device_version="1.0",
        qubits={
            str(x): {
                "t1_us": DatedValue(2.0, "2026-08-28T06:00:00+00:00", "us"),
                "readout_error": DatedValue(0.15, "2026-08-28T06:00:00+00:00", ""),
            }
            for x in q
        },
        gates={
            f"x:{x}": {
                "gate_error": DatedValue(0.08, "2026-08-28T06:00:00+00:00", ""),
                "gate_length_ns": DatedValue(120.0, "2026-08-28T06:00:00+00:00", "ns"),
            }
            for x in q
        },
        basis_gates=["x"],
        coupling_map=[[0, 1], [1, 2]],
    )

    def fidelite(snap):
        r = capture(
            _ghz(), backend="hardware-sim", seed=7, repetitions=600, calibration=snap
        )
        c = bitstring_counts(r.samples["m"])
        return (c.get(0b000, 0) + c.get(0b111, 0)) / sum(c.values())

    assert fidelite(sale) < fidelite(propre)


# ---------- le registre ----------


def test_le_registre_sait_quels_backends_exigent_une_calibration():
    assert "hardware-sim" in NEEDS_CALIBRATION
    assert "qsim" not in NEEDS_CALIBRATION


def test_make_backend_refuse_un_nom_inconnu():
    with pytest.raises(KeyError, match="inconnu"):
        make_backend("machine-imaginaire")


def test_la_version_du_backend_porte_l_appareil(snapshot):
    backend = SimulatedHardwareBackend(snapshot)
    assert snapshot.device_id in backend.version


def test_le_nom_du_backend_annonce_que_c_est_un_substitut():
    """Un manifeste ne doit jamais laisser croire qu'il vient d'une vraie
    machine."""
    assert SimulatedHardwareBackend.name == "hardware-sim"
