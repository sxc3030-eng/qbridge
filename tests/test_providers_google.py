"""Ingestion de calibrations Google REELLES.

Ces tests ne touchent aucun reseau : `cirq-google` embarque les calibrations
medianes de trois processeurs. Ils verifient que qbridge sait lire des donnees
de fournisseur telles qu'elles sont publiees, et non seulement les instantanes
synthetiques fabriques pour les besoins du protocole.
"""

from __future__ import annotations

import pytest

cirq = pytest.importorskip("cirq")
pytest.importorskip("cirq_google")

from qbridge import capture, replay  # noqa: E402
from qbridge.calibration import CalibrationSnapshot  # noqa: E402
from qbridge.providers import PROCESSEURS, from_google_calibration  # noqa: E402
from qbridge.record import RunRecord  # noqa: E402
from qbridge.verdict import Verdict, bitstring_counts  # noqa: E402


@pytest.fixture(scope="module")
def willow():
    instantane, avertissements = from_google_calibration("willow_pink")
    return instantane, avertissements


def test_les_trois_processeurs_se_chargent():
    for pid in PROCESSEURS:
        instantane, _ = from_google_calibration(pid)
        instantane.verify_self()
        assert instantane.device_id == f"google:{pid}"
        assert instantane.qubits, f"{pid} sans donnees de qubit"


def test_un_processeur_inconnu_est_refuse():
    with pytest.raises(ValueError, match="Processeur inconnu"):
        from_google_calibration("machine-imaginaire")


def test_willow_a_bien_105_qubits(willow):
    instantane, _ = willow
    assert len(instantane.qubits) == 105


def test_les_valeurs_sont_physiquement_plausibles(willow):
    """Garde-fou contre une erreur d'unite ou de mappage : un T1 en secondes
    au lieu de microsecondes passerait inapercu sans cette borne."""
    instantane, _ = willow
    for nom, params in instantane.qubits.items():
        t1 = params["t1_us"].value
        assert 1.0 < t1 < 1000.0, f"{nom} : T1={t1} us, invraisemblable"
        lecture = params["readout_error"].value
        assert 0.0 <= lecture < 0.5, f"{nom} : erreur de lecture={lecture}"


def test_l_inhomogeneite_de_l_appareil_est_preservee(willow):
    """Le point de tout l'exercice : un vrai appareil n'est pas uniforme, et
    l'instantane doit garder cette variation plutot que de la moyenner.
    Mesure : sur la chaine testee, un qubit a la moitie du T1 de ses voisins.
    """
    instantane, _ = willow
    valeurs = [p["t1_us"].value for p in instantane.qubits.values()]
    assert max(valeurs) / min(valeurs) > 2.0, (
        "aucune variation de T1 : le mappage a probablement ecrase les donnees"
    )


def test_la_progression_generationnelle_est_visible():
    """rainbow (2021) et willow_pink (2024). Si le mappage etait faux, ces deux
    instantanes ne se distingueraient pas comme ils le font reellement."""
    ancien, _ = from_google_calibration("rainbow")
    recent, _ = from_google_calibration("willow_pink")

    def moyenne_t1(snap):
        v = [p["t1_us"].value for p in snap.qubits.values()]
        return sum(v) / len(v)

    assert moyenne_t1(recent) > moyenne_t1(ancien) * 2


def test_les_hypotheses_sont_signalees_jamais_silencieuses(willow):
    """Google ne publie pas les durees de porte. Les inventer en silence serait
    exactement ce que ce projet refuse."""
    _, avertissements = willow
    joint = " ".join(avertissements)
    assert "durees de porte" in joint
    assert "hypotheses de l'appelant" in joint


def test_la_provenance_d_une_valeur_supposee_est_SCELLEE(willow):
    """Dans cinq ans, personne ne doit pouvoir prendre une duree fournie par
    l'appelant pour une mesure de Google."""
    instantane, _ = willow
    une_porte = next(iter(instantane.gates.values()))
    assert "ABSENT de la calibration Google" in une_porte["gate_length_ns"].unit


def test_une_metrique_absente_est_signalee_pas_ignoree():
    """rainbow utilise des sqrt_iswap, pas des CZ : la metrique CZ manque.
    L'adaptateur doit le DIRE plutot que de produire un instantane sans portes
    a deux qubits sans prevenir."""
    _, avertissements = from_google_calibration("rainbow")
    assert any("metrique absente" in a for a in avertissements)


def test_google_publie_un_horodatage_unique(willow):
    """Difference reelle avec IBM, qui date chaque parametre. Sur une
    calibration MEDIANE l'etalement vaut 0 par construction — ce n'est pas un
    defaut, et le dire evite qu'on le prenne pour une garantie de simultaneite.
    """
    instantane, avertissements = willow
    assert instantane.temporal_spread_seconds() == 0.0
    assert any("horodatage unique" in a for a in avertissements)


def test_l_instantane_reel_survit_a_un_aller_retour_json(willow):
    instantane, _ = willow
    relu = CalibrationSnapshot.from_json(instantane.to_json())
    relu.verify_self()
    assert relu.snapshot_hash == instantane.snapshot_hash


# ---------- le cycle complet sur des donnees reelles ----------


def _chaine(device, longueur=4):
    graphe = device.metadata.nx_graph
    chaine = [sorted(device.metadata.qubit_set)[0]]
    while len(chaine) < longueur:
        voisins = [v for v in graphe.neighbors(chaine[-1]) if v not in chaine]
        if not voisins:
            break
        chaine.append(sorted(voisins)[0])
    return chaine


@pytest.fixture(scope="module")
def ghz_reel(willow):
    from cirq_google.engine import create_device_from_processor_id

    instantane, _ = willow
    chaine = _chaine(create_device_from_processor_id("willow_pink"))
    circuit = cirq.Circuit([cirq.H(chaine[0])])
    for a, b in zip(chaine, chaine[1:]):
        circuit.append(cirq.CNOT(a, b))
    circuit.append(cirq.measure(*chaine, key="m"))
    return circuit, chaine, instantane


def test_le_cycle_complet_tourne_sur_une_calibration_reelle(ghz_reel, tmp_path):
    circuit, chaine, instantane = ghz_reel
    run = capture(
        circuit,
        backend="hardware-sim",
        seed=7,
        repetitions=800,
        calibration=instantane,
    )
    assert run.manifest.calibration().device_id == "google:willow_pink"

    record = RunRecord.from_capture(run)
    record.save(tmp_path / "willow")
    relu = RunRecord.load(tmp_path / "willow")
    relu.verify_integrity()

    from qbridge import verify_archival

    rapport = verify_archival(relu)
    assert rapport.manifest_intact and rapport.results_intact
    assert rapport.total_shots == 800


def test_le_bruit_reel_degrade_sans_detruire(ghz_reel):
    """Un GHZ sous bruit Willow reel doit rester domine par |0...0> et |1...1>
    tout en fuyant vers d'autres bitstrings. Ni parfait, ni detruit."""
    circuit, chaine, instantane = ghz_reel
    run = capture(
        circuit,
        backend="hardware-sim",
        seed=7,
        repetitions=800,
        calibration=instantane,
    )
    comptes = bitstring_counts(run.samples["m"])
    n = len(chaine)
    domine = comptes.get(0, 0) + comptes.get((1 << n) - 1, 0)
    fidelite = domine / sum(comptes.values())

    assert 0.80 < fidelite < 1.0, f"fidelite {fidelite:.3f} invraisemblable"
    assert len(comptes) > 2, "aucune fuite : le bruit reel n'a pas ete applique"


def test_le_plafond_tient_aussi_sur_une_calibration_reelle(ghz_reel):
    circuit, _, instantane = ghz_reel
    run = capture(
        circuit,
        backend="hardware-sim",
        seed=7,
        repetitions=400,
        calibration=instantane,
    )
    assert replay(run.manifest).verdict is not Verdict.BIT_EXACT


def test_la_calibration_reelle_entre_dans_le_hash_semantique(ghz_reel):
    """Deux appareils differents ne sont pas la meme experience."""
    circuit, _, willow_snap = ghz_reel
    rainbow_snap, _ = from_google_calibration("rainbow")

    a = capture(
        circuit,
        backend="hardware-sim",
        seed=7,
        repetitions=50,
        calibration=willow_snap,
    ).manifest
    b = capture(
        circuit,
        backend="hardware-sim",
        seed=7,
        repetitions=50,
        calibration=rainbow_snap,
    ).manifest
    assert a.semantic_hash != b.semantic_hash
