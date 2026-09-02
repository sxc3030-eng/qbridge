"""Le verdict de plausibilite physique.

CE QU'IL AJOUTE AUX TROIS AUTRES. Une archive fabriquee de toutes pieces,
scellee proprement et annoncant un GHZ parfait passe `verify_integrity()` sans
broncher : les octets sont coherents avec leur propre empreinte. Ce controle-ci
est le seul qui regarde si la PHYSIQUE tient.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

cirq = pytest.importorskip("cirq")
pytest.importorskip("qiskit_ibm_runtime")

from qiskit_ibm_runtime.fake_provider import FakeManilaV2  # noqa: E402

from qbridge import capture  # noqa: E402
from qbridge.backends.ibm_runtime import IbmRuntimeBackend  # noqa: E402
from qbridge.capture import hash_samples  # noqa: E402
from qbridge.plausibility import (  # noqa: E402
    MIN_SHOTS,
    Plausibility,
    fidelite_predite,
    verify_physical_plausibility,
)
from qbridge.record import RunRecord  # noqa: E402


@pytest.fixture
def backend():
    return IbmRuntimeBackend(FakeManilaV2())


def _ghz(n=3):
    q = cirq.LineQubit.range(n)
    ops = [cirq.H(q[0])]
    ops += [cirq.CNOT(q[i], q[i + 1]) for i in range(n - 1)]
    ops.append(cirq.measure(*q, key="m"))
    return cirq.Circuit(ops)


def _truquer(record, tirages):
    """Remplace les tirages ET recalcule l'empreinte : le faux est
    PROPREMENT scelle, exactement comme le ferait un faussaire soigneux."""
    samples = {"m": tirages}
    return dataclasses.replace(
        record, samples=samples, result_hash=hash_samples(samples)
    )


@pytest.fixture
def brut(backend):
    """Archive telle que le backend factice la produit.

    ATTENTION : ses tirages NE SONT PAS coherents avec sa propre calibration
    (voir `test_le_simulateur_factice_contredit_sa_calibration`). Utile pour
    tout sauf pour illustrer un cas nominal.
    """
    return RunRecord.from_capture(
        capture(_ghz(), backend=backend, seed=7, repetitions=1024)
    )


def _tirages_a_la_fidelite(fidelite, n=1024, graine=0):
    """Fabrique des tirages GHZ a une fidelite VOULUE.

    Le backend factice d'IBM ne peut pas servir de cas nominal : il declare du
    bruit et simule sans. On synthetise donc l'entree pour controler exactement
    ce que le verdict doit voir.
    """
    generateur = np.random.default_rng(graine)
    dans = generateur.random(n) < fidelite
    tirages = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        if dans[i]:
            valeur = 0 if generateur.random() < 0.5 else 7
        else:
            valeur = int(generateur.integers(1, 7))  # hors du support {0, 7}
        for bit in range(3):
            tirages[i, 2 - bit] = (valeur >> bit) & 1
    return tirages


@pytest.fixture
def archive(brut):
    """Archive dont les tirages COLLENT a la calibration scellee."""
    from qbridge.calibration import CalibrationSnapshot

    instantane = CalibrationSnapshot.from_json(brut.manifest.calibration_json)
    comptes = json.loads(brut.manifest.device_provenance_json)["gate_counts"]
    attendue, _, _ = fidelite_predite(instantane, comptes)
    return _truquer(brut, _tirages_a_la_fidelite(attendue))


# ---------- le cas nominal ----------


def test_un_vrai_resultat_est_plausible(archive):
    rapport = verify_physical_plausibility(archive)
    assert rapport.verdict is Plausibility.PLAUSIBLE
    assert 0.5 < rapport.predicted_fidelity < 1.0
    assert rapport.sigma < 2
    assert rapport.support_size == 2, "un GHZ a 3 qubits : 000 et 111"
    assert rapport.total_bitstrings == 8


def test_le_budget_d_erreur_est_rendu(archive):
    """Sur `ibm_marrakesh`, la LECTURE pesait 73.8 % du budget contre 20 %
    pour les portes a deux qubits. Ce n'est pas la qu'on regarde d'habitude,
    donc le rapport doit le dire."""
    rapport = verify_physical_plausibility(archive)
    assert rapport.error_budget
    assert abs(sum(rapport.error_budget.values()) - 1.0) < 1e-9


def test_le_simulateur_factice_contredit_sa_calibration(brut):
    """CONSTAT, pas defaut de qbridge : `FakeManilaV2` declare des erreurs de
    lecture de 3.5 %, 2.2 % et 9.6 % — soit ~16 % d'erreur totale attendue —
    et son `SamplerV2` rend 1024/1024 tirages PARFAITS.

    Le backend factice publie du bruit et simule sans. C'est precisement le
    genre d'incoherence que ce verdict existe pour attraper, et il l'attrape
    sur l'outil d'IBM lui-meme. Consequence pratique : aucun cas nominal ne
    peut etre bati dessus, d'ou les tirages synthetiques."""
    rapport = verify_physical_plausibility(brut)
    assert rapport.observed_weight == 1.0
    assert rapport.predicted_fidelity < 0.90
    assert rapport.verdict is Plausibility.IMPLAUSIBLE


# ---------- L'ATTAQUE : une archive fabriquee ----------


def test_un_resultat_FABRIQUE_est_declare_implausible(archive):
    """LE controle qui justifie ce module.

    Un GHZ parfait — 100 % sur 000 et 111 — sur une machine dont la
    calibration scellee dit qu'elle ne peut pas depasser ~97 %. L'archive est
    scellee dans les regles : son empreinte est juste, `verify_integrity`
    passe. Seule la physique la trahit.
    """
    n = 1024
    parfait = np.zeros((n, 3), dtype=np.uint8)
    parfait[n // 2 :] = 1
    faux = _truquer(archive, parfait)

    faux.verify_integrity()  # le faux est PROPREMENT scelle

    rapport = verify_physical_plausibility(faux)
    assert rapport.verdict is Plausibility.IMPLAUSIBLE
    assert rapport.observed_weight == 1.0
    assert rapport.sigma > 4


def test_un_resultat_trop_BRUYANT_est_aussi_implausible(archive):
    """La detection va dans les deux sens : une archive qui annonce un bruit
    bien pire que ce que la machine declare est tout aussi incoherente."""
    generateur = np.random.default_rng(0)
    bruit = generateur.integers(0, 2, size=(1024, 3)).astype(np.uint8)
    rapport = verify_physical_plausibility(_truquer(archive, bruit))
    assert rapport.verdict is Plausibility.IMPLAUSIBLE


def test_le_sigma_reste_fini_quand_le_faux_annonce_100_pourcent(archive):
    """Defaut de ma premiere version : la variance etait calculee sur la
    proportion OBSERVEE, qui devient degeneree a 100 %, produisant un sigma de
    838 142 — un artefact, pas une mesure. Elle se calcule sous l'hypothese
    nulle, donc sur la fidelite PREDITE."""
    n = 1024
    parfait = np.zeros((n, 3), dtype=np.uint8)
    parfait[n // 2 :] = 1
    rapport = verify_physical_plausibility(_truquer(archive, parfait))
    assert rapport.sigma < 1000, "un sigma astronomique trahit une variance fausse"
    assert 4 < rapport.sigma < 100


# ---------- ce sur quoi il refuse de se prononcer ----------


def test_sans_calibration_scellee_le_verdict_est_INDETERMINE(archive):
    """C'etait l'etat de TOUTES les archives materielles avant le scellement
    de l'etat d'appareil : il n'y avait rien a confronter."""
    nu = dataclasses.replace(
        archive, manifest=dataclasses.replace(archive.manifest, calibration_json=None)
    )
    rapport = verify_physical_plausibility(nu)
    assert rapport.verdict is Plausibility.INDETERMINE
    assert "aucun etat d'appareil" in rapport.reason


def test_sans_provenance_le_verdict_est_INDETERMINE(archive):
    nu = dataclasses.replace(
        archive,
        manifest=dataclasses.replace(archive.manifest, device_provenance_json=None),
    )
    assert (
        verify_physical_plausibility(nu).verdict is Plausibility.INDETERMINE
    )


def test_trop_peu_de_tirages_donne_INDETERMINE(backend):
    record = RunRecord.from_capture(
        capture(_ghz(), backend=backend, seed=7, repetitions=MIN_SHOTS - 1)
    )
    rapport = verify_physical_plausibility(record)
    assert rapport.verdict is Plausibility.INDETERMINE
    assert "tirages" in rapport.reason


def test_un_support_NON_DISCRIMINANT_donne_INDETERMINE(backend):
    """Un circuit dont la loi ideale couvre tous les bitstrings ne permet
    aucun controle : un resultat totalement depolarise tomberait deja dans le
    support. Se taire est la seule reponse honnete."""
    q = cirq.LineQubit.range(3)
    uniforme = cirq.Circuit([cirq.H(x) for x in q] + [cirq.measure(*q, key="m")])
    record = RunRecord.from_capture(
        capture(uniforme, backend=backend, seed=7, repetitions=1024)
    )
    rapport = verify_physical_plausibility(record)
    assert rapport.verdict is Plausibility.INDETERMINE
    assert "discriminerait" in rapport.reason


def test_INDETERMINE_ne_peut_pas_etre_pris_pour_une_acceptation():
    """Meme discipline que `Verdict.INDETERMINATE` : un test ecrit
    `<= TENSION` ne doit jamais accepter une absence de conclusion."""
    assert Plausibility.INDETERMINE > Plausibility.IMPLAUSIBLE
    assert not (Plausibility.INDETERMINE <= Plausibility.TENSION)


# ---------- la fidelite predite ----------


def test_une_porte_absente_de_la_calibration_est_SIGNALEE(archive):
    """Compter son erreur comme nulle SURESTIME la fidelite predite, donc rend
    le controle plus permissif. Le taire serait le pire des choix."""
    from qbridge.calibration import CalibrationSnapshot

    instantane = CalibrationSnapshot.from_json(archive.manifest.calibration_json)
    _, _, avertissements = fidelite_predite(
        instantane, {"porte_inventee": 3, "cx": 2}
    )
    assert any("porte_inventee" in a for a in avertissements)
    assert any("SUREVALUEE" in a for a in avertissements)


def test_l_erreur_de_lecture_n_est_pas_comptee_deux_fois(archive):
    """Sur une vraie machine IBM, `measure` est une porte du target ET
    `readout_error` existe par qubit, avec la MEME valeur. Les additionner
    doublerait la source d'erreur dominante — 73.8 % du budget."""
    from qbridge.calibration import CalibrationSnapshot

    instantane = CalibrationSnapshot.from_json(archive.manifest.calibration_json)
    une_seule, _, _ = fidelite_predite(instantane, {"measure": 1})
    lectures = [
        p["readout_error"].value
        for p in instantane.qubits.values()
        if "readout_error" in p
    ]
    attendue = 1.0 - sum(lectures) / len(lectures)
    assert abs(une_seule - attendue) < 1e-12


def test_la_fidelite_predite_decroit_avec_le_nombre_de_portes(archive):
    from qbridge.calibration import CalibrationSnapshot

    instantane = CalibrationSnapshot.from_json(archive.manifest.calibration_json)
    peu, _, _ = fidelite_predite(instantane, {"cx": 1})
    beaucoup, _, _ = fidelite_predite(instantane, {"cx": 20})
    assert beaucoup < peu < 1.0


# ---------- serialisation ----------


def test_le_rapport_est_serialisable(archive):
    d = verify_physical_plausibility(archive).to_dict()
    json.dumps(d)
    assert d["verdict"] == "PLAUSIBLE"
    assert d["shots"] == 1024


# ---------- la CLI ----------


def _lancer(*args):
    """Lance la CLI en capturant sa sortie. Rend (code, texte)."""
    import contextlib
    import io

    from qbridge.cli import main

    flux = io.StringIO()
    with contextlib.redirect_stdout(flux), contextlib.redirect_stderr(flux):
        code = main(list(args))
    return code, flux.getvalue()


def test_verify_rapporte_la_plausibilite(archive, tmp_path):
    archive.save(tmp_path / "a")
    code, texte = _lancer("verify", str(tmp_path / "a"))
    assert code == 0
    assert "plausibilite physique" in texte
    assert "PLAUSIBLE" in texte
    assert "budget d'erreur" in texte


def test_un_faux_passe_l_integrite_mais_PAS_la_physique(archive, tmp_path):
    """LA demonstration. Une archive fabriquee, proprement scellee, franchit
    tous les controles d'integrite — le code de sortie reste 0 — et n'est
    attrapee que par le controle de plausibilite."""
    n = 1024
    parfait = np.zeros((n, 3), dtype=np.uint8)
    parfait[n // 2 :] = 1
    _truquer(archive, parfait).save(tmp_path / "faux")

    code, texte = _lancer("verify", str(tmp_path / "faux"))
    assert code == 0, "l'integrite du faux est irreprochable"
    assert "resultats              : intacts" in texte or "intacts" in texte
    assert "IMPLAUSIBLE" in texte


def test_physique_stricte_fait_echouer_un_faux(archive, tmp_path):
    n = 1024
    parfait = np.zeros((n, 3), dtype=np.uint8)
    parfait[n // 2 :] = 1
    _truquer(archive, parfait).save(tmp_path / "faux")

    code, _ = _lancer("verify", str(tmp_path / "faux"), "--physique-stricte")
    assert code == 1


def test_physique_stricte_n_affecte_pas_une_archive_saine(archive, tmp_path):
    archive.save(tmp_path / "sain")
    code, _ = _lancer("verify", str(tmp_path / "sain"), "--physique-stricte")
    assert code == 0


def test_le_json_de_verify_porte_le_verdict(archive, tmp_path):
    archive.save(tmp_path / "a")
    code, texte = _lancer("verify", str(tmp_path / "a"), "--json")
    assert code == 0
    charge = json.loads(texte)
    assert charge["physical_plausibility"]["verdict"] == "PLAUSIBLE"
    assert charge["physical_plausibility"]["shots"] == 1024
