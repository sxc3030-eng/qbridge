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


INFIDELITE_CIBLE = 0.025
"""Cible des fixtures nominales : DANS le domaine de validite (<= 3 %), mais
assez loin de 1.0 pour qu'un faux annoncant 100 % reste detectable."""


def _calibration_mise_a_l_echelle(instantane, facteur):
    """Meme instantane, erreurs multipliees par `facteur`.

    `FakeManilaV2` affiche 16.6 % d'infidelite sur ce circuit : hors du domaine
    de validite, et a juste titre. Un cas NOMINAL demande une machine plus
    propre, donc on en fabrique une plutot que de tordre le seuil.
    """
    from qbridge.calibration import CalibrationSnapshot, DatedValue

    bruyants = {"gate_error", "readout_error", "prob_meas0_prep1", "prob_meas1_prep0"}

    def echelle(params):
        return {
            nom: DatedValue(
                dv.value * facteur if nom in bruyants else dv.value, dv.date, dv.unit
            )
            for nom, dv in params.items()
        }

    return CalibrationSnapshot.build(
        device_id=instantane.device_id,
        device_version=instantane.device_version,
        qubits={k: echelle(v) for k, v in instantane.qubits.items()},
        gates={k: echelle(v) for k, v in instantane.gates.items()},
        basis_gates=instantane.basis_gates,
        coupling_map=instantane.coupling_map,
    )


def _resceller(manifeste, calibration_json):
    """Rescelle les empreintes apres modification, comme le ferait `build`."""
    m = dataclasses.replace(
        manifeste,
        calibration_json=calibration_json,
        semantic_hash="",
        content_hash="",
    )
    m = dataclasses.replace(m, semantic_hash=m._compute_semantic_hash())
    return dataclasses.replace(m, content_hash=m._compute_content_hash())


@pytest.fixture
def archive(brut):
    """Archive NOMINALE : machine propre, tirages colles a sa calibration."""
    from qbridge.calibration import CalibrationSnapshot

    instantane = CalibrationSnapshot.from_json(brut.manifest.calibration_json)
    operations = json.loads(brut.manifest.device_provenance_json)["operations"]

    brute, _, _ = fidelite_predite(instantane, {}, operations)
    facteur = INFIDELITE_CIBLE / (1.0 - brute)
    propre = _calibration_mise_a_l_echelle(instantane, facteur)

    attendue, _, _ = fidelite_predite(propre, {}, operations)
    assert 1.0 - attendue <= 0.03, "la fixture nominale doit etre DANS le domaine"

    record = dataclasses.replace(
        brut, manifest=_resceller(brut.manifest, propre.to_json())
    )
    return _truquer(record, _tirages_a_la_fidelite(attendue))


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


def test_un_resultat_trop_BRUYANT_est_refuse_DANS_le_domaine(archive):
    """Dans le domaine de validite, la detection va dans les deux sens : une
    archive bien plus bruyante que la machine declaree est incoherente."""
    generateur = np.random.default_rng(0)
    bruit = generateur.integers(0, 2, size=(1024, 3)).astype(np.uint8)
    rapport = verify_physical_plausibility(_truquer(archive, bruit))
    assert rapport.within_domain is True
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


# ---------- les operations EXACTES, pas des comptes ----------


def test_les_operations_exactes_sont_scellees(backend):
    """`gate_counts` dit COMBIEN de cz, jamais LESQUELS. Sur
    `ibm_marrakesh`, les paires cz scellees vont de 1.65e-3 a 3.63e-3 : un
    facteur 2.2 que des comptes agreges ne peuvent pas distinguer."""
    run = capture(_ghz(), backend=backend, seed=7, repetitions=100)
    provenance = json.loads(run.manifest.device_provenance_json)

    operations = provenance["operations"]
    assert operations, "les operations exactes doivent etre scellees"
    for cle in operations:
        assert ":q(" in cle, f"{cle!r} doit nommer ses qubits physiques"
    assert sum(v for k, v in operations.items() if k.startswith("measure")) == 3


def test_les_cles_d_operation_suivent_le_format_de_la_calibration(backend):
    """Un lookup direct, pas une traduction — une traduction serait un endroit
    de plus ou se tromper."""
    from qbridge.calibration import CalibrationSnapshot

    run = capture(_ghz(), backend=backend, seed=7, repetitions=100)
    operations = json.loads(run.manifest.device_provenance_json)["operations"]
    instantane = CalibrationSnapshot.from_json(run.manifest.calibration_json)

    portes = {k for k in operations if not k.startswith("measure")}
    connues = set(instantane.gates)
    assert portes & connues, "aucune cle d'operation ne correspond a la calibration"


def test_la_moyenne_ne_distingue_pas_une_transpilation_MALCHANCEUSE(archive):
    """Deux placements de meme profondeur, l'un sur la paire propre et l'autre
    sur la bruyante, doivent donner des predictions DIFFERENTES."""
    from qbridge.calibration import CalibrationSnapshot

    instantane = CalibrationSnapshot.from_json(archive.manifest.calibration_json)
    paires = sorted(k for k in instantane.gates if k.startswith("cx:"))
    erreurs = {k: instantane.gates[k]["gate_error"].value for k in paires}
    propre = min(erreurs, key=erreurs.get)
    bruyante = max(erreurs, key=erreurs.get)
    assert erreurs[bruyante] > erreurs[propre]

    chanceuse, _, _ = fidelite_predite(instantane, {}, {propre: 4})
    malchanceuse, _, _ = fidelite_predite(instantane, {}, {bruyante: 4})
    assert malchanceuse < chanceuse

    moyenne, _, _ = fidelite_predite(instantane, {"cx": 4})
    assert malchanceuse < moyenne < chanceuse, (
        "la moyenne tombe entre les deux et ne peut trancher ni dans un sens "
        "ni dans l'autre"
    )


def test_l_approximation_par_moyenne_est_SIGNALEE(archive):
    """Une archive scellee avant que les operations exactes existent retombe
    sur la moyenne. C'est acceptable ; le taire ne l'est pas."""
    from qbridge.calibration import CalibrationSnapshot

    instantane = CalibrationSnapshot.from_json(archive.manifest.calibration_json)
    _, _, avertissements = fidelite_predite(instantane, {"cx": 2}, None)
    assert any("APPROXIMATIF" in a for a in avertissements)

    _, _, sans = fidelite_predite(instantane, {}, {"cx:q(0),q(1)": 2})
    assert not any("APPROXIMATIF" in a for a in sans)


def test_a_grande_profondeur_la_moyenne_produirait_un_FAUX_verdict(archive):
    """LE test qui justifie ce changement.

    L'ecart entre moyenne et exact croit avec le nombre de portes a deux
    qubits. Mesure sur la calibration reelle d'`ibm_marrakesh` : 0.9 sigma a
    2 portes, 1.9 a 10, et 4.1 a 50 — au-dela du seuil IMPLAUSIBLE. A cette
    profondeur, l'approximation ne perd plus en precision : elle produit un
    verdict FAUX sur une archive parfaitement honnete.
    """
    import math

    from qbridge.calibration import CalibrationSnapshot

    instantane = CalibrationSnapshot.from_json(archive.manifest.calibration_json)
    paires = sorted(k for k in instantane.gates if k.startswith("cx:"))
    bruyante = max(paires, key=lambda k: instantane.gates[k]["gate_error"].value)

    moyenne, _, _ = fidelite_predite(instantane, {"cx": 50})
    exact, _, _ = fidelite_predite(instantane, {}, {bruyante: 50})

    ecart_type = math.sqrt(moyenne * (1 - moyenne) / 1024)
    assert abs(moyenne - exact) / ecart_type > 2, (
        "a 50 portes a deux qubits, le seul choix moyenne/exact deplace deja "
        "le verdict de plus de deux sigma"
    )


def test_une_operation_inconnue_retombe_sur_la_moyenne_en_le_disant(archive):
    from qbridge.calibration import CalibrationSnapshot

    instantane = CalibrationSnapshot.from_json(archive.manifest.calibration_json)
    _, _, avertissements = fidelite_predite(
        instantane, {}, {"cx:q(40),q(41)": 2}
    )
    assert any("erreur moyenne" in a for a in avertissements)


def test_la_lecture_exacte_utilise_le_qubit_CONCERNE(archive):
    """Les erreurs de lecture varient d'un qubit a l'autre — 3.5 %, 2.2 % et
    9.6 % sur FakeManila. Mesurer trois fois le pire n'est pas mesurer trois
    qubits differents."""
    from qbridge.calibration import CalibrationSnapshot

    instantane = CalibrationSnapshot.from_json(archive.manifest.calibration_json)
    lectures = {
        cle: params["readout_error"].value
        for cle, params in instantane.qubits.items()
        if "readout_error" in params
    }
    meilleur = min(lectures, key=lectures.get)
    pire = max(lectures, key=lectures.get)
    assert lectures[pire] > lectures[meilleur]

    bon, _, _ = fidelite_predite(instantane, {}, {f"measure:{meilleur}": 3})
    mauvais, _, _ = fidelite_predite(instantane, {}, {f"measure:{pire}": 3})
    assert mauvais < bon


# ---------- le domaine de validite du modele ----------


def _hors_domaine(brut, fidelite_voulue):
    """Archive dont l'infidelite predite depasse le domaine du modele."""
    return _truquer(brut, _tirages_a_la_fidelite(fidelite_voulue))


def test_hors_domaine_un_resultat_HONNETE_n_est_plus_accuse(brut):
    """LA correction, mesuree sur ibm_marrakesh.

    Un GHZ suivi de paires CNOT.CNOT — l'identite, donc l'etat ideal ne bouge
    pas. A 10 portes cz le modele predisait 94.09 %, la machine a rendu
    88.77 %, et le verdict criait IMPLAUSIBLE a 7.2 sigma sur une archive
    parfaitement honnete. A 34 cz : 40.7 sigma.

    La cause est visible dans les distributions : a 34 cz les etats dominants
    sont `010` (33.5 %) et `101` (23.2 %), tous deux un basculement du qubit
    que les paires CNOT martelent. Erreur COHERENTE, qui s'accumule en
    amplitude et croit en n^2 — un modele d'erreurs independantes ne peut pas
    la voir.

    Une fausse accusation est le pire mode de defaillance de cet outil.
    """
    rapport = verify_physical_plausibility(_hors_domaine(brut, 0.45))
    assert rapport.within_domain is False
    assert rapport.verdict is Plausibility.INDETERMINE
    assert "coherentes" in rapport.reason


def test_hors_domaine_les_chiffres_restent_rendus(brut):
    """INDETERMINE ne doit pas vouloir dire « rapport vide » : le lecteur a
    besoin des chiffres pour juger lui-meme."""
    rapport = verify_physical_plausibility(_hors_domaine(brut, 0.45))
    assert rapport.predicted_fidelity is not None
    assert rapport.observed_weight is not None
    assert rapport.upper_bound is not None
    assert rapport.error_budget


def test_la_borne_reste_opposable_HORS_domaine(brut):
    """Le bruit non modelise ne peut que degrader, jamais faire mieux que les
    erreurs declarees. Depasser la borne est donc impossible a TOUTE
    profondeur — verifie sur les six archives reelles, de 2 a 258 portes cz."""
    n = 1024
    parfait = np.zeros((n, 3), dtype=np.uint8)
    parfait[n // 2 :] = 1
    rapport = verify_physical_plausibility(_truquer(brut, parfait))
    assert rapport.within_domain is False
    assert rapport.verdict is Plausibility.IMPLAUSIBLE
    assert rapport.observed_weight > rapport.upper_bound
    assert "DEPASSE" in rapport.reason


def test_la_borne_vaut_F_plus_le_reste_au_hasard(archive):
    """`F + (1-F) * |support|/2^n` : avec proba F le calcul reussit, sinon le
    resultat brouille tombe dans le support par hasard."""
    rapport = verify_physical_plausibility(archive)
    hasard = rapport.support_size / rapport.total_bitstrings
    attendue = rapport.predicted_fidelity + (1 - rapport.predicted_fidelity) * hasard
    assert abs(rapport.upper_bound - attendue) < 1e-12
    assert rapport.upper_bound > rapport.predicted_fidelity


def test_la_borne_ne_peut_jamais_depasser_un(archive):
    rapport = verify_physical_plausibility(archive)
    assert rapport.upper_bound <= 1.0


def test_le_domaine_suit_l_infidelite_predite(brut):
    from qbridge.plausibility import INFIDELITE_DOMAINE_MAX

    assert INFIDELITE_DOMAINE_MAX == 0.03
    rapport = verify_physical_plausibility(_hors_domaine(brut, 0.5))
    assert (1 - rapport.predicted_fidelity) > INFIDELITE_DOMAINE_MAX
    assert rapport.within_domain is False
