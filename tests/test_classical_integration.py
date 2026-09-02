"""Soudure des deux moities du pont : contexte classique scelle dans le manifeste.

Le point de conception verifie ici est la SEPARATION des deux hashes :
`semantic_hash` predit le resultat quantique, `content_hash` scelle le document.
Les confondre forcerait a choisir entre « detecter toute modification » et
« ne pas invalider un rejeu pour une raison qui n'en est pas une ».
"""

from __future__ import annotations

import numpy as np
import pytest

import cirq
from qbridge import (
    Verdict,
    capture,
    capture_classical,
    replay,
    verify_source_unchanged,
)
from qbridge.manifest import Manifest
from qbridge.record import RunRecord


def reduire(samples):
    """Reduction publiee, version 1."""
    return float(np.mean(samples))


def reduire_v2(samples):
    """Reduction publiee, version 2 — la formule a change."""
    return float(np.mean(samples)) * 2.0


def _bell_mesure():
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1), cirq.measure(q0, q1, key="m"))


def _ctx(fn=reduire):
    return capture_classical(
        callables={"reduce": fn}, input_data={"probleme": "bell", "shots": 200}
    )


def _run(fn=reduire):
    return capture(
        _bell_mesure(), backend="qsim", seed=7, repetitions=200, classical=_ctx(fn)
    )


# ---------- scellement et reconstruction ----------


def test_le_contexte_classique_est_scelle_dans_le_manifeste():
    m = _run().manifest
    assert m.classical_json is not None
    ctx = m.classical()
    assert ctx is not None
    assert ctx.verifiable_roles() == ("reduce",)


def test_le_contexte_survit_a_un_aller_retour_disque(tmp_path):
    run = _run()
    run.manifest.save(tmp_path / "m.json")
    relu = Manifest.load(tmp_path / "m.json")
    relu.verify_self()
    assert relu.classical().context_hash == run.manifest.classical().context_hash


def test_un_manifeste_sans_contexte_reste_valide():
    """Le champ est optionnel : sceller le versant classique ne doit pas
    devenir obligatoire pour un simple essai."""
    m = capture(_bell_mesure(), backend="qsim", seed=7, repetitions=50).manifest
    assert m.classical_json is None and m.classical() is None
    m.verify_self()


# ---------- la separation des deux hashes ----------


def test_le_contexte_classique_ne_change_PAS_le_hash_semantique():
    """Propriete centrale. Le code qui reduit les tirages ne peut pas modifier
    l'execution quantique : deux manifestes qui ne different que par lui doivent
    rester semantiquement identiques, sinon on invaliderait un rejeu pour une
    raison qui n'en est pas une."""
    a = _run(reduire).manifest
    b = _run(reduire_v2).manifest
    assert a.classical_json != b.classical_json
    assert a.semantic_hash == b.semantic_hash


def test_le_contexte_classique_change_le_hash_de_contenu():
    """...mais il doit rester scelle : le document differe, et ca doit se voir."""
    a = _run(reduire).manifest
    b = _run(reduire_v2).manifest
    assert a.content_hash != b.content_hash


def test_le_rejeu_quantique_reste_bit_exact_malgre_un_changement_de_reduction():
    """Consequence pratique : changer la post-analyse n'invalide pas le rejeu."""
    assert replay(_run(reduire_v2).manifest).verdict is Verdict.BIT_EXACT


# ---------- content_hash couvre ce que semantic_hash ignore ----------


@pytest.mark.parametrize(
    "champ,valeur",
    [
        ("created_at", "1999-01-01T00:00:00+00:00"),
        ("classical_json", None),
    ],
)
def test_content_hash_detecte_les_champs_hors_perimetre_semantique(champ, valeur):
    """Ces champs etaient tous NON couverts avant : on pouvait les reecrire sans
    que rien ne le signale.

    `backend_version` figurait dans cette liste. C'ETAIT LE DEFAUT 24 : voir
    `test_le_moteur_qui_a_produit_le_resultat_est_SEMANTIQUE`.
    """
    m = _run().manifest
    altere = Manifest.from_dict({**m.to_dict(), champ: valeur})
    assert altere.semantic_hash == m.semantic_hash, (
        "ce champ est hors du perimetre semantique par conception"
    )
    with pytest.raises(ValueError, match="contenu"):
        altere.verify_self()


def test_content_hash_detecte_une_reecriture_de_l_environnement():
    m = _run().manifest
    altere = Manifest.from_dict(
        {**m.to_dict(), "environment": {**m.environment, "cpu_count": 9999}}
    )
    with pytest.raises(ValueError, match="contenu"):
        altere.verify_self()


def test_content_hash_couvre_automatiquement_tout_champ_futur():
    """Il est calcule par enumeration des champs de la dataclass, pas depuis une
    liste ecrite a la main : un champ ajoute plus tard est couvert d'office.
    Une liste manuelle est exactement ce qui avait laisse `circuit_json` dehors."""
    from dataclasses import fields

    m = _run().manifest
    couverts = {f.name for f in fields(m)} - {"content_hash"}
    for nom in couverts:
        assert hasattr(m, nom)
    assert len(couverts) == len(fields(m)) - 1


# ---------- la boucle archivistique complete ----------


def test_les_chiffres_publies_sont_regenerables_sans_ressource_quantique(
    tmp_path, monkeypatch
):
    """La garantie qui justifie tout le projet.

    Depuis une archive : verifier l'integrite, retrouver le code de reduction,
    confirmer qu'il n'a pas derive, et recalculer le chiffre publie — le tout
    avec les deux backends volontairement casses.
    """
    run = _run()
    chiffre_publie = reduire(run.samples["m"])
    RunRecord.from_capture(run).save(tmp_path / "archive")

    import qbridge.backends.cirq_ref as cr
    import qbridge.backends.qsim as qs

    def interdit(*a, **k):
        raise AssertionError("aucun circuit ne doit etre execute ici")

    for cls in (qs.QsimBackend, cr.CirqReferenceBackend):
        monkeypatch.setattr(cls, "sample", interdit)
        monkeypatch.setattr(cls, "simulate", interdit)

    archive = RunRecord.load(tmp_path / "archive")
    archive.verify_integrity()

    ctx = archive.manifest.classical()
    assert verify_source_unchanged(ctx, {"reduce": reduire}).has_drift is False

    assert reduire(archive.samples["m"]) == chiffre_publie


def test_une_derive_du_code_de_reduction_est_signalee_depuis_l_archive(tmp_path):
    """Ce qui dit a quelqu'un, dans cinq ans, que le code qu'il s'apprete a
    lancer n'est PAS celui qui a produit le chiffre publie."""
    RunRecord.from_capture(_run(reduire)).save(tmp_path / "archive")
    ctx = RunRecord.load(tmp_path / "archive").manifest.classical()

    rapport = verify_source_unchanged(ctx, {"reduce": reduire_v2})
    assert rapport.has_drift is True
    assert "reduce" in rapport.drifted


def test_le_moteur_qui_a_produit_le_resultat_est_SEMANTIQUE():
    """DEFAUT 24, trouve en classant les 23 precedents par cause.

    `backend_version` etait hors du hash semantique. Le test qui validait cette
    exclusion le disait pourtant lui-meme : « permettait a un manifeste de
    mentir sur le qsim qui l'avait produit ». `content_hash` attrape bien une
    REECRITURE, mais il ne rend pas deux executions REELLEMENT differentes
    semantiquement differentes — et c'est la question a laquelle le hash
    semantique pretend repondre.

    Le champ est devenu porteur de sens le jour ou le backend IBM est arrive :
    `backend_name` vaut « ibm-runtime » pour TOUTES les machines d'IBM, et
    `backend_version` est le seul champ qui distingue ibm_marrakesh d'ibm_fez.
    Le sceau n'a pas suivi le changement.

    Degat mesure : `qbridge diff` rendait « semantiquement IDENTIQUES » avec un
    code de sortie 0 pour deux machines differentes.
    """
    m = _run().manifest

    # `from_dict` CONSERVE les hashes stockes — c'est `verify_self` qui les
    # recalcule. On interroge donc le calcul, pas la valeur relue.
    autre_machine = Manifest.from_dict(
        {**m.to_dict(), "backend_version": "une-autre-machine"}
    )
    assert autre_machine._compute_semantic_hash() != m.semantic_hash, (
        "deux moteurs differents ne peuvent pas promettre le meme resultat"
    )
    # Et une archive dont on a reecrit ce champ ne passe plus la verification.
    with pytest.raises(ValueError):
        autre_machine.verify_self()


def test_deux_machines_IBM_ne_se_confondent_plus():
    """`backend_name` ne distingue pas les machines d'un meme fournisseur."""
    import cirq

    from qbridge.manifest import Manifest as M

    q = cirq.LineQubit.range(2)
    base = dict(
        circuit=cirq.Circuit([cirq.H(q[0]), cirq.measure(*q, key="m")]),
        backend_name="ibm-runtime",
        seed=7,
        repetitions=100,
        options={},
        noise_json=None,
    )
    marrakesh = M.build(backend_version="ibm_marrakesh", **base)
    fez = M.build(backend_version="ibm_fez", **base)

    assert marrakesh.backend_name == fez.backend_name
    assert marrakesh.semantic_hash != fez.semantic_hash
