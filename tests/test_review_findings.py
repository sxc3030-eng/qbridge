"""Non-regression sur les defauts trouves par la relecture externe.

Chaque test reproduit une attaque VERIFIEE comme fonctionnelle avant
correction. Aucun n'est hypothetique.
"""

from __future__ import annotations

import functools
import json

import cirq
import numpy as np
import pytest

from qbridge import Verdict, capture, capture_classical, replay, verify_source_unchanged
from qbridge.calibration import CalibrationSnapshot, synthetic_snapshot
from qbridge.capture import hash_samples
from qbridge.cli import EXIT_ERROR, EXIT_OK, main
from qbridge.record import RunRecord
from qbridge.signing import (
    Ed25519Signer,
    Ed25519Verifier,
    HmacSigner,
    Signature,
    SignatureScope,
    sign_manifest,
    sign_record,
    verify_manifest_signature,
    verify_record_signature,
)


def _bell_mesure():
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1), cirq.measure(q0, q1, key="m"))


def _record(repetitions=200):
    return RunRecord.from_capture(
        capture(_bell_mesure(), backend="qsim", seed=7, repetitions=repetitions)
    )


# ================= CRITIQUE 1 : la signature doit couvrir les tirages =========


def test_la_signature_couvre_les_tirages(tmp_path):
    """AVANT correction : `sign_manifest` ne signait que `content_hash`, qui ne
    couvre QUE la recette. `result_hash` vivait dans record.json, dans aucun
    hash et aucune signature. On remplacait samples.npz, on recalculait
    result_hash — public, deux lignes — sans toucher au manifeste ni a la
    signature, et l'archive se verifiait « valide et opposable ».

    Les bitstrings, seule donnee non regenerable de la chaine, etaient
    exactement ce que la signature n'atteignait pas.
    """
    record = _record()
    signer, _priv, pub = Ed25519Signer.generate("simon")
    signature = sign_record(record, signer)

    faux = {"m": np.zeros(record.samples["m"].shape, dtype=record.samples["m"].dtype)}
    substitue = RunRecord(
        schema_version=record.schema_version,
        manifest=record.manifest,  # INTOUCHE
        result_hash=hash_samples(faux),  # recalcule par l'attaquant
        samples=faux,
        state_vector_hash=record.state_vector_hash,
    )

    rapport = verify_record_signature(
        substitue, signature, Ed25519Verifier(pub, "simon")
    )
    assert rapport.valid is False
    assert rapport.binds_this_manifest is False


def test_une_signature_de_recette_ne_vaut_pas_pour_une_archive():
    """Une signature de RECETTE ne dit rien des resultats. Les confondre etait
    la faille : il faut que la portee soit liee dans le message signe."""
    record = _record()
    signer, _priv, pub = Ed25519Signer.generate("simon")
    sig_recette = sign_manifest(record.manifest, signer)

    assert sig_recette.scope == SignatureScope.MANIFEST.value
    rapport = verify_record_signature(
        record, sig_recette, Ed25519Verifier(pub, "simon")
    )
    assert rapport.valid is False
    assert "portee" in rapport.detail


def test_une_signature_d_archive_ne_vaut_pas_pour_une_recette():
    record = _record()
    signer, _priv, pub = Ed25519Signer.generate("simon")
    sig_archive = sign_record(record, signer)
    rapport = verify_manifest_signature(
        record.manifest, sig_archive, Ed25519Verifier(pub, "simon")
    )
    assert rapport.valid is False and "portee" in rapport.detail


def test_le_cycle_honnete_d_archive_reste_valide():
    record = _record()
    signer, _priv, pub = Ed25519Signer.generate("simon")
    rapport = verify_record_signature(
        record, sign_record(record, signer), Ed25519Verifier(pub, "simon")
    )
    assert rapport.valid is True and rapport.third_party_verifiable is True


def test_le_hash_d_archive_couvre_bien_les_tirages():
    a = _record()
    faux = {"m": np.zeros(a.samples["m"].shape, dtype=a.samples["m"].dtype)}
    b = RunRecord(
        schema_version=a.schema_version,
        manifest=a.manifest,
        result_hash=hash_samples(faux),
        samples=faux,
        state_vector_hash=a.state_vector_hash,
    )
    assert a.manifest.content_hash == b.manifest.content_hash, (
        "les recettes sont identiques : c'est ce qui rend le test significatif"
    )
    assert a.content_hash() != b.content_hash()


# ============ CRITIQUE 2 : le plafond doit regarder le backend de CAPTURE =====


def test_rejouer_une_archive_materielle_sur_qsim_ne_la_blanchit_pas():
    """AVANT correction : le plafond ne regardait que le backend de REJEU. Une
    archive capturee sur materiel, rejouee avec `--backend qsim`, ressortait
    BIT_EXACT. Le materiel n'a jamais produit un resultat bit-reproductible ;
    le rejouer sur un simulateur ne le rend pas tel."""
    q = tuple(cirq.LineQubit.range(2))
    snapshot = synthetic_snapshot(q)
    run = capture(
        _bell_mesure(),
        backend="hardware-sim",
        seed=7,
        repetitions=200,
        calibration=snapshot,
    )
    assert run.manifest.backend_name == "hardware-sim"

    sur_qsim = replay(run.manifest, backend="qsim")
    assert sur_qsim.verdict is not Verdict.BIT_EXACT
    assert sur_qsim.verdict >= Verdict.STATISTICALLY_COMPATIBLE


def test_un_rejeu_simulateur_sur_simulateur_reste_bit_exact():
    """Controle : sans lui, le test precedent ne distinguerait pas « plafonne »
    de « le harnais ne dit plus jamais BIT_EXACT »."""
    run = capture(_bell_mesure(), backend="qsim", seed=7, repetitions=200)
    assert replay(run.manifest, backend="qsim").verdict is Verdict.BIT_EXACT


# ================== HAUT 3 : le key_id doit etre verifie =====================


def test_un_key_id_qui_ne_correspond_pas_est_refuse():
    """AVANT correction : `verifier.key_id` n'etait compare a rien. Le message
    signe liait l'identite fournie par l'attaquant a elle-meme, donc
    `--key-id` pouvait valoir n'importe quoi."""
    record = _record(50)
    signer, _priv, pub = Ed25519Signer.generate("simon")
    signature = sign_record(record, signer)

    rapport = verify_record_signature(
        record, signature, Ed25519Verifier(pub, "JE-NE-SUIS-PAS-SIMON")
    )
    assert rapport.valid is False
    assert "identite de cle" in rapport.detail


def test_le_bon_key_id_passe_toujours():
    record = _record(50)
    signer, _priv, pub = Ed25519Signer.generate("simon")
    rapport = verify_record_signature(
        record, sign_record(record, signer), Ed25519Verifier(pub, "simon")
    )
    assert rapport.valid is True


# =========== HAUT 4 et 5 : identite du module, et decorateurs ================


def _ctx_pour(fn):
    return capture_classical(callables={"reduce": fn}, input_data=None)


def reduire_reference(donnees):
    """Fonction de reference, definie dans CE module."""
    return sum(donnees) / len(donnees)


def test_une_fonction_identique_d_un_AUTRE_module_est_une_derive(tmp_path, monkeypatch):
    """AVANT correction : seul le hash du TEXTE etait compare. Deux fonctions
    redigees a l'identique dans des modules differents resolvent des globales
    differentes et publient des resultats opposes."""
    import sys
    import textwrap

    source = textwrap.dedent(
        '''
        def reduire_reference(donnees):
            """Fonction de reference, definie dans CE module."""
            return sum(donnees) / len(donnees)
        '''
    ).strip()
    (tmp_path / "jumeau.py").write_text(source, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("jumeau", None)
    import jumeau

    ctx = _ctx_pour(reduire_reference)
    rapport = verify_source_unchanged(ctx, {"reduce": jumeau.reduire_reference})

    assert rapport.has_drift is True
    assert rapport.fully_verified is False
    assert rapport.detail["reduce"]["sealed_module"] != (
        rapport.detail["reduce"]["current_module"]
    )


def test_une_fonction_decoree_n_est_PAS_pleinement_verifiee():
    """AVANT correction : `inspect.unwrap` remonte a la fonction nue, donc un
    decorateur qui change le resultat passait pour « inchange » ET
    `fully_verified` valait True. L'avertissement existait ; personne ne le
    lisait."""

    def saboteur(fn):
        @functools.wraps(fn)
        def enveloppe(*args, **kwargs):
            return "SABOTE"

        return enveloppe

    ctx = _ctx_pour(reduire_reference)
    decoree = saboteur(reduire_reference)
    assert decoree([1, 2]) != reduire_reference([1, 2])

    rapport = verify_source_unchanged(ctx, {"reduce": decoree})
    assert rapport.fully_verified is False
    assert "reduce" in rapport.partially_sealed


def test_le_code_intact_reste_pleinement_verifie():
    """Controle indispensable : les deux tests precedents doivent distinguer
    « detecte » de « refuse tout »."""
    ctx = _ctx_pour(reduire_reference)
    rapport = verify_source_unchanged(ctx, {"reduce": reduire_reference})
    assert rapport.has_drift is False
    assert rapport.fully_verified is True
    assert rapport.partially_sealed == ()


# ======= HAUT 6 : cirq-reference ne doit pas jeter une option scellee ========


def test_cirq_reference_refuse_de_jeter_une_option_scellee():
    """AVANT correction : `if nom == "cirq-reference": options = {}` jetait en
    silence des options SEMANTIC scellees. Un changement de backend suffisait a
    contourner le verrou que `override_performance` refuse bruyamment."""
    q = cirq.LineQubit.range(2)
    mid = cirq.Circuit(
        cirq.H(q[0]),
        cirq.CX(q[0], q[1]),
        cirq.measure(q[0], key="a"),
        cirq.X(q[0]) ** 0.5,
        cirq.measure(q[0], q[1], key="b"),
    )
    run = capture(
        mid, backend="qsim", seed=7, repetitions=100, options={"cpu_threads": 4}
    )
    assert run.manifest.semantic_options == {"cpu_threads": 4}

    with pytest.raises(ValueError, match="n'accepte aucune option"):
        replay(run.manifest, backend="cirq-reference")


def test_cirq_reference_reste_utilisable_sans_option_scellee():
    q0, q1 = cirq.LineQubit.range(2)
    run = capture(cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1)), backend="qsim", seed=7)
    assert replay(run.manifest, backend="cirq-reference").verdict <= (
        Verdict.NUMERICALLY_EQUIVALENT
    )


# ====== HAUT 7 : le code 2 d'argparse ne doit pas se lire « DIVERGENT » ======


def test_une_erreur_d_usage_ne_rend_pas_le_code_de_divergence(tmp_path):
    """argparse rend 2 pour toute erreur d'usage, or 2 est reserve a DIVERGENT.
    Une CI qui se trompe de nom d'option lirait une divergence physique."""
    from qbridge.cli import EXIT_DIVERGENT

    for argv in (["verify"], ["replay", str(tmp_path), "--option-inventee"], []):
        code = main(argv)
        assert code != EXIT_DIVERGENT, f"{argv} rend le code de divergence"
        assert code == EXIT_ERROR


def test_help_rend_toujours_zero():
    assert main(["--help"]) == EXIT_OK


# ====== MOYEN 9 : l'erreur de lecture ne vise que les qubits mesures =========


def test_l_erreur_de_lecture_ne_touche_que_les_qubits_mesures():
    """AVANT correction : la branche parcourait `moment.qubits` — TOUS les
    qubits du moment. Une inversion de bit reelle etait injectee dans l'etat de
    qubits non lus, et persistait ensuite dans tout le circuit : une propriete
    de l'appareil de mesure transformee en erreur physique sur des voisins."""
    q = tuple(cirq.LineQubit.range(3))
    snapshot = synthetic_snapshot(q)
    modele = snapshot.noise_model()

    moment = cirq.Moment([cirq.measure(q[0], key="m"), cirq.X(q[1]), cirq.X(q[2])])
    sortie = modele.noisy_moment(moment, list(q))

    inversions = [
        op
        for element in sortie
        if isinstance(element, cirq.Moment)
        for op in element.operations
        if "bit_flip" in str(op.gate)
    ]
    vises = {q_ for op in inversions for q_ in op.qubits}
    assert vises == {q[0]}, (
        f"l'erreur de lecture vise {sorted(vises)} alors que seul {q[0]} est mesure"
    )


# ===== le plafond ne doit dependre d'AUCUNE construction d'objet =============


def test_le_plafond_ne_depend_pas_de_la_construction_d_un_temoin(monkeypatch):
    """AVANT correction (v0.8) : le plafond construisait un backend temoin pour
    interroger le backend de CAPTURE, dans un `try/except Exception: pass`. Des
    que cette construction echouait, le plafond cessait de s'appliquer EN
    SILENCE et le blanchiment d'archive materielle redevenait possible.

    Mesure : avec le temoin sabote, un rejeu sur qsim ressortait BIT_EXACT.

    Une exception avalee qui desactive un controle de securite est exactement
    le defaut que ce plafond existe pour empecher. La replicabilite est
    desormais un attribut de CLASSE, lu sans rien construire.
    """
    from qbridge.backends.hardware import SimulatedHardwareBackend

    neutre = CalibrationSnapshot.build(
        device_id="neutre",
        device_version="1",
        qubits={},
        gates={},
        basis_gates=[],
        coupling_map=[],
    )
    run = capture(
        _bell_mesure(),
        backend="hardware-sim",
        seed=7,
        repetitions=200,
        calibration=neutre,
    )

    original = SimulatedHardwareBackend.__init__

    def sabotage_du_temoin(self, calibration=None):
        # Le temoin etait construit avec calibration=None ; l'execution reelle
        # recoit toujours une calibration.
        if calibration is None:
            raise RuntimeError("temoin non constructible")
        original(self, calibration)

    monkeypatch.setattr(SimulatedHardwareBackend, "__init__", sabotage_du_temoin)

    rapport = replay(run.manifest, backend="qsim")
    assert rapport.verdict is not Verdict.BIT_EXACT, (
        "le plafond a saute parce qu'un objet n'a pas pu etre construit"
    )
    assert "plafonne" in rapport.detail


def test_un_backend_de_capture_inconnu_est_traite_comme_non_reproductible():
    """On ne peut pas PROUVER qu'un backend inconnu est bit-reproductible :
    on suppose donc qu'il ne l'est pas. Se taire laisserait passer un verdict
    trop fort pour un backend dont on ignore tout."""
    from qbridge.replay import _plafonner_selon_le_backend
    from qbridge.backends.qsim import QsimBackend
    from qbridge.verdict import ComparisonResult

    depart = ComparisonResult(Verdict.BIT_EXACT, "octets identiques")
    plafonne = _plafonner_selon_le_backend(
        depart, QsimBackend(), "backend-qui-n-existe-pas"
    )
    assert plafonne.verdict is Verdict.STATISTICALLY_COMPATIBLE
    assert "inconnu" in plafonne.detail


def test_la_replicabilite_est_lisible_sans_instancier():
    """C'est ce qui rend la construction du temoin inutile."""
    from qbridge.backends import BACKENDS

    for nom, classe in BACKENDS.items():
        assert isinstance(classe.BIT_EXACT_REPLAYABLE, bool), nom


def test_classe_et_instance_ne_peuvent_pas_diverger():
    from qbridge.backends import BACKENDS
    from qbridge.backends.hardware import SimulatedHardwareBackend

    assert SimulatedHardwareBackend(None).is_bit_exact_replayable() is False
    for nom, classe in BACKENDS.items():
        if nom == "hardware-sim":
            continue
        assert classe().is_bit_exact_replayable() is classe.BIT_EXACT_REPLAYABLE


def test_une_archive_sans_tirages_ne_se_confond_pas_avec_une_archive_vide():
    """`samples=None` (mode vecteur d'etat) et `samples={}` (des tirages, zero
    cle) sont deux etats differents ; ils partageaient une empreinte."""
    from qbridge.capture import hash_samples

    base = _record(50)
    sans = RunRecord(
        schema_version=base.schema_version,
        manifest=base.manifest,
        result_hash=hash_samples({}),
        samples=None,
        state_vector_hash=base.state_vector_hash,
    )
    vide = RunRecord(
        schema_version=base.schema_version,
        manifest=base.manifest,
        result_hash=hash_samples({}),
        samples={},
        state_vector_hash=base.state_vector_hash,
    )
    assert sans.content_hash() != vide.content_hash()


def test_aucun_backend_ne_scelle_un_moteur_qu_il_n_utilise_PAS():
    """DEFAUT 28, et c'est le defaut 21 sous un autre nom.

    Le manifeste scelle une empreinte du noyau qsim — jeu d'instructions SIMD,
    mode GPU — dans le hash SEMANTIQUE. Pour un backend qui n'appelle jamais
    qsim, cela decrit un moteur qui n'a pas tourne.

    Corrige ce matin pour `ibm-runtime` seulement, la ou je l'avais remarque.
    `cirq-reference` enveloppe `cirq.Simulator` et portait toujours le defaut :
    reparer une instance ne repare pas la classe. Ce test balaye TOUS les
    backends pour que la question ne se repose plus un par un.
    """
    import cirq

    from qbridge import capture
    from qbridge.backends import BACKENDS

    utilise_qsim = {"qsim", "hardware-sim"}  # verifie : les deux importent qsimcirq

    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([cirq.H(q[0]), cirq.measure(*q, key="m")])

    for nom in sorted(BACKENDS):
        if nom == "hardware-sim":
            continue  # exige une calibration, couvert par ses propres tests
        manifeste = capture(circuit, backend=nom, seed=7, repetitions=20).manifest
        if nom in utilise_qsim:
            assert manifeste.kernel, f"{nom} utilise qsim : son noyau doit etre scelle"
        else:
            assert manifeste.kernel == {}, (
                f"{nom} n'utilise pas qsim : sceller son noyau decrirait un "
                "moteur qui n'a pas tourne"
            )
