"""Tests du versant classique.

Regle de ces tests : aucun ne doit pouvoir passer parce qu'une exception a ete
avalee. Chaque chemin d'echec est verifie par sa RAISON precise, et le chemin
nominal est verifie a cote pour qu'un `except: pass` generalise fasse tomber
la suite au lieu de la faire passer.
"""

import functools
import importlib
import json
import linecache
import sys

import numpy as np
import pytest

from qbridge.classical import (
    ABSENT,
    CLASSICAL_SCHEMA_VERSION,
    REASON_BUILTIN,
    REASON_LAMBDA,
    REASON_NOT_A_FUNCTION,
    REASON_NO_SOURCE,
    WARNING_DECORATED,
    CallableCapture,
    ClassicalContext,
    Evidence,
    capture_callable,
    capture_classical,
    environment_lock,
    hash_input_data,
    installed_distributions,
    normalize_input,
    verify_source_unchanged,
)
from qbridge.digest import canonical_json, sha256_of, sha256_of_text
from qbridge.fingerprint import environment_fingerprint

# --------------------------------------------------------------------------
# Materiel de test : deux reductions differentes, une preparation
# --------------------------------------------------------------------------


def _prepare_circuit(n_qubits):
    """Role `prepare` : construirait le circuit."""
    return [("H", q) for q in range(n_qubits)]


def _reduce_v1(shots):
    """Role `reduce`, version publiee."""
    return float(np.mean(shots))


def _reduce_v2(shots):
    """Role `reduce`, version corrigee apres publication : AUTRE resultat."""
    return float(np.mean(shots)) - 0.5


_LAMBDA_A, _LAMBDA_B = (lambda x: x + 1), (lambda x: x + 2)


def _decorateur(fn):
    @functools.wraps(fn)
    def enveloppe(*args, **kwargs):
        return fn(*args, **kwargs)

    return enveloppe


@_decorateur
def _reduce_decore(shots):
    return len(shots)


class _Reducteur:
    def reduire(self, shots):
        return sum(shots)


def _fonction_sans_fichier():
    """Compile un vrai objet fonction dont le fichier n'existe nulle part.

    C'est le cas du REPL et de `exec` : `co_filename` designe une chaine que
    `linecache` ne connait pas. Le nom est unique pour qu'aucune autre partie
    du processus ne puisse l'avoir enregistre.
    """
    code = compile(
        "def reduce_repl(shots):\n    return sum(shots)\n",
        "<qbridge-test-synthetique>",
        "exec",
    )
    espace = {}
    exec(code, espace)
    return espace["reduce_repl"]


def _ctx(**kwargs):
    """Contexte de test : sans l'empreinte quantique, sauf demande contraire."""
    kwargs.setdefault("include_quantum_fingerprint", False)
    return capture_classical(**kwargs)


# --------------------------------------------------------------------------
# Capture de source : le chemin nominal
# --------------------------------------------------------------------------


def test_scelle_le_texte_reel_de_la_fonction():
    c = capture_callable("reduce", _reduce_v1)
    assert c.evidence == Evidence.CAPTURED.value
    assert c.reason is None
    # Falsifiable : c'est le corps qu'on exige, pas une chaine vide.
    assert "def _reduce_v1(shots):" in c.source
    assert "return float(np.mean(shots))" in c.source
    assert c.module == __name__
    assert c.qualname == "_reduce_v1"
    assert c.source_file.endswith("test_classical.py")
    assert c.first_line > 0


def test_le_hash_est_bien_celui_du_texte_scelle():
    c = capture_callable("reduce", _reduce_v1)
    assert c.source_hash == sha256_of_text(c.source)
    assert len(c.source_hash) == 64


def test_le_hash_change_quand_le_corps_change():
    a = capture_callable("reduce", _reduce_v1)
    b = capture_callable("reduce", _reduce_v2)
    assert a.is_verifiable and b.is_verifiable
    assert a.source != b.source
    assert a.source_hash != b.source_hash


def test_le_hash_est_stable_pour_la_meme_fonction():
    assert (
        capture_callable("reduce", _reduce_v1).source_hash
        == capture_callable("autre_role", _reduce_v1).source_hash
    )


def test_une_methode_liee_est_capturee():
    c = capture_callable("reduce", _Reducteur().reduire)
    assert c.evidence == Evidence.CAPTURED.value
    assert "def reduire(self, shots):" in c.source


# --------------------------------------------------------------------------
# Capture de source : les trois modes d'echec reels
# --------------------------------------------------------------------------


def test_builtin_c_est_unavailable_et_ne_leve_pas():
    c = capture_callable("reduce", len)
    assert c.evidence == Evidence.UNAVAILABLE.value
    assert c.reason == REASON_BUILTIN
    assert c.source is None and c.source_hash is None
    assert not c.is_verifiable
    assert c.qualname == "len"


def test_fonction_sans_fichier_est_unavailable_avec_l_erreur_reelle():
    c = capture_callable("reduce", _fonction_sans_fichier())
    assert c.evidence == Evidence.UNAVAILABLE.value
    assert c.reason.startswith(REASON_NO_SOURCE)
    # La cause est conservee : sans elle, l'indisponibilite est un blanc.
    assert "OSError" in c.reason
    assert c.source_hash is None


def test_lambda_est_unavailable_malgre_une_source_lisible():
    c = capture_callable("reduce", _LAMBDA_A)
    assert c.evidence == Evidence.UNAVAILABLE.value
    assert c.reason == REASON_LAMBDA
    assert c.source is not None  # conserve pour l'humain...
    assert c.source_hash is None  # ...mais jamais promu en preuve
    assert not c.is_verifiable


def test_pourquoi_le_lambda_est_refuse_deux_lambdas_meme_texte():
    """La justification du refus, verifiee et non postulee."""
    a = capture_callable("a", _LAMBDA_A)
    b = capture_callable("b", _LAMBDA_B)
    assert _LAMBDA_A(0) != _LAMBDA_B(0)  # comportements differents
    assert a.source == b.source  # texte identique : il ne prouve rien


def test_partial_est_unavailable_avec_sa_propre_raison():
    c = capture_callable("reduce", functools.partial(_reduce_v1))
    assert c.evidence == Evidence.UNAVAILABLE.value
    assert c.reason == REASON_NOT_A_FUNCTION


def test_un_non_callable_leve_au_lieu_d_etre_enregistre():
    """Un role qui ne designe pas un callable est un bug d'appel, pas une
    indisponibilite : il doit crier."""
    with pytest.raises(TypeError):
        capture_callable("reduce", 42)
    with pytest.raises(TypeError):
        _ctx(reduce=42)


def test_fonction_decoree_capture_l_origine_et_signale_le_trou():
    c = capture_callable("reduce", _reduce_decore)
    assert c.evidence == Evidence.CAPTURED.value
    # getsourcelines deballe : c'est bien le corps d'origine qui est scelle.
    assert "def _reduce_decore(shots):" in c.source
    assert "return len(shots)" in c.source
    # Mais le corps du decorateur, lui, n'y est pas.
    assert "def enveloppe" not in c.source
    assert WARNING_DECORATED in c.warnings
    # Le fichier doit designer le texte scelle, pas celui du decorateur.
    assert c.source_file.endswith("test_classical.py")


# --------------------------------------------------------------------------
# Hachage de l'entree
# --------------------------------------------------------------------------


def test_hash_d_entree_distingue_le_dtype():
    a = np.array([1, 0], dtype=np.complex64)
    b = np.array([1, 0], dtype=np.complex128)
    assert hash_input_data(a) != hash_input_data(b)


def test_hash_d_entree_distingue_la_forme():
    a = np.zeros((2, 2), dtype=np.uint8)
    b = np.zeros((4,), dtype=np.uint8)
    assert hash_input_data(a) != hash_input_data(b)


def test_hash_d_entree_distingue_les_valeurs():
    a = np.array([0, 1], dtype=np.uint8)
    b = np.array([1, 0], dtype=np.uint8)
    assert hash_input_data(a) != hash_input_data(b)


def test_hash_d_entree_est_stable():
    donnee = {"shots": np.arange(6, dtype=np.uint8).reshape(2, 3), "n": 3}
    assert hash_input_data(donnee) == hash_input_data(
        {"n": 3, "shots": np.arange(6, dtype=np.uint8).reshape(2, 3)}
    )


def test_hash_d_entree_traverse_les_structures_imbriquees():
    base = {"lots": [np.zeros(2, np.uint8), {"x": np.zeros(2, np.uint8)}]}
    autre = {"lots": [np.zeros(2, np.uint8), {"x": np.ones(2, np.uint8)}]}
    assert hash_input_data(base) != hash_input_data(autre)


def test_normalize_input_delegue_a_sha256_of_array():
    a = np.zeros((2, 3), np.float32)
    d = normalize_input(a)
    assert d["dtype"] == "float32" and d["shape"] == [2, 3]
    # Rien n'est re-implemente : le condense est celui de digest.py.
    from qbridge.digest import sha256_of_array

    assert d["__ndarray__"] == sha256_of_array(a)


def test_entree_absente_est_not_applicable():
    ctx = _ctx(reduce=_reduce_v1)
    assert ctx.evidence["input_data"] == Evidence.NOT_APPLICABLE.value
    assert ctx.input_hash is None and ctx.input_kind is None
    assert ctx.input_error is None


def test_entree_none_est_hachee_et_distincte_de_l_absence():
    ctx = _ctx(reduce=_reduce_v1, input_data=None)
    assert ctx.evidence["input_data"] == Evidence.DERIVED.value
    assert ctx.input_hash == sha256_of(None)
    assert ctx.input_kind == "NoneType"
    assert ABSENT is not None


def test_hash_d_entree_refuse_nan_et_le_dit():
    """`canonical_json` refuse NaN par construction : la capture l'enregistre
    comme indisponibilite motivee, elle ne l'invente pas."""
    with pytest.raises(ValueError):
        hash_input_data({"p": float("nan")})
    ctx = _ctx(input_data={"p": float("nan")})
    assert ctx.evidence["input_data"] == Evidence.UNAVAILABLE.value
    assert ctx.input_hash is None
    assert "ValueError" in ctx.input_error


def test_hash_d_entree_refuse_un_objet_opaque_et_le_dit():
    with pytest.raises(TypeError):
        hash_input_data(object())
    ctx = _ctx(input_data=object())
    assert ctx.evidence["input_data"] == Evidence.UNAVAILABLE.value
    assert "TypeError" in ctx.input_error
    assert ctx.input_kind == "object"


# --------------------------------------------------------------------------
# Verrou d'environnement
# --------------------------------------------------------------------------


def test_distributions_forme_nom_version_triee():
    d = installed_distributions()
    assert d
    assert all("==" in ligne for ligne in d)
    assert any(ligne.startswith("numpy==") for ligne in d)
    # Le tri porte sur le NOM, pas sur la ligne rendue : « cirq » precede
    # « cirq-aqt », alors que « cirq-aqt==... » precederait « cirq==... » en
    # tri de chaines ('-' < '='). C'est le nom qui fait l'ordre attendu.
    noms = [ligne.split("==", 1)[0] for ligne in d]
    assert noms == sorted(noms)


def test_distributions_sans_doublon():
    d = installed_distributions()
    noms = [ligne.split("==", 1)[0] for ligne in d]
    assert len(noms) == len(set(noms))
    assert all(n == n.lower() for n in noms)  # normalisation PEP 503


def test_distributions_deterministe_dans_un_meme_processus():
    assert installed_distributions() == installed_distributions()


def test_environnement_compose_l_empreinte_au_lieu_de_la_refaire():
    lock = environment_lock(include_quantum_fingerprint=True)
    assert lock["quantum_fingerprint_error"] is None
    assert set(lock["quantum_fingerprint"]) == set(environment_fingerprint())
    assert lock["quantum_fingerprint_source"].endswith("environment_fingerprint")


def test_le_bloc_interpreteur_ne_duplique_rien_de_l_empreinte():
    lock = environment_lock(include_quantum_fingerprint=True)
    assert set(lock["interpreter"]) & set(lock["quantum_fingerprint"]) == set()
    assert lock["interpreter"]["executable"].endswith("python.exe") or lock[
        "interpreter"
    ]["executable"].endswith("python")
    assert sys.version.split()[0] in lock["interpreter"]["version_full"]


def test_empreinte_quantique_desactivee_est_not_applicable():
    ctx = _ctx(reduce=_reduce_v1)
    assert ctx.evidence["quantum_fingerprint"] == Evidence.NOT_APPLICABLE.value
    assert ctx.environment["quantum_fingerprint"] is None
    # Le reste du verrou reste capture : la desactivation est ciblee.
    assert ctx.evidence["distributions"] == Evidence.CAPTURED.value
    assert ctx.environment["distribution_count"] > 0


def test_les_cles_d_erreur_existent_toujours():
    """Une cle absente serait ambigue : elles sont a None, pas supprimees."""
    lock = environment_lock(include_quantum_fingerprint=False)
    assert lock["distributions_error"] is None
    assert lock["quantum_fingerprint_error"] is None
    assert "quantum_fingerprint" in lock


# --------------------------------------------------------------------------
# Marqueurs de preuve
# --------------------------------------------------------------------------


def test_les_quatre_marqueurs_sont_atteignables():
    bon = _ctx(prepare=_prepare_circuit, reduce=_reduce_v1, input_data={"n": 2})
    mauvais = _ctx(callables={"builtin": len}, input_data={"p": float("inf")})
    obtenus = set(bon.evidence.values()) | set(mauvais.evidence.values())
    assert obtenus == {e.value for e in Evidence}


def test_agregat_callables_pessimiste_des_qu_un_role_echoue():
    assert (
        _ctx(prepare=_prepare_circuit, reduce=_reduce_v1).evidence["callables"]
        == Evidence.CAPTURED.value
    )
    assert (
        _ctx(prepare=_prepare_circuit, callables={"b": len}).evidence["callables"]
        == Evidence.UNAVAILABLE.value
    )
    assert _ctx().evidence["callables"] == Evidence.NOT_APPLICABLE.value


def test_role_fourni_deux_fois_refuse():
    with pytest.raises(ValueError):
        _ctx(reduce=_reduce_v1, callables={"reduce": _reduce_v2})


def test_verifiable_roles_exclut_les_indisponibles():
    ctx = _ctx(reduce=_reduce_v1, callables={"b": len, "l": _LAMBDA_A})
    assert ctx.verifiable_roles() == ("reduce",)


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------


def test_serialisable_par_canonical_json_et_sha256_of():
    ctx = capture_classical(
        prepare=_prepare_circuit,
        reduce=_reduce_v1,
        callables={"b": len, "l": _LAMBDA_A},
        input_data={"shots": np.zeros((4, 2), np.uint8)},
    )
    texte = canonical_json(ctx.to_dict())
    assert len(sha256_of(ctx.to_dict())) == 64
    assert json.loads(texte)["schema_version"] == CLASSICAL_SCHEMA_VERSION


def test_round_trip_json_reconstruit_a_l_identique():
    ctx = _ctx(
        prepare=_prepare_circuit,
        callables={"b": len},
        input_data=[1, 2, np.zeros(2, np.uint8)],
    )
    retour = ClassicalContext.from_dict(json.loads(canonical_json(ctx.to_dict())))
    assert retour == ctx
    assert retour.to_dict() == ctx.to_dict()
    assert isinstance(retour.callables["prepare"], CallableCapture)
    assert retour.callables["prepare"].warnings == ()


def test_round_trip_conserve_les_avertissements():
    ctx = _ctx(reduce=_reduce_decore)
    retour = ClassicalContext.from_dict(json.loads(canonical_json(ctx.to_dict())))
    assert retour.callables["reduce"].warnings == (WARNING_DECORATED,)
    assert retour == ctx


def test_embarquable_comme_un_seul_champ_de_manifeste():
    """Le contexte doit tenir dans un champ JSON sans traitement particulier."""
    ctx = _ctx(reduce=_reduce_v1)
    enveloppe = {"classical_context": ctx.to_dict(), "circuit_hash": "0" * 64}
    relu = json.loads(canonical_json(enveloppe))
    assert ClassicalContext.from_dict(relu["classical_context"]) == ctx


# --------------------------------------------------------------------------
# Hash du contexte
# --------------------------------------------------------------------------


def test_context_hash_ignore_l_horodatage():
    a = _ctx(reduce=_reduce_v1)
    b = _ctx(reduce=_reduce_v1)
    assert a.created_at != b.created_at or True  # l'horloge peut ne pas bouger
    assert a.context_hash == b.context_hash


def test_context_hash_change_si_le_code_change():
    assert _ctx(reduce=_reduce_v1).context_hash != _ctx(reduce=_reduce_v2).context_hash


def test_context_hash_change_si_l_entree_change():
    a = _ctx(reduce=_reduce_v1, input_data={"n": 1})
    b = _ctx(reduce=_reduce_v1, input_data={"n": 2})
    assert a.context_hash != b.context_hash


def test_verify_integrity_detecte_une_falsification():
    ctx = _ctx(reduce=_reduce_v1)
    ctx.verify_integrity()  # intact

    donnees = json.loads(canonical_json(ctx.to_dict()))
    donnees["callables"]["reduce"]["source"] = "def _reduce_v1(shots):\n    return 0.0\n"
    falsifie = ClassicalContext.from_dict(donnees)
    with pytest.raises(ValueError, match="integrite"):
        falsifie.verify_integrity()


# --------------------------------------------------------------------------
# Detection de derive
# --------------------------------------------------------------------------


def test_aucune_derive_sur_du_code_inchange():
    ctx = _ctx(prepare=_prepare_circuit, reduce=_reduce_v1)
    r = verify_source_unchanged(
        ctx, {"prepare": _prepare_circuit, "reduce": _reduce_v1}
    )
    assert r.unchanged == ("prepare", "reduce")
    assert r.drifted == () and not r.has_drift
    assert r.fully_verified


def test_derive_detectee_quand_la_reduction_a_change():
    ctx = _ctx(prepare=_prepare_circuit, reduce=_reduce_v1)
    r = verify_source_unchanged(
        ctx, {"prepare": _prepare_circuit, "reduce": _reduce_v2}
    )
    assert r.drifted == ("reduce",)
    assert r.unchanged == ("prepare",)
    assert r.has_drift and not r.fully_verified
    d = r.detail["reduce"]
    assert d["status"] == "drifted"
    assert d["sealed_hash"] != d["current_hash"]
    assert d["sealed_hash"] and d["current_hash"]


def test_un_role_non_scellable_n_est_jamais_declare_inchange():
    ctx = _ctx(callables={"b": len})
    r = verify_source_unchanged(ctx, {"b": len})
    assert r.unchanged == () and r.drifted == ()
    assert REASON_BUILTIN in r.unverifiable["b"]
    assert not r.fully_verified


def test_role_manquant_signale_sans_etre_confondu_avec_une_derive():
    ctx = _ctx(prepare=_prepare_circuit, reduce=_reduce_v1)
    r = verify_source_unchanged(ctx, {"reduce": _reduce_v1})
    assert r.missing == ("prepare",)
    assert r.unchanged == ("reduce",)
    assert r.drifted == ()
    assert not r.fully_verified


def test_role_inconnu_signale_sans_polluer_le_verdict():
    ctx = _ctx(reduce=_reduce_v1)
    r = verify_source_unchanged(ctx, {"reduce": _reduce_v1, "posttraitement": _reduce_v2})
    assert r.unknown == ("posttraitement",)
    assert r.unchanged == ("reduce",)
    assert r.fully_verified  # un role en trop ne casse pas le sceau


def test_rapport_de_derive_serialisable():
    ctx = _ctx(reduce=_reduce_v1, callables={"b": len})
    r = verify_source_unchanged(ctx, {"reduce": _reduce_v2, "b": len})
    canonical_json(r.to_dict())
    assert r.to_dict()["drifted"] == ["reduce"]


# --------------------------------------------------------------------------
# Le cas reel : un fichier de reduction modifie entre deux dates
# --------------------------------------------------------------------------


def _module_temporaire(tmp_path, monkeypatch, nom, source):
    chemin = tmp_path / f"{nom}.py"
    chemin.write_text(source, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    return chemin


def test_derive_reelle_dans_un_fichier_importe(tmp_path, monkeypatch):
    """Le scenario que le module existe pour couvrir : le fichier de reduction
    a change entre la publication et le rejeu."""
    nom = "qbridge_reduction_publiee"
    chemin = _module_temporaire(
        tmp_path,
        monkeypatch,
        nom,
        "def reduce_shots(shots):\n    return sum(shots) / len(shots)\n",
    )
    try:
        module = importlib.import_module(nom)
        ctx = _ctx(reduce=module.reduce_shots)
        assert verify_source_unchanged(ctx, {"reduce": module.reduce_shots}).fully_verified

        chemin.write_text(
            "def reduce_shots(shots):\n"
            "    # correction appliquee apres la publication\n"
            "    return sum(shots) / max(len(shots) - 1, 1)\n",
            encoding="utf-8",
        )
        linecache.checkcache(str(chemin))
        module = importlib.reload(module)
        assert module.reduce_shots([1, 1, 1]) != 1.0  # le comportement a change

        r = verify_source_unchanged(ctx, {"reduce": module.reduce_shots})
        assert r.drifted == ("reduce",)
        assert r.has_drift and not r.fully_verified
    finally:
        sys.modules.pop(nom, None)


def test_trou_connu_la_source_suit_le_disque_pas_le_bytecode(tmp_path, monkeypatch):
    """Limite structurelle, verrouillee pour qu'elle reste visible.

    Si le fichier est modifie SANS reimport, la fonction vivante execute encore
    l'ancien bytecode alors que la capture rend le NOUVEAU texte. Le bytecode
    n'expose pas son texte d'origine : `verify_source_unchanged` ne peut donc
    pas prouver que le texte scelle est celui qui s'executera.
    """
    nom = "qbridge_reduction_perimee"
    chemin = _module_temporaire(
        tmp_path, monkeypatch, nom, "def reduce_shots(shots):\n    return 1\n"
    )
    try:
        module = importlib.import_module(nom)
        chemin.write_text(
            "def reduce_shots(shots):\n    # texte modifie, jamais reimporte\n"
            "    return 999\n",
            encoding="utf-8",
        )
        capture = capture_callable("reduce", module.reduce_shots)
        assert module.reduce_shots([]) == 1  # l'ancien bytecode tourne encore
        assert "return 999" in capture.source  # la capture voit le nouveau texte
        assert "return 1\n" not in capture.source
    finally:
        sys.modules.pop(nom, None)
