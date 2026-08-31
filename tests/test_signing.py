"""Tests de la signature detachee.

Chaque test d'attaque reproduit une facon concrete de tromper un verifieur
naif. Le module doit les refuser toutes, et les refuser en disant LAQUELLE :
« document altere », « signature qui vise autre chose » et « signature
invalide » sont trois situations differentes.
"""

from __future__ import annotations

import json
import os

import cirq
import pytest

from qbridge import capture
from qbridge.manifest import Manifest
from qbridge.signing import (
    SIGNATURE_SCHEMA_VERSION,
    Ed25519Signer,
    Ed25519Verifier,
    HmacSigner,
    Signature,
    SignatureAlgorithm,
    sign_manifest,
    signing_payload,
    verify_manifest_signature,
)


def _bell():
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1))


def _manifest(seed=7):
    return capture(_bell(), backend="qsim", seed=seed).manifest


@pytest.fixture
def cle():
    return os.urandom(32)


@pytest.fixture
def hmac_signer(cle):
    return HmacSigner(cle, key_id="ci-interne")


# ---------- HMAC : le chemin nominal ----------


def test_signer_puis_verifier(hmac_signer):
    m = _manifest()
    sig = sign_manifest(m, hmac_signer)
    r = verify_manifest_signature(m, sig, hmac_signer)
    assert r.valid and r.manifest_intact and r.binds_this_manifest and r.signature_valid


def test_la_signature_porte_le_hash_de_contenu(hmac_signer):
    m = _manifest()
    assert sign_manifest(m, hmac_signer).content_hash == m.content_hash


def test_aller_retour_disque(tmp_path, hmac_signer):
    m = _manifest()
    chemin = tmp_path / "signature.json"
    sign_manifest(m, hmac_signer).save(chemin)
    assert verify_manifest_signature(m, Signature.load(chemin), hmac_signer).valid


def test_une_signature_hmac_n_est_PAS_opposable(hmac_signer):
    """Elle est valide, mais quiconque peut la verifier peut aussi la forger."""
    m = _manifest()
    r = verify_manifest_signature(m, sign_manifest(m, hmac_signer), hmac_signer)
    assert r.valid is True
    assert r.third_party_verifiable is False
    assert "SYMETRIQUE" in r.detail


# ---------- attaques ----------


def test_un_manifeste_altere_est_refuse(hmac_signer):
    """Toucher au manifeste apres signature doit invalider."""
    m = _manifest()
    sig = sign_manifest(m, hmac_signer)
    altere = Manifest.from_dict({**m.to_dict(), "backend_version": "0.0.0-mensonge"})
    r = verify_manifest_signature(altere, sig, hmac_signer)
    assert r.valid is False
    assert r.manifest_intact is False


def test_une_signature_authentique_d_un_AUTRE_document_est_refusee(hmac_signer):
    """Attaque de substitution : la signature est vraie, le document n'est pas
    celui qu'elle vise. Un verifieur naif qui se contenterait de valider la
    cryptographie l'accepterait."""
    a = _manifest(seed=7)
    b = _manifest(seed=8)
    sig_de_a = sign_manifest(a, hmac_signer)

    r = verify_manifest_signature(b, sig_de_a, hmac_signer)
    assert r.valid is False
    assert r.manifest_intact is True, "le document b est parfaitement coherent"
    assert r.binds_this_manifest is False
    assert r.signature_valid is True, "la signature EST authentique, pour a"
    assert "autre document" in r.detail


def test_une_mauvaise_cle_est_refusee(cle):
    m = _manifest()
    sig = sign_manifest(m, HmacSigner(cle, key_id="alice"))
    autre = HmacSigner(os.urandom(32), key_id="alice")
    r = verify_manifest_signature(m, sig, autre)
    assert r.valid is False and r.signature_valid is False


def test_changer_le_key_id_declare_invalide_la_signature(cle):
    """Le key_id est LIE dans le message signe : le reecrire casse tout.
    Sans ce lien, on pourrait attribuer une signature a une autre cle."""
    m = _manifest()
    signer = HmacSigner(cle, key_id="alice")
    sig = sign_manifest(m, signer)
    usurpe = Signature.from_dict({**sig.to_dict(), "key_id": "bob"})
    assert verify_manifest_signature(m, usurpe, signer).signature_valid is False


def test_changer_l_algorithme_declare_est_refuse(cle):
    """Substitution d'algorithme : la signature declare autre chose que ce que
    le verifieur sait faire."""
    m = _manifest()
    signer = HmacSigner(cle, key_id="alice")
    sig = sign_manifest(m, signer)
    substitue = Signature.from_dict({**sig.to_dict(), "algorithm": "ed25519"})
    r = verify_manifest_signature(m, substitue, signer)
    assert r.valid is False and "algorithme incompatible" in r.detail


def test_une_signature_tronquee_est_refusee(hmac_signer):
    m = _manifest()
    sig = sign_manifest(m, hmac_signer)
    tronquee = Signature.from_dict({**sig.to_dict(), "signature": sig.signature[:32]})
    assert verify_manifest_signature(m, tronquee, hmac_signer).valid is False


def test_une_signature_non_hexadecimale_ne_fait_pas_planter(hmac_signer):
    m = _manifest()
    sig = sign_manifest(m, hmac_signer)
    pourrie = Signature.from_dict({**sig.to_dict(), "signature": "pas-de-l-hexa!!"})
    r = verify_manifest_signature(m, pourrie, hmac_signer)
    assert r.valid is False and "illisible" in r.detail


def test_signer_un_manifeste_incoherent_est_refuse(hmac_signer):
    """Signer un document incoherent reviendrait a attester l'incoherence."""
    m = _manifest()
    casse = Manifest.from_dict(
        {**m.to_dict(), "circuit_json": cirq.to_json(cirq.Circuit(cirq.X(cirq.LineQubit(0))))}
    )
    with pytest.raises(ValueError, match="circuit"):
        sign_manifest(casse, hmac_signer)


# ---------- hygiene des cles ----------


def test_une_cle_trop_courte_est_refusee():
    with pytest.raises(ValueError, match="trop courte"):
        HmacSigner(os.urandom(16), key_id="alice")


def test_une_cle_en_texte_est_refusee():
    with pytest.raises(TypeError, match="octets"):
        HmacSigner("mon-mot-de-passe", key_id="alice")


def test_un_key_id_vide_est_refuse(cle):
    with pytest.raises(ValueError, match="key_id"):
        HmacSigner(cle, key_id="")


def test_la_cle_n_apparait_jamais_dans_la_representation(cle):
    signer = HmacSigner(cle, key_id="alice")
    assert cle.hex() not in repr(signer)
    assert "alice" in repr(signer)


def test_la_cle_n_apparait_jamais_dans_la_signature(cle):
    m = _manifest()
    sig = sign_manifest(m, HmacSigner(cle, key_id="alice"))
    serialise = json.dumps(sig.to_dict())
    assert cle.hex() not in serialise


# ---------- ed25519 : la seule des deux qui soit opposable ----------


def test_ed25519_signe_et_verifie():
    signer, _priv, pub = Ed25519Signer.generate(key_id="simon")
    m = _manifest()
    sig = sign_manifest(m, signer)
    r = verify_manifest_signature(m, sig, Ed25519Verifier(pub, key_id="simon"))
    assert r.valid is True
    assert r.third_party_verifiable is True
    assert "opposable" in r.detail


def test_ed25519_refuse_une_autre_cle_publique():
    signer, _priv, _pub = Ed25519Signer.generate(key_id="simon")
    _autre, _p2, pub_etranger = Ed25519Signer.generate(key_id="etranger")
    m = _manifest()
    sig = sign_manifest(m, signer)
    r = verify_manifest_signature(m, sig, Ed25519Verifier(pub_etranger, "etranger"))
    assert r.valid is False and r.signature_valid is False


def test_le_verifieur_ed25519_ne_peut_pas_signer():
    """C'est toute la difference avec HMAC, et elle doit etre structurelle."""
    _signer, _priv, pub = Ed25519Signer.generate(key_id="simon")
    assert not hasattr(Ed25519Verifier(pub, "simon"), "sign")


def test_seul_ed25519_est_declare_opposable():
    assert SignatureAlgorithm.ED25519.is_third_party_verifiable is True
    assert SignatureAlgorithm.HMAC_SHA256.is_third_party_verifiable is False


# ---------- le message signe ----------


def test_le_message_signe_lie_algorithme_cle_et_hash():
    base = signing_payload("aa" * 32, "hmac-sha256", "alice")
    assert base != signing_payload("bb" * 32, "hmac-sha256", "alice")
    assert base != signing_payload("aa" * 32, "ed25519", "alice")
    assert base != signing_payload("aa" * 32, "hmac-sha256", "bob")


def test_le_schema_de_signature_est_verifie(hmac_signer):
    sig = sign_manifest(_manifest(), hmac_signer)
    with pytest.raises(ValueError, match="schema"):
        Signature.from_dict({**sig.to_dict(), "schema_version": "99.0"})


def test_un_champ_inconnu_dans_la_signature_est_refuse(hmac_signer):
    sig = sign_manifest(_manifest(), hmac_signer)
    with pytest.raises(ValueError, match="inconnus"):
        Signature.from_dict({**sig.to_dict(), "invente": 1})


def test_la_version_de_schema_est_celle_attendue(hmac_signer):
    assert sign_manifest(_manifest(), hmac_signer).schema_version == (
        SIGNATURE_SCHEMA_VERSION
    )
