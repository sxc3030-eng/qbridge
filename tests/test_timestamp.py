"""Le temoin exterieur : horodatage RFC 3161 de la tete du journal.

TESTS HERMETIQUES. Les jetons sont fabriques ici, octet par octet. Un test qui
appellerait une autorite echouerait en avion, en CI sans reseau, et le jour ou
le domaine du DFN changera. Le seul test reseau est explicitement marque et
saute par defaut.
"""

from __future__ import annotations

import hashlib
import os

import pytest

from qbridge.timestamp import (
    OID_SHA256,
    OID_TSTINFO,
    Timestamp,
    _enfants,
    _lire,
    _tlv,
    construire_requete,
)

TETE = "e6f7dc37c33f32370bad03c814a14eea89f9ae32474a0d8002654336da9b3cde"


def _jeton(
    empreinte: bytes,
    *,
    statut: int = 0,
    genTime: bytes = b"20260902203344Z",
    avec_tstinfo: bool = True,
) -> bytes:
    """Fabrique un TimeStampResp de structure valide.

    Ce n'est PAS un jeton authentique : rien n'est signe. C'est exactement ce
    que le module dit ne pas verifier, et le fait que ces tests passent avec un
    faux le PROUVE plutot que de le cacher.
    """
    algo = _tlv(0x30, _tlv(0x06, OID_SHA256) + _tlv(0x05, b""))
    imprint = _tlv(0x30, algo + _tlv(0x04, empreinte))
    tstinfo = _tlv(
        0x30,
        _tlv(0x02, b"\x01")  # version
        + _tlv(0x06, bytes([0x2A, 0x03]))  # policy, quelconque
        + imprint
        + _tlv(0x02, b"\x01\x02\x03")  # serialNumber
        + _tlv(0x18, genTime),  # GeneralizedTime
    )
    encap = _tlv(
        0x30,
        _tlv(0x06, OID_TSTINFO) + _tlv(0xA0, _tlv(0x04, tstinfo)),
    )
    if not avec_tstinfo:
        encap = _tlv(0x30, _tlv(0x06, bytes([0x2A, 0x04])) + _tlv(0x04, b"rien"))

    signed = _tlv(0x30, _tlv(0x02, b"\x03") + _tlv(0x31, b"") + encap)
    contenu = _tlv(
        0x30,
        _tlv(0x06, bytes([0x2A, 0x86, 0x48, 0x86, 0xF7, 0x0D, 0x01, 0x07, 0x02]))
        + _tlv(0xA0, signed),
    )
    return _tlv(0x30, _tlv(0x30, _tlv(0x02, bytes([statut]))) + contenu)


def _empreinte(tete: str) -> bytes:
    return hashlib.sha256(bytes.fromhex(tete)).digest()


# ---------- la requete ----------


def test_la_requete_porte_l_empreinte_demandee():
    empreinte = _empreinte(TETE)
    requete = construire_requete(empreinte, nonce=0x1122334455667788)
    assert empreinte in requete
    _, corps, _ = _lire(requete)
    enfants = _enfants(corps)
    assert enfants[0][1] == b"\x01", "version 1"
    assert enfants[-1][1] == b"\xff", "certReq TRUE : on veut le certificat"


def test_le_nonce_change_a_chaque_requete():
    """Sans nonce, un intermediaire pourrait resservir un vieux jeton."""
    empreinte = _empreinte(TETE)
    assert construire_requete(empreinte) != construire_requete(empreinte)


def test_une_empreinte_de_mauvaise_taille_est_refusee():
    with pytest.raises(ValueError, match="32"):
        construire_requete(b"trop court")


# ---------- la liaison, qui est le controle utile ----------


def test_un_jeton_qui_date_CETTE_tete_est_lie():
    rapport = Timestamp(_jeton(_empreinte(TETE))).verify(TETE)
    assert rapport.bound and rapport.granted
    assert rapport.stamped_at == "2026-09-02T20:33:44+00:00"


def test_un_jeton_qui_date_UNE_AUTRE_empreinte_est_refuse():
    """LE controle utile. Un jeton authentique d'un AUTRE document ne prouve
    rien sur celui-ci — meme attaque de substitution que pour les signatures."""
    rapport = Timestamp(_jeton(_empreinte("a" * 64))).verify(TETE)
    assert not rapport.bound
    assert rapport.granted, "le jeton est valide, il date juste autre chose"
    assert "AUTRE empreinte" in rapport.detail


def test_un_horodatage_REFUSE_par_l_autorite_ne_lie_rien():
    rapport = Timestamp(_jeton(_empreinte(TETE), statut=2)).verify(TETE)
    assert not rapport.bound and not rapport.granted
    assert "refuse" in rapport.detail


def test_un_jeton_illisible_ne_leve_pas():
    """Un jeton corrompu doit rendre un verdict, pas une trace de pile : le
    verdict d'integrite reste a produire."""
    rapport = Timestamp(b"\x30\x82\xff\xff pas du DER").verify(TETE)
    assert not rapport.bound
    assert "illisible" in rapport.detail


def test_un_jeton_tronque_ne_leve_pas():
    complet = _jeton(_empreinte(TETE))
    rapport = Timestamp(complet[: len(complet) // 2]).verify(TETE)
    assert not rapport.bound
    assert "illisible" in rapport.detail


def test_un_jeton_sans_tstinfo_est_refuse():
    rapport = Timestamp(_jeton(_empreinte(TETE), avec_tstinfo=False)).verify(TETE)
    assert not rapport.bound


def test_un_jeton_vide_est_refuse():
    with pytest.raises(ValueError, match="vide"):
        Timestamp(b"")


# ---------- ce que le module NE verifie PAS ----------


def test_un_jeton_FABRIQUE_passe_la_liaison():
    """LA limite, verrouillee par un test plutot que promise dans un texte.

    Les jetons de ce fichier ne sont signes par personne. Ils passent quand
    meme, parce que le module verifie la COHERENCE et non l'AUTHENTICITE. La
    signature de l'autorite demande une racine de confiance ; sans elle, un
    faussaire capable de fabriquer un jeton n'est pas arrete ici.
    """
    faux = Timestamp(_jeton(_empreinte(TETE)))
    rapport = faux.verify(TETE)
    assert rapport.bound, "un faux passe : c'est la limite, pas un defaut cache"
    assert rapport.signature_verified is False
    assert "NON verifiee" in rapport.detail


def test_la_commande_openssl_porte_la_bonne_empreinte():
    """Ce que le module ne fait pas, il doit dire comment le faire."""
    commande = Timestamp(_jeton(_empreinte(TETE))).commande_openssl("j.tsr", TETE)
    assert hashlib.sha256(bytes.fromhex(TETE)).hexdigest() in commande
    assert "ts -verify" in commande
    assert "-CAfile" in commande


# ---------- persistance ----------


def test_un_aller_retour_preserve_le_jeton(tmp_path):
    original = _jeton(_empreinte(TETE))
    Timestamp(original).save(tmp_path)
    relu = Timestamp.load(tmp_path)
    assert relu.token == original
    assert relu.verify(TETE).bound


def test_le_jeton_se_verifie_HORS_LIGNE(tmp_path, monkeypatch):
    """Une preuve qui exigerait de joindre un serveur dans dix ans n'en serait
    pas une : les autorites disparaissent, les domaines expirent."""
    Timestamp(_jeton(_empreinte(TETE))).save(tmp_path)

    def interdit(*a, **k):
        raise AssertionError("la verification ne doit JAMAIS toucher au reseau")

    import requests

    monkeypatch.setattr(requests, "post", interdit)
    monkeypatch.setattr(requests, "get", interdit)

    assert Timestamp.load(tmp_path).verify(TETE).bound


# ---------- reseau, hors CI ----------


@pytest.mark.skipif(
    os.environ.get("QBRIDGE_TSA_RESEAU") != "1",
    reason="appelle une autorite reelle ; QBRIDGE_TSA_RESEAU=1 pour l'activer",
)
def test_une_vraie_autorite_repond():
    from qbridge.timestamp import stamp

    jeton = stamp(TETE)
    rapport = jeton.verify(TETE)
    assert rapport.bound and rapport.granted and rapport.stamped_at
