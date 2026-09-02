"""Temoin exterieur : dater la tete du journal chez un tiers (RFC 3161).

CE QUE LA CHAINE NE POUVAIT PAS FAIRE SEULE. Le journal rend une suppression
couteuse : effacer une entree oblige a reecrire tout le journal, et sa tete
change. Mais si personne n'a jamais vu l'ancienne tete, la reecriture reste
indetectable. Il manquait un temoin exterieur.

CE QUE FAIT UN HORODATAGE RFC 3161. Une autorite independante signe
« l'empreinte X m'a ete presentee a l'instant T ». Personne ne peut antidater
un jeton sans la cle de l'autorite. Une tete horodatee ne peut donc plus etre
remplacee apres coup : la reecriture produirait une tete differente, et
l'ancienne resterait attestee.

CE QUI EST ENVOYE. Une empreinte de 32 octets, rien d'autre. Ni circuit, ni
tirages, ni manifeste. L'autorite ne peut pas savoir ce qu'elle date — c'est
tout l'interet du protocole.

RESEAU POUR HORODATER, JAMAIS POUR VERIFIER. `stamp()` appelle une autorite ;
`Timestamp.verify()` ne parle a personne. Une preuve qui exigerait de joindre
un serveur dans dix ans n'en serait pas une : les autorites disparaissent, les
domaines expirent. Le jeton est autoportant, on le garde a cote de l'archive.

CE QUI EST VERIFIE ICI, ET CE QUI NE L'EST PAS. Le module verifie en Python
pur, hors ligne : le statut accorde par l'autorite, la LIAISON — l'empreinte
datee est bien celle de cette tete-ci — et la date declaree. Il ne verifie PAS
la signature cryptographique de l'autorite, qui demande une racine de confiance
et un chemin de certification. `commande_openssl()` rend la commande exacte
pour le faire. Sans cette verification, un jeton FABRIQUE passerait : ce qui
est atteste ici, c'est la coherence, pas l'authenticite. Le dire autrement
serait le meme mensonge que les autres.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TIMESTAMP_FILENAME = "journal.tsr"

TSA_PAR_DEFAUT = "https://zeitstempel.dfn.de"
"""Autorite du DFN, le reseau academique allemand. Gratuite, sans compte.

Verifiee le 2026-09-02 : jeton de 6 670 octets, statut « Granted », liaison
exacte. `freetsa.org` a expire au bout de 20 s le meme jour.
"""

OID_SHA256 = bytes([0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01])
OID_TSTINFO = bytes(
    [0x2A, 0x86, 0x48, 0x86, 0xF7, 0x0D, 0x01, 0x09, 0x10, 0x01, 0x04]
)
"""id-ct-TSTInfo, 1.2.840.113549.1.9.16.1.4 : marque le contenu date."""


# --------------------------------------------------------------------------
# DER, ecrit a la main
# --------------------------------------------------------------------------
#
# POURQUOI PAS UNE BIBLIOTHEQUE. Les structures en jeu tiennent en quelques
# lignes et ce code decide si une preuve tient. Chaque octet est ici lisible et
# verifiable ; une dependance de plus le serait moins.


def _longueur(n: int) -> bytes:
    if n < 0x80:
        return bytes([n])
    corps = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(corps)]) + corps


def _tlv(tag: int, contenu: bytes) -> bytes:
    return bytes([tag]) + _longueur(len(contenu)) + contenu


def _lire(donnees: bytes, position: int = 0) -> Tuple[int, bytes, int]:
    """Lit un TLV. Rend (tag, contenu, position suivante)."""
    if position + 2 > len(donnees):
        raise ValueError("DER tronque : plus assez d'octets pour un en-tete")
    tag = donnees[position]
    premier = donnees[position + 1]
    position += 2
    if premier < 0x80:
        longueur = premier
    else:
        octets = premier & 0x7F
        if octets == 0 or position + octets > len(donnees):
            raise ValueError("DER invalide : longueur indefinie ou tronquee")
        longueur = int.from_bytes(donnees[position : position + octets], "big")
        position += octets
    fin = position + longueur
    if fin > len(donnees):
        raise ValueError("DER tronque : contenu plus court qu'annonce")
    return tag, donnees[position:fin], fin


def _enfants(sequence: bytes) -> List[Tuple[int, bytes]]:
    resultat: List[Tuple[int, bytes]] = []
    position = 0
    while position < len(sequence):
        tag, contenu, position = _lire(sequence, position)
        resultat.append((tag, contenu))
    return resultat


def _trouver_tstinfo(jeton: bytes) -> bytes:
    """Rend le CONTENU du TSTInfo, prêt pour `_enfants`.

    Chercher l'OID par balayage d'octets serait plus court et faux : un OID peut
    apparaitre ailleurs, dans un certificat par exemple. On suit donc la
    structure.

    Attention au niveau d'imbrication : l'eContent est un OCTET STRING dont le
    contenu est un SEQUENCE DER complet. Il faut le deballer une fois de plus,
    sans quoi on croit lire les champs du TSTInfo et on lit le TSTInfo entier
    comme unique enfant.
    """
    _, contenu_info, _ = _lire(jeton)  # ContentInfo
    enfants = _enfants(contenu_info)
    if len(enfants) < 2:
        raise ValueError("jeton invalide : ContentInfo sans contenu")
    _, explicite = enfants[1]  # [0] EXPLICIT SignedData
    _, signed_data, _ = _lire(explicite)

    for tag, contenu in _enfants(signed_data):
        if tag != 0x30:
            continue
        interne = _enfants(contenu)
        if not interne or interne[0][0] != 0x06:
            continue
        if interne[0][1] != OID_TSTINFO:
            continue
        if len(interne) < 2:
            raise ValueError("encapContentInfo sans eContent")
        _, octets, _ = _lire(interne[1][1])  # [0] EXPLICIT -> OCTET STRING
        _, tstinfo, _ = _lire(octets)  # OCTET STRING -> SEQUENCE TSTInfo
        return tstinfo
    raise ValueError(
        "aucun TSTInfo dans ce jeton : ce n'est pas une reponse d'horodatage"
    )


def _date_generalisee(brut: bytes) -> str:
    """GeneralizedTime -> ISO 8601 UTC."""
    texte = brut.decode("ascii").rstrip("Z")
    base, _, fraction = texte.partition(".")
    instant = _dt.datetime.strptime(base, "%Y%m%d%H%M%S").replace(
        tzinfo=_dt.timezone.utc
    )
    if fraction:
        instant = instant.replace(microsecond=int(fraction.ljust(6, "0")[:6]))
    return instant.isoformat()


# --------------------------------------------------------------------------
# requete et jeton
# --------------------------------------------------------------------------


def construire_requete(empreinte: bytes, nonce: Optional[int] = None) -> bytes:
    """TimeStampReq (RFC 3161, section 2.4.1).

    Le nonce interdit a l'autorite de rejouer une reponse deja produite : sans
    lui, un intermediaire pourrait resservir un vieux jeton.
    """
    if len(empreinte) != 32:
        raise ValueError(
            f"empreinte de {len(empreinte)} octets : sha256 en attend 32."
        )
    if nonce is None:
        nonce = int.from_bytes(os.urandom(8), "big")

    algorithme = _tlv(0x30, _tlv(0x06, OID_SHA256) + _tlv(0x05, b""))
    imprint = _tlv(0x30, algorithme + _tlv(0x04, empreinte))
    corps_nonce = nonce.to_bytes((nonce.bit_length() + 8) // 8 or 1, "big")
    return _tlv(
        0x30,
        _tlv(0x02, b"\x01")  # version 1
        + imprint
        + _tlv(0x02, corps_nonce)
        + _tlv(0x01, b"\xff"),  # certReq : renvoyer le certificat signataire
    )


@dataclass(frozen=True)
class TimestampReport:
    """Verdict sur un jeton. `bound` est le champ a lire pour decider."""

    bound: bool
    granted: bool
    stamped_at: Optional[str]
    imprint: Optional[str]
    expected_imprint: Optional[str]
    detail: str
    signature_verified: bool = False
    """TOUJOURS False ici. La signature de l'autorite demande une racine de
    confiance ; `commande_openssl()` rend la commande qui la verifie."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bound": self.bound,
            "granted": self.granted,
            "stamped_at": self.stamped_at,
            "imprint": self.imprint,
            "expected_imprint": self.expected_imprint,
            "signature_verified": self.signature_verified,
            "detail": self.detail,
        }


class Timestamp:
    """Jeton RFC 3161, lu hors ligne."""

    def __init__(self, token: bytes) -> None:
        if not token:
            raise ValueError("jeton vide")
        self._token = bytes(token)

    @property
    def token(self) -> bytes:
        return self._token

    # ---------- lecture ----------

    def _reponse(self) -> Tuple[int, Optional[bytes]]:
        """(statut PKIStatus, jeton) depuis un TimeStampResp."""
        _, corps, _ = _lire(self._token)
        enfants = _enfants(corps)
        if not enfants:
            raise ValueError("reponse d'horodatage vide")
        statut_seq = _enfants(enfants[0][1])
        if not statut_seq or statut_seq[0][0] != 0x02:
            raise ValueError("PKIStatusInfo illisible")
        statut = int.from_bytes(statut_seq[0][1], "big")
        jeton = None
        if len(enfants) > 1:
            jeton = _tlv(enfants[1][0], enfants[1][1])
        return statut, jeton

    def details(self) -> Dict[str, Any]:
        """Statut, empreinte datee et date declaree. Ne parle a personne."""
        statut, jeton = self._reponse()
        if jeton is None:
            return {"granted": statut == 0, "status": statut}

        tstinfo = _enfants(_trouver_tstinfo(jeton))
        if len(tstinfo) < 5:
            raise ValueError(
                f"TSTInfo incomplet : {len(tstinfo)} champs, 5 attendus au moins"
            )
        # TSTInfo : version, policy, messageImprint, serialNumber, genTime...
        imprint = _enfants(tstinfo[2][1])
        empreinte = imprint[1][1]
        return {
            "granted": statut == 0,
            "status": statut,
            "imprint": empreinte.hex(),
            "stamped_at": _date_generalisee(tstinfo[4][1]),
        }

    # ---------- verification ----------

    def verify(self, head: str) -> TimestampReport:
        """Le jeton date-t-il BIEN cette tete de journal ?

        Hors ligne, integralement. Ce qui est verifie : le statut, la liaison a
        cette tete, la date. Ce qui ne l'est PAS : la signature de l'autorite.
        """
        attendue = hashlib.sha256(bytes.fromhex(head)).hexdigest()
        try:
            infos = self.details()
        except (ValueError, IndexError, UnicodeDecodeError) as exc:
            return TimestampReport(
                bound=False,
                granted=False,
                stamped_at=None,
                imprint=None,
                expected_imprint=attendue,
                detail=f"jeton illisible : {type(exc).__name__} : {exc}",
            )

        if not infos.get("granted"):
            return TimestampReport(
                bound=False,
                granted=False,
                stamped_at=None,
                imprint=None,
                expected_imprint=attendue,
                detail=(
                    f"l'autorite a refuse l'horodatage (statut "
                    f"{infos.get('status')}) : il n'y a rien a opposer"
                ),
            )

        obtenue = infos.get("imprint")
        if obtenue != attendue:
            return TimestampReport(
                bound=False,
                granted=True,
                stamped_at=infos.get("stamped_at"),
                imprint=obtenue,
                expected_imprint=attendue,
                detail=(
                    "ce jeton date une AUTRE empreinte que cette tete de "
                    "journal : il n'atteste rien sur cette serie"
                ),
            )

        return TimestampReport(
            bound=True,
            granted=True,
            stamped_at=infos.get("stamped_at"),
            imprint=obtenue,
            expected_imprint=attendue,
            detail=(
                f"tete attestee comme existante au {infos.get('stamped_at')} ; "
                "signature de l'autorite NON verifiee ici"
            ),
        )

    def commande_openssl(self, chemin_jeton: str | Path, head: str) -> str:
        """Commande qui verifie la SIGNATURE, ce que ce module ne fait pas.

        Demande le certificat racine de l'autorite. Pour le DFN il se recupere
        sur pki.pca.dfn.de ; toute autre autorite publie le sien.
        """
        empreinte = hashlib.sha256(bytes.fromhex(head)).hexdigest()
        return (
            f"openssl ts -verify -in {chemin_jeton} "
            f"-digest {empreinte} -CAfile racine-tsa.pem"
        )

    # ---------- persistance ----------

    def save(self, path: str | Path) -> None:
        chemin = Path(path)
        if chemin.is_dir():
            chemin = chemin / TIMESTAMP_FILENAME
        chemin.parent.mkdir(parents=True, exist_ok=True)
        chemin.write_bytes(self._token)

    @classmethod
    def load(cls, path: str | Path) -> "Timestamp":
        chemin = Path(path)
        if chemin.is_dir():
            chemin = chemin / TIMESTAMP_FILENAME
        return cls(chemin.read_bytes())


# --------------------------------------------------------------------------
# le seul appel reseau du module
# --------------------------------------------------------------------------


def stamp(head: str, *, url: str = TSA_PAR_DEFAUT, timeout: float = 30.0) -> Timestamp:
    """Fait dater `head` par une autorite. SEULE fonction qui sort de la machine.

    N'envoie qu'une empreinte de 32 octets : l'autorite ne peut pas savoir ce
    qu'elle date. Rien de l'experience ne quitte le poste.
    """
    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "`requests` est absent : impossible de joindre une autorite."
        ) from exc

    try:
        empreinte = hashlib.sha256(bytes.fromhex(head)).digest()
    except ValueError as exc:
        raise ValueError(
            f"tete de journal invalide : {head!r} n'est pas de l'hexadecimal."
        ) from exc

    reponse = requests.post(
        url,
        data=construire_requete(empreinte),
        headers={"Content-Type": "application/timestamp-query"},
        timeout=timeout,
    )
    if reponse.status_code != 200:
        raise RuntimeError(
            f"l'autorite {url} a repondu HTTP {reponse.status_code} : "
            "aucun jeton obtenu."
        )
    jeton = Timestamp(reponse.content)

    # On verifie la LIAISON avant de rendre le jeton : un jeton qui date une
    # autre empreinte n'a aucune valeur, et le decouvrir plus tard serait pire.
    rapport = jeton.verify(head)
    if not rapport.bound:
        raise RuntimeError(
            f"l'autorite a rendu un jeton qui n'atteste pas cette tete : "
            f"{rapport.detail}"
        )
    return jeton
