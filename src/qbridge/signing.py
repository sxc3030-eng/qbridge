"""Signature detachee d'un manifeste.

POURQUOI CE MODULE EXISTE. `content_hash` prouve qu'un document est coherent
avec lui-meme, rien de plus : il est public et deterministe, donc quiconque
modifie le manifeste le recalcule en deux lignes. C'est un controle contre la
corruption, pas contre un adversaire. Signer ajoute la seule chose qui manque —
l'identite de celui qui a scelle.

DEUX ALGORITHMES, DEUX GARANTIES QU'IL NE FAUT PAS CONFONDRE.

- `hmac-sha256` est SYMETRIQUE. Quiconque peut verifier peut aussi forger,
  puisque c'est la meme cle. Cela prouve « ce document n'a pas bouge depuis que
  MON organisation l'a scelle » — utile en integration continue ou pour ses
  propres archives, sans aucune valeur comme preuve opposable a un tiers.
  Bibliotheque standard uniquement : marchera encore dans dix ans.

- `ed25519` est ASYMETRIQUE. Seule la cle privee signe, n'importe qui verifie
  avec la publique. C'est le seul des deux qui rende une archive opposable.
  Demande le paquet `cryptography` (extra `sign`), donc une dependance
  compilee : le noyau de qbridge reste sans dependance, la signature
  asymetrique est optionnelle et son absence est signalee clairement.

CE QUI EST SIGNE. Pas le `content_hash` nu, mais un lien canonique
`{schema, algorithme, key_id, content_hash}`. Signer le hash seul laisserait
deux failles : une signature produite en HMAC pourrait etre presentee comme une
ed25519 (substitution d'algorithme), et une signature de la cle A pourrait etre
revendiquee pour la cle B (confusion de cles). Lier ces champs DANS le message
signe ferme les deux.

CE QUE CE MODULE NE FAIT PAS. Il ne genere, ne stocke et ne distribue aucune
cle en dehors de ce que l'appelant lui passe explicitement. La gestion des cles
— ou vit la cle privee, qui y accede, comment on revoque — est hors perimetre
et reste entierement a la charge de l'utilisateur. Aucune cle n'est jamais
ecrite dans un manifeste, une signature ou un journal.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import json
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Protocol, Tuple, runtime_checkable

from qbridge.digest import canonical_json

SIGNATURE_SCHEMA_VERSION = "2.0"

SIGNATURE_FILENAME = "signature.json"
"""Nom du fichier detache, a cote de `manifest.json` dans un dossier d'archive."""


class SignatureAlgorithm(str, Enum):
    HMAC_SHA256 = "hmac-sha256"
    """Symetrique. Integrite verifiable par le porteur de la cle. Pas de preuve
    d'origine opposable : verifier et forger demandent la meme cle."""

    ED25519 = "ed25519"
    """Asymetrique. Origine verifiable par quiconque detient la cle publique."""

    @property
    def is_third_party_verifiable(self) -> bool:
        """Vrai si un tiers peut verifier sans pouvoir forger."""
        return self is SignatureAlgorithm.ED25519


# --------------------------------------------------------------------------
# Le message signe
# --------------------------------------------------------------------------


class SignatureScope(str, Enum):
    """Ce sur quoi porte une signature.

    La distinction est necessaire, pas cosmetique : une signature de RECETTE ne
    dit rien des resultats. Confondre les deux permettait de remplacer les
    tirages sous une signature restee valide.
    """

    MANIFEST = "manifest"
    """La recette seule. Ne couvre AUCUN resultat."""

    RECORD = "record"
    """L'archive entiere : recette et tirages."""


def signing_payload(
    content_hash: str,
    algorithm: str,
    key_id: str,
    scope: str = SignatureScope.MANIFEST.value,
) -> bytes:
    """Construit le message effectivement signe.

    Lie l'algorithme, l'identifiant de cle ET la portee au hash. Sans le lien
    de portee, une signature de recette pourrait etre presentee comme couvrant
    une archive complete.
    """
    return canonical_json(
        {
            "qbridge_signature_schema": SIGNATURE_SCHEMA_VERSION,
            "algorithm": algorithm,
            "key_id": key_id,
            "scope": scope,
            "content_hash": content_hash,
        }
    ).encode("utf-8")


# --------------------------------------------------------------------------
# Protocoles
# --------------------------------------------------------------------------


@runtime_checkable
class Signer(Protocol):
    algorithm: str
    key_id: str

    def sign(self, payload: bytes) -> bytes: ...


@runtime_checkable
class Verifier(Protocol):
    algorithm: str
    key_id: str

    def verify(self, payload: bytes, signature: bytes) -> bool: ...


# --------------------------------------------------------------------------
# HMAC-SHA256 — bibliotheque standard
# --------------------------------------------------------------------------


class HmacSigner:
    """Signe ET verifie avec une cle secrete partagee.

    La meme classe joue les deux roles : c'est la nature du symetrique, et le
    dire dans le code evite de laisser croire a une propriete d'origine qui
    n'existe pas ici.
    """

    algorithm = SignatureAlgorithm.HMAC_SHA256.value

    def __init__(self, key: bytes, key_id: str) -> None:
        if not isinstance(key, (bytes, bytearray)):
            raise TypeError("La cle HMAC doit etre des octets, pas du texte.")
        if len(key) < 32:
            raise ValueError(
                f"Cle HMAC trop courte ({len(key)} octets). Au moins 32 octets "
                "sont attendus pour SHA-256 ; utiliser os.urandom(32)."
            )
        if not key_id:
            raise ValueError("Un key_id non vide est obligatoire.")
        self._key = bytes(key)
        self.key_id = key_id

    def sign(self, payload: bytes) -> bytes:
        return hmac.new(self._key, payload, hashlib.sha256).digest()

    def verify(self, payload: bytes, signature: bytes) -> bool:
        # compare_digest : comparaison a temps constant. Un `==` fuirait par le
        # temps d'execution le nombre d'octets corrects en tete.
        return hmac.compare_digest(self.sign(payload), signature)

    def __repr__(self) -> str:  # pragma: no cover - confort de debogage
        return f"HmacSigner(key_id={self.key_id!r})"  # jamais la cle


# --------------------------------------------------------------------------
# Ed25519 — optionnel
# --------------------------------------------------------------------------


def _require_cryptography() -> Any:
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as exc:  # pragma: no cover - depend de l'installation
        raise RuntimeError(
            "La signature ed25519 demande le paquet `cryptography`, absent de "
            "cet environnement. Installer `pip install qbridge[sign]`, ou "
            "utiliser HmacSigner qui ne depend que de la bibliotheque standard "
            "— en gardant en tete qu'il ne fournit pas de preuve d'origine "
            "opposable a un tiers."
        ) from exc
    return ed25519


class Ed25519Signer:
    """Signe avec une cle privee. Ne verifie pas : c'est le role du verifieur."""

    algorithm = SignatureAlgorithm.ED25519.value

    def __init__(self, private_key_bytes: bytes, key_id: str) -> None:
        ed25519 = _require_cryptography()
        if not key_id:
            raise ValueError("Un key_id non vide est obligatoire.")
        self._key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)
        self.key_id = key_id

    @classmethod
    def generate(cls, key_id: str) -> Tuple["Ed25519Signer", bytes, bytes]:
        """Fabrique une paire de cles. Rend (signeur, cle_privee, cle_publique).

        Les octets sont rendus a l'appelant pour qu'il decide ou les ecrire :
        ce module ne touche jamais au disque pour une cle.
        """
        ed25519 = _require_cryptography()
        from cryptography.hazmat.primitives import serialization

        private = ed25519.Ed25519PrivateKey.generate()
        priv = private.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        pub = private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return cls(priv, key_id), priv, pub

    def sign(self, payload: bytes) -> bytes:
        return self._key.sign(payload)

    def __repr__(self) -> str:  # pragma: no cover - confort de debogage
        return f"Ed25519Signer(key_id={self.key_id!r})"  # jamais la cle


class Ed25519Verifier:
    """Verifie avec une cle publique. Ne peut pas signer — c'est le point."""

    algorithm = SignatureAlgorithm.ED25519.value

    def __init__(self, public_key_bytes: bytes, key_id: str) -> None:
        ed25519 = _require_cryptography()
        if not key_id:
            raise ValueError("Un key_id non vide est obligatoire.")
        self._key = ed25519.Ed25519PublicKey.from_public_bytes(public_key_bytes)
        self.key_id = key_id

    def verify(self, payload: bytes, signature: bytes) -> bool:
        from cryptography.exceptions import InvalidSignature

        try:
            self._key.verify(signature, payload)
        except InvalidSignature:
            return False
        return True


# --------------------------------------------------------------------------
# La signature detachee
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Signature:
    """Signature detachee d'un manifeste.

    Detachee, jamais rangee DANS le manifeste : `content_hash` couvre tous les
    champs, donc y ajouter la signature changerait le hash qu'elle signe.
    """

    schema_version: str
    algorithm: str
    key_id: str
    scope: str
    content_hash: str
    signature: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Signature":
        version = data.get("schema_version")
        if version != SIGNATURE_SCHEMA_VERSION:
            raise ValueError(
                f"Version de schema de signature incompatible : {version!r} "
                f"(attendu {SIGNATURE_SCHEMA_VERSION!r})."
            )
        connus = {f.name for f in fields(cls)}
        inconnus = set(data) - connus
        if inconnus:
            raise ValueError(f"Champs inconnus dans la signature : {sorted(inconnus)}")
        manquants = connus - set(data)
        if manquants:
            raise ValueError(f"Champs absents de la signature : {sorted(manquants)}")
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "Signature":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _sign(hash_scelle: str, scope: str, signer: Signer) -> Signature:
    payload = signing_payload(hash_scelle, signer.algorithm, signer.key_id, scope)
    return Signature(
        schema_version=SIGNATURE_SCHEMA_VERSION,
        algorithm=signer.algorithm,
        key_id=signer.key_id,
        scope=scope,
        content_hash=hash_scelle,
        signature=signer.sign(payload).hex(),
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
    )


def sign_manifest(manifest: Any, signer: Signer) -> Signature:
    """Signe une RECETTE seule. Ne couvre aucun resultat.

    Le manifeste est verifie AVANT signature : signer un document incoherent
    reviendrait a attester une incoherence.

    Pour une archive contenant des tirages, utiliser `sign_record` — cette
    fonction-ci laisserait les bitstrings hors de la signature.
    """
    manifest.verify_self()
    return _sign(manifest.content_hash, SignatureScope.MANIFEST.value, signer)


def sign_record(record: Any, signer: Signer) -> Signature:
    """Signe une ARCHIVE entiere : recette ET tirages.

    C'est ce qu'il faut pour une archive. `sign_manifest` ne couvrirait que la
    recette, laissant les bitstrings — la seule donnee non regenerable —
    remplacables sous une signature restee valide.
    """
    record.verify_integrity()
    return _sign(record.content_hash(), SignatureScope.RECORD.value, signer)


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SignatureReport:
    """Verdict de verification, decompose par cause.

    Meme discipline d'attribution que `verify_archival` : « document altere »,
    « signature qui vise un autre document » et « signature invalide » sont
    trois situations differentes, et les confondre envoie sur une fausse piste.
    """

    valid: bool
    manifest_intact: bool
    binds_this_manifest: bool
    signature_valid: bool
    algorithm: str
    key_id: str
    third_party_verifiable: bool
    detail: str


def verify_record_signature(
    record: Any, signature: Signature, verifier: Verifier
) -> SignatureReport:
    """Verifie une signature d'ARCHIVE : recette et tirages."""
    return _verify(
        record.content_hash(),
        SignatureScope.RECORD.value,
        lambda: record.verify_integrity(),
        signature,
        verifier,
    )


def verify_manifest_signature(
    manifest: Any, signature: Signature, verifier: Verifier
) -> SignatureReport:
    """Verifie une signature detachee contre un manifeste.

    Verifie une signature de RECETTE. Ne dit rien des resultats.
    """
    return _verify(
        manifest.content_hash,
        SignatureScope.MANIFEST.value,
        lambda: manifest.verify_self(),
        signature,
        verifier,
    )


def _verify(
    hash_attendu: str,
    scope_attendu: str,
    controle_integrite: Any,
    signature: Signature,
    verifier: Verifier,
) -> SignatureReport:
    """Six controles independants, tous necessaires :

    1. Le document est coherent avec lui-meme.
    2. L'algorithme declare est bien celui du verifieur.
    3. Le key_id declare est bien celui attendu par le verifieur.
    4. La portee declaree est celle qu'on verifie.
    5. La signature vise CE document (meme hash).
    6. La signature est cryptographiquement valide.
    """
    motifs = []

    manifeste_ok = True
    try:
        controle_integrite()
    except ValueError as exc:
        manifeste_ok = False
        motifs.append(f"document altere : {exc}")

    if signature.algorithm != verifier.algorithm:
        return SignatureReport(
            valid=False,
            manifest_intact=manifeste_ok,
            binds_this_manifest=False,
            signature_valid=False,
            algorithm=signature.algorithm,
            key_id=signature.key_id,
            third_party_verifiable=False,
            detail=(
                f"algorithme incompatible : la signature declare "
                f"{signature.algorithm!r}, le verifieur est {verifier.algorithm!r}"
            ),
        )

    # Le key_id du verifieur n'etait compare a RIEN : le message signe liait
    # l'identite fournie par l'attaquant a elle-meme, donc `--key-id` pouvait
    # valoir n'importe quoi et la signature restait « valide ».
    if signature.key_id != verifier.key_id:
        return SignatureReport(
            valid=False,
            manifest_intact=manifeste_ok,
            binds_this_manifest=False,
            signature_valid=False,
            algorithm=signature.algorithm,
            key_id=signature.key_id,
            third_party_verifiable=False,
            detail=(
                f"identite de cle incompatible : la signature declare "
                f"{signature.key_id!r}, le verifieur attend {verifier.key_id!r}"
            ),
        )

    if getattr(signature, "scope", None) != scope_attendu:
        return SignatureReport(
            valid=False,
            manifest_intact=manifeste_ok,
            binds_this_manifest=False,
            signature_valid=False,
            algorithm=signature.algorithm,
            key_id=signature.key_id,
            third_party_verifiable=False,
            detail=(
                f"portee incompatible : la signature couvre "
                f"{getattr(signature, 'scope', None)!r}, on verifie "
                f"{scope_attendu!r}. Une signature de recette ne dit rien des "
                "resultats."
            ),
        )

    lie = signature.content_hash == hash_attendu
    if not lie:
        motifs.append(
            "la signature vise un autre document : elle porte sur "
            f"{signature.content_hash[:16]}..., ce document vaut "
            f"{hash_attendu[:16]}..."
        )

    payload = signing_payload(
        signature.content_hash,
        signature.algorithm,
        signature.key_id,
        signature.scope,
    )
    # Un drapeau explicite plutot qu'une recherche de mot dans les messages.
    # La version precedente testait `"illisible" not in " ".join(motifs)` :
    # le comportement dependait alors d'un mot francais dans un texte destine a
    # l'utilisateur, et traduire ce texte aurait change la logique en silence.
    illisible = False
    try:
        sig_ok = verifier.verify(payload, bytes.fromhex(signature.signature))
    except ValueError:
        sig_ok = False
        illisible = True
        motifs.append("signature illisible : ce n'est pas de l'hexadecimal")
    if not sig_ok and not illisible:
        motifs.append("signature cryptographiquement invalide")

    try:
        tiers = SignatureAlgorithm(signature.algorithm).is_third_party_verifiable
    except ValueError:
        tiers = False

    valide = manifeste_ok and lie and sig_ok
    if valide and not tiers:
        motifs.append(
            "signature valide, mais SYMETRIQUE : elle prouve l'integrite pour le "
            "porteur de la cle, pas l'origine face a un tiers"
        )

    return SignatureReport(
        valid=valide,
        manifest_intact=manifeste_ok,
        binds_this_manifest=lie,
        signature_valid=sig_ok,
        algorithm=signature.algorithm,
        key_id=signature.key_id,
        third_party_verifiable=tiers and valide,
        detail=" | ".join(motifs) if motifs else "signature valide et opposable",
    )
