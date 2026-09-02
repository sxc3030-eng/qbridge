"""Chaine de scellement : rendre une SERIE d'executions inalterable.

CE QUE LES EMPREINTES PAR ARCHIVE NE FONT PAS. Chaque archive porte son propre
hash de contenu, et il attrape toute modification de SON contenu. Mais chaque
archive est un ilot : rien ne dit qu'il y en avait six.

L'ATTAQUE, MESUREE SUR LA SERIE DE PROFONDEUR REELLE. Six executions honnetes
sur ibm_marrakesh. Deux contredisent la these qu'on veut publier. On supprime
les deux dossiers. Les quatre restantes verifient TOUTES sans broncher :
manifeste intact, resultats intacts, hashes conformes. Aucun controle de
qbridge ne peut voir que deux runs ont disparu.

C'est la publication SELECTIVE, et c'est la forme la plus courante de
manipulation d'un resultat scientifique - bien plus repandue que la
fabrication de donnees, parce qu'elle ne demande de fabriquer RIEN.

CE QUE LA CHAINE APPORTE. Chaque entree porte l'empreinte de la precedente.
Supprimer, reordonner, inserer ou substituer une execution casse le chainage,
et la tete du journal s'engage sur toute l'histoire. Signer la tete revient a
signer la serie entiere.

CE QU'ELLE NE FAIT PAS, ET IL FAUT LE DIRE. Elle ne prouve pas que la serie est
COMPLETE. Une execution jamais inscrite n'y laisse aucune trace : on ne peut
pas prouver l'absence d'un evenement dont rien n'a garde memoire. Ce que la
chaine change, c'est le cout : effacer une entree deja inscrite oblige a
reecrire et resigner TOUT le journal a partir d'elle. Si la tete a ete signee,
datee ou communiquee ne serait-ce qu'une fois, la reecriture devient
detectable. Sans temoin exterieur, elle ne l'est pas.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from qbridge.digest import canonical_json, sha256_of

JOURNAL_SCHEMA_VERSION = "1.0"
JOURNAL_FILENAME = "journal.json"

GENESE = "0" * 64
"""Predecesseur de la premiere entree. Une chaine de longueur 1 doit etre
verifiable comme les autres, sans cas particulier dans le code de controle."""


@dataclass(frozen=True)
class JournalEntry:
    """Une execution inscrite, liee a celle qui la precede."""

    index: int
    label: str
    record_content_hash: str
    recorded_at: str
    previous_hash: str
    entry_hash: str = ""

    def _compute_hash(self) -> str:
        return sha256_of(
            {
                "schema_version": JOURNAL_SCHEMA_VERSION,
                "index": self.index,
                "label": self.label,
                "record_content_hash": self.record_content_hash,
                "recorded_at": self.recorded_at,
                "previous_hash": self.previous_hash,
            }
        )

    @classmethod
    def build(
        cls,
        *,
        index: int,
        label: str,
        record_content_hash: str,
        recorded_at: str,
        previous_hash: str,
    ) -> "JournalEntry":
        brut = cls(
            index=index,
            label=label,
            record_content_hash=record_content_hash,
            recorded_at=recorded_at,
            previous_hash=previous_hash,
        )
        return cls(**{**brut.__dict__, "entry_hash": brut._compute_hash()})

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JournalEntry":
        connus = {f for f in cls.__dataclass_fields__}
        inconnus = set(data) - connus
        if inconnus:
            raise ValueError(
                f"Champs inconnus dans une entree de journal : {sorted(inconnus)}. "
                "Les ignorer laisserait passer un champ ajoute par un tiers."
            )
        return cls(**data)


@dataclass(frozen=True)
class JournalReport:
    """Verdict sur une chaine. `intact` est le seul champ a lire pour decider."""

    intact: bool
    entries: int
    head: Optional[str]
    detail: str
    broken_at: Optional[int] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intact": self.intact,
            "entries": self.entries,
            "head": self.head,
            "detail": self.detail,
            "broken_at": self.broken_at,
            "warnings": list(self.warnings),
        }


class Journal:
    """Suite ordonnee d'executions, chainee par empreintes."""

    def __init__(self, entries: Optional[List[JournalEntry]] = None) -> None:
        self._entries: List[JournalEntry] = list(entries or [])

    # ---------- lecture ----------

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> List[JournalEntry]:
        return list(self._entries)

    @property
    def head(self) -> Optional[str]:
        """Empreinte de la derniere entree : elle s'engage sur toute l'histoire.

        Signer cette valeur revient a signer la serie entiere. C'est le seul
        chiffre a publier, dater ou communiquer pour rendre une reecriture
        ulterieure detectable.
        """
        return self._entries[-1].entry_hash if self._entries else None

    # ---------- ecriture ----------

    def append(self, record: Any, *, label: str) -> JournalEntry:
        """Inscrit une execution a la suite.

        `record` est un `RunRecord` : on inscrit son hash de CONTENU, qui couvre
        le manifeste et les tirages. Le journal ne duplique pas les donnees, il
        s'engage dessus.
        """
        if not label:
            raise ValueError(
                "Une entree sans etiquette serait introuvable : le journal ne "
                "stocke pas les donnees, seulement de quoi les retrouver."
            )
        if any(e.label == label for e in self._entries):
            raise ValueError(
                f"L'etiquette {label!r} est deja inscrite. Deux entrees de meme "
                "nom rendraient la verification ambigue."
            )
        entree = JournalEntry.build(
            index=len(self._entries),
            label=label,
            record_content_hash=record.content_hash(),
            recorded_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
            previous_hash=self.head or GENESE,
        )
        self._entries.append(entree)
        return entree

    # ---------- verification ----------

    def verify(self) -> JournalReport:
        """Controle le chainage seul. Ne lit aucune archive."""
        if not self._entries:
            return JournalReport(
                intact=True, entries=0, head=None, detail="journal vide"
            )

        precedent = GENESE
        for position, entree in enumerate(self._entries):
            if entree.index != position:
                return JournalReport(
                    intact=False,
                    entries=len(self._entries),
                    head=self.head,
                    broken_at=position,
                    detail=(
                        f"entree en position {position} numerotee {entree.index} : "
                        "une entree a ete supprimee, inseree ou reordonnee"
                    ),
                )
            if entree.previous_hash != precedent:
                return JournalReport(
                    intact=False,
                    entries=len(self._entries),
                    head=self.head,
                    broken_at=position,
                    detail=(
                        f"entree {position} ({entree.label!r}) ne suit pas la "
                        "precedente : le chainage est rompu"
                    ),
                )
            if entree.entry_hash != entree._compute_hash():
                return JournalReport(
                    intact=False,
                    entries=len(self._entries),
                    head=self.head,
                    broken_at=position,
                    detail=(
                        f"entree {position} ({entree.label!r}) : son empreinte "
                        "ne correspond pas a son contenu"
                    ),
                )
            precedent = entree.entry_hash

        return JournalReport(
            intact=True,
            entries=len(self._entries),
            head=self.head,
            detail=(
                f"{len(self._entries)} entree(s) chainees sans rupture ; la tete "
                "s'engage sur toute la serie"
            ),
        )

    def verify_records(self, base: str | Path) -> JournalReport:
        """Controle le chainage ET la presence de chaque archive inscrite.

        C'est ce controle-ci qui attrape la publication selective : une archive
        inscrite puis effacee laisse son entree derriere elle.
        """
        rapport = self.verify()
        if not rapport.intact:
            return rapport

        from qbridge.record import RunRecord

        racine = Path(base)
        manquantes: List[str] = []
        divergentes: List[str] = []
        avertissements: List[str] = []

        for entree in self._entries:
            dossier = racine / entree.label
            if not dossier.is_dir():
                manquantes.append(entree.label)
                continue
            try:
                record = RunRecord.load(dossier)
            except Exception as exc:
                avertissements.append(f"{entree.label} : illisible ({exc})")
                divergentes.append(entree.label)
                continue
            if record.content_hash() != entree.record_content_hash:
                divergentes.append(entree.label)

        if manquantes or divergentes:
            morceaux = []
            if manquantes:
                morceaux.append(f"ABSENTES : {sorted(manquantes)}")
            if divergentes:
                morceaux.append(f"NE CORRESPONDENT PLUS : {sorted(divergentes)}")
            return JournalReport(
                intact=False,
                entries=len(self._entries),
                head=self.head,
                detail=(
                    "le chainage est intact mais des archives inscrites "
                    "manquent a l'appel - " + " ; ".join(morceaux)
                ),
                warnings=avertissements,
            )

        return JournalReport(
            intact=True,
            entries=len(self._entries),
            head=self.head,
            detail=(
                f"{len(self._entries)} archive(s) presentes, conformes et "
                "chainees sans rupture"
            ),
            warnings=avertissements,
        )

    # ---------- persistance ----------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "head": self.head,
            "entries": [e.to_dict() for e in self._entries],
        }

    def save(self, path: str | Path) -> None:
        chemin = Path(path)
        if chemin.is_dir():
            chemin = chemin / JOURNAL_FILENAME
        chemin.parent.mkdir(parents=True, exist_ok=True)
        chemin.write_text(canonical_json(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Journal":
        chemin = Path(path)
        if chemin.is_dir():
            chemin = chemin / JOURNAL_FILENAME
        donnees = json.loads(chemin.read_text(encoding="utf-8"))

        version = donnees.get("schema_version")
        if version != JOURNAL_SCHEMA_VERSION:
            raise ValueError(
                f"Version de schema de journal incompatible : {version!r} "
                f"(attendu {JOURNAL_SCHEMA_VERSION!r})."
            )
        journal = cls([JournalEntry.from_dict(e) for e in donnees["entries"]])

        # LE CHAINAGE EST VERIFIE ICI, PAS SEULEMENT DANS `verify()`.
        #
        # Premiere version de ce module : `load` ne controlait que la tete
        # stockee. Retirer deux entrees AU MILIEU ne la change pas - la tete est
        # l'empreinte de la DERNIERE entree, restee intacte - et le journal se
        # chargeait sans broncher. Un appelant qui oubliait `verify()` tenait
        # alors une chaine rompue en croyant l'avoir validee.
        #
        # C'est un controle qui AVAIT L'AIR de valider. Charger un journal
        # rompu dans un objet utilisable est trop dangereux pour etre laisse a
        # la discipline de l'appelant : on refuse a la porte.
        rapport = journal.verify()
        if not rapport.intact:
            raise ValueError(
                f"Journal rompu a l'entree {rapport.broken_at} : {rapport.detail}"
            )

        # La tete stockee est REDONDANTE avec la chaine. Si elle diverge, c'est
        # que le fichier a ete edite : on refuse plutot que de faire confiance a
        # l'une des deux valeurs.
        tete = donnees.get("head")
        if tete != journal.head:
            raise ValueError(
                "La tete enregistree ne correspond pas a la chaine "
                f"({tete!r} contre {journal.head!r}) : le journal a ete edite."
            )
        return journal
