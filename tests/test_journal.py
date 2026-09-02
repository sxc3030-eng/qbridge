"""La chaine de scellement, et l'attaque qu'elle existe pour attraper.

LA PUBLICATION SELECTIVE. Six executions honnetes, deux qui derangent, on
supprime les deux dossiers. Avant le journal, les quatre restantes verifiaient
TOUTES sans broncher : manifeste intact, resultats intacts, hashes conformes.
Aucun controle ne pouvait voir que deux runs avaient disparu.

C'est la forme la plus courante de manipulation d'un resultat, et la seule qui
ne demande de fabriquer RIEN.
"""

from __future__ import annotations

import json
import shutil

import pytest

cirq = pytest.importorskip("cirq")

from qbridge import capture  # noqa: E402
from qbridge.journal import (  # noqa: E402
    GENESE,
    JOURNAL_SCHEMA_VERSION,
    Journal,
    JournalEntry,
)
from qbridge.record import RunRecord  # noqa: E402


def _archive(tmp_path, nom, graine):
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([cirq.H(q[0]), cirq.measure(*q, key="m")])
    record = RunRecord.from_capture(
        capture(circuit, backend="qsim", seed=graine, repetitions=50)
    )
    record.save(tmp_path / nom)
    return record


@pytest.fixture
def serie(tmp_path):
    """Six executions inscrites, comme la serie de profondeur reelle."""
    journal = Journal()
    for i in range(6):
        nom = f"run_{i:03d}"
        journal.append(_archive(tmp_path, nom, graine=i + 1), label=nom)
    journal.save(tmp_path)
    return journal, tmp_path


# ---------- l'attaque ----------


def test_supprimer_une_execution_EST_DETECTE(serie):
    """LE controle qui justifie ce module."""
    journal, base = serie
    assert journal.verify_records(base).intact

    shutil.rmtree(base / "run_002")
    shutil.rmtree(base / "run_003")

    rapport = journal.verify_records(base)
    assert not rapport.intact
    assert "run_002" in rapport.detail and "run_003" in rapport.detail
    assert "ABSENTES" in rapport.detail


def test_sans_journal_la_suppression_reste_INVISIBLE(serie):
    """Le contraste. Chaque archive est un ilot : elle ne sait pas qu'elle
    faisait partie d'une serie de six."""
    from qbridge import verify_archival

    _, base = serie
    shutil.rmtree(base / "run_002")

    for i in (0, 1, 3, 4, 5):
        record = RunRecord.load(base / f"run_{i:03d}")
        rapport = verify_archival(record)
        assert rapport.manifest_intact and rapport.results_intact, (
            "chaque archive survivante verifie parfaitement — c'est le probleme"
        )


def test_substituer_une_archive_est_detecte(serie, tmp_path):
    """Une archive remplacee par une AUTRE, elle-meme parfaitement scellee."""
    journal, base = serie
    shutil.rmtree(base / "run_002")
    _archive(base, "run_002", graine=999)

    rapport = journal.verify_records(base)
    assert not rapport.intact
    assert "NE CORRESPONDENT PLUS" in rapport.detail


# ---------- editer le journal lui-meme ----------


def _charger_altere(base, transformation):
    donnees = json.loads((base / "journal.json").read_text(encoding="utf-8"))
    (base / "altere.json").write_text(
        json.dumps(transformation(donnees)), encoding="utf-8"
    )
    return Journal.load(base / "altere.json")


def test_retirer_des_entrees_au_milieu_est_refuse(serie):
    """DEFAUT 29, dans ce module meme.

    Premiere version : `load` ne controlait que la tete stockee. Retirer deux
    entrees AU MILIEU ne la change pas — la tete est l'empreinte de la DERNIERE
    entree, restee intacte — et le journal se chargeait sans broncher. Un
    appelant qui oubliait `verify()` tenait une chaine rompue en croyant
    l'avoir validee. Un controle qui AVAIT L'AIR de valider.
    """
    _, base = serie
    with pytest.raises(ValueError, match="rompu"):
        _charger_altere(
            base,
            lambda d: {
                **d,
                "entries": [e for e in d["entries"] if e["index"] not in (2, 3)],
            },
        )


def test_retirer_puis_RENUMEROTER_est_refuse(serie):
    """La suppression naive laisse un trou dans les index. Un faussaire un peu
    soigneux renumerote — et c'est le chainage qui le rattrape."""
    _, base = serie

    def retirer_proprement(d):
        restants = [dict(e) for e in d["entries"] if e["index"] not in (2, 3)]
        for i, e in enumerate(restants):
            e["index"] = i
        return {**d, "entries": restants, "head": restants[-1]["entry_hash"]}

    with pytest.raises(ValueError, match="rompu"):
        _charger_altere(base, retirer_proprement)


def test_reordonner_est_refuse(serie):
    _, base = serie
    with pytest.raises(ValueError, match="rompu"):
        _charger_altere(base, lambda d: {**d, "entries": list(reversed(d["entries"]))})


def test_modifier_le_hash_d_une_archive_est_refuse(serie):
    _, base = serie
    with pytest.raises(ValueError, match="rompu"):
        _charger_altere(
            base,
            lambda d: {
                **d,
                "entries": [
                    {**e, "record_content_hash": "f" * 64} if e["index"] == 2 else e
                    for e in d["entries"]
                ],
            },
        )


def test_une_tete_incoherente_est_refusee(serie):
    _, base = serie
    with pytest.raises(ValueError, match="tete enregistree"):
        _charger_altere(base, lambda d: {**d, "head": "a" * 64})


def test_un_champ_inconnu_est_refuse(serie):
    """Les ignorer laisserait passer un champ ajoute par un tiers."""
    _, base = serie
    with pytest.raises(ValueError, match="inconnus"):
        _charger_altere(
            base,
            lambda d: {
                **d,
                "entries": [{**e, "note": "ajoute apres coup"} for e in d["entries"]],
            },
        )


# ---------- ce que la chaine NE fait pas ----------


def test_reconstruire_toute_la_chaine_produit_une_AUTRE_tete(serie, tmp_path):
    """LA limite honnete. Un faussaire qui controle le journal PEUT le refaire
    sans les entrees genantes, et le resultat sera valide.

    Ce que la chaine change, c'est le cout : il doit tout reecrire, et sa tete
    ne sera plus celle qui a ete publiee. Sans temoin exterieur — signature,
    horodatage, simple communication — la reecriture reste indetectable.
    """
    journal, base = serie
    tete_publiee = journal.head

    refait = Journal()
    for i in (0, 1, 4, 5):
        refait.append(RunRecord.load(base / f"run_{i:03d}"), label=f"run_{i:03d}")

    assert refait.verify().intact, "un journal refait EST valide en soi"
    assert refait.head != tete_publiee, "mais sa tete trahit la reecriture"


def test_une_execution_jamais_inscrite_ne_laisse_aucune_trace(serie, tmp_path):
    """On ne peut pas prouver l'absence d'un evenement dont rien n'a garde
    memoire. Le journal prouve « ces N executions, dans cet ordre », jamais
    « ce sont toutes celles qui ont eu lieu »."""
    journal, base = serie
    _archive(base, "jamais_inscrite", graine=42)
    assert journal.verify_records(base).intact, (
        "une archive hors journal ne le rompt pas — c'est une limite, pas un bug"
    )


# ---------- le chainage ----------


def test_la_premiere_entree_suit_la_GENESE(serie):
    journal, _ = serie
    assert journal.entries[0].previous_hash == GENESE
    assert len(GENESE) == 64


def test_chaque_entree_pointe_sur_la_precedente(serie):
    journal, _ = serie
    entrees = journal.entries
    for precedente, suivante in zip(entrees, entrees[1:]):
        assert suivante.previous_hash == precedente.entry_hash


def test_la_tete_change_a_chaque_ajout(tmp_path):
    journal = Journal()
    tetes = []
    for i in range(4):
        journal.append(_archive(tmp_path, f"r{i}", graine=i + 1), label=f"r{i}")
        tetes.append(journal.head)
    assert len(set(tetes)) == 4, "chaque ajout doit deplacer la tete"


def test_un_journal_vide_est_valide(tmp_path):
    """Sans cas particulier : une chaine de longueur 0 ou 1 se verifie comme
    les autres."""
    journal = Journal()
    rapport = journal.verify()
    assert rapport.intact and rapport.entries == 0 and rapport.head is None


def test_une_etiquette_en_double_est_refusee(tmp_path):
    """Deux entrees de meme nom rendraient la verification ambigue."""
    journal = Journal()
    journal.append(_archive(tmp_path, "a", graine=1), label="a")
    with pytest.raises(ValueError, match="deja inscrite"):
        journal.append(_archive(tmp_path, "b", graine=2), label="a")


def test_une_etiquette_vide_est_refusee(tmp_path):
    journal = Journal()
    with pytest.raises(ValueError, match="introuvable"):
        journal.append(_archive(tmp_path, "a", graine=1), label="")


# ---------- persistance ----------


def test_un_aller_retour_preserve_la_chaine(serie):
    journal, base = serie
    relu = Journal.load(base)
    assert relu.head == journal.head
    assert len(relu) == len(journal)
    assert relu.verify().intact


def test_une_version_de_schema_inconnue_est_refusee(serie):
    _, base = serie
    with pytest.raises(ValueError, match="schema de journal"):
        _charger_altere(base, lambda d: {**d, "schema_version": "0.1"})


def test_le_schema_est_dans_l_empreinte_des_entrees():
    """Sans lui, changer le sens d'un champ sans changer son nom passerait."""
    entree = JournalEntry.build(
        index=0,
        label="x",
        record_content_hash="a" * 64,
        recorded_at="2026-01-01T00:00:00+00:00",
        previous_hash=GENESE,
    )
    assert JOURNAL_SCHEMA_VERSION in json.dumps(
        {"schema_version": JOURNAL_SCHEMA_VERSION}
    )
    assert entree.entry_hash == entree._compute_hash()
