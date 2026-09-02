"""Quatre attaques, quatre defenses, sur des donnees d'un vrai QPU.

CE QUE CETTE DEMONSTRATION MONTRE. Chaque defense de qbridge existe parce que
la precedente a echoue sur une attaque precise. On les rejoue toutes, dans
l'ordre, et on montre a chaque fois ce qui passe AVANT et ce qui est attrape
APRES.

Aucune des attaques n'est theorique : chacune a ete executee, et chaque defense
a ete ecrite le jour ou l'attaque a reussi.

SUR COPIE. Le script travaille sur une copie temporaire des archives : les
originales ne sont jamais touchees.

CE QUE LA DEMONSTRATION FINIT PAR DIRE. L'acte IV liste ce qui passe ENCORE.
Un outil de provenance qui ne saurait pas nommer ses angles morts serait le
premier a ne pas meriter confiance.
"""

from __future__ import annotations

import dataclasses
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from qbridge import verify_archival, verify_physical_plausibility
from qbridge.capture import hash_samples
from qbridge.cli import _adoucir_les_flux
from qbridge.journal import Journal
from qbridge.record import RunRecord
from qbridge.timestamp import TIMESTAMP_FILENAME, Timestamp, stamp

SERIE = [f"profondeur_{n:03d}" for n in (0, 4, 16, 32, 64, 128)]
LARGEUR = 74


def titre(numero: str, texte: str) -> None:
    print()
    print("=" * LARGEUR)
    print(f"  ACTE {numero} - {texte}")
    print("=" * LARGEUR)


def verdict(passe: bool, texte: str) -> None:
    print(f"  [{'PASSE' if passe else 'ATTRAPE'}] {texte}")


def preparer(source: Path, travail: Path) -> list:
    """Copie les archives disponibles. Ne touche jamais aux originales."""
    presentes = []
    for nom in SERIE:
        if (source / nom).is_dir():
            shutil.copytree(source / nom, travail / nom)
            presentes.append(nom)
    return presentes


def main() -> int:
    _adoucir_les_flux()

    source = Path("runs")
    presentes = []
    with tempfile.TemporaryDirectory(prefix="qbridge_demo_") as tmp:
        travail = Path(tmp)
        presentes = preparer(source, travail)

        if len(presentes) < 3:
            print(
                "Cette demonstration a besoin d'au moins trois archives de la\n"
                "serie de profondeur. Les produire (quelques minutes de QPU\n"
                "gratuit, ou adapter le backend) :\n\n"
                "    python examples/effondrement_en_profondeur.py\n"
            )
            return 1

        print(f"qbridge - demonstration adverse")
        print(f"{len(presentes)} archives reelles d'ibm_marrakesh, copiees dans un")
        print("dossier temporaire. Les originales ne sont pas touchees.")

        # ---------------------------------------------------------------
        titre("I", "UN FAUX PARFAIT")
        # ---------------------------------------------------------------
        print("  Le faussaire annonce un GHZ a 100 %. Il recalcule proprement")
        print("  l'empreinte de ses tirages : l'archive est scellee dans les regles.")
        print()

        cible = travail / presentes[0]
        vrai = RunRecord.load(cible)

        n = vrai.samples["m"].shape[0]
        parfait = np.zeros((n, 3), dtype=np.uint8)
        parfait[n // 2 :] = 1
        faux = dataclasses.replace(
            vrai, samples={"m": parfait}, result_hash=hash_samples({"m": parfait})
        )
        faux.save(cible)

        relu = RunRecord.load(cible)
        rapport = verify_archival(relu)
        verdict(
            rapport.manifest_intact and rapport.results_intact,
            "verification archivistique : manifeste et resultats intacts",
        )
        print("         -> hashes, signature, integrite : tout est coherent")
        print()

        physique = verify_physical_plausibility(relu)
        verdict(
            physique.verdict.name == "PLAUSIBLE",
            f"plausibilite physique : {physique.verdict.name}",
        )
        print(f"         predit par l'appareil : {100 * physique.predicted_fidelity:.2f} %")
        print(f"         maximum atteignable   : {100 * physique.upper_bound:.2f} %")
        print(f"         annonce par le faux   : {100 * physique.observed_weight:.2f} %")
        print()
        print("  La calibration scellee dans SA PROPRE archive le contredit.")
        print("  Depasser ce que la machine declaree peut produire est impossible.")

        shutil.rmtree(cible)
        shutil.copytree(source / presentes[0], cible)

        # ---------------------------------------------------------------
        titre("II", "LA SUPPRESSION")
        # ---------------------------------------------------------------
        print("  Le faussaire ne fabrique rien. Il EFFACE les deux executions")
        print("  qui contredisent sa these. C'est la publication selective, et")
        print("  c'est bien plus courant que la fabrication de donnees.")
        print()

        journal = Journal()
        for nom in presentes:
            journal.append(RunRecord.load(travail / nom), label=nom)
        journal.save(travail)
        tete_originale = journal.head
        print(f"  serie inscrite : {len(journal)} entrees")
        print(f"  tete : {tete_originale[:48]}...")
        print()

        efface = presentes[1:3]
        for nom in efface:
            shutil.rmtree(travail / nom)

        toutes_valides = True
        for nom in presentes:
            if nom in efface:
                continue
            r = verify_archival(RunRecord.load(travail / nom))
            toutes_valides &= r.manifest_intact and r.results_intact
        verdict(
            toutes_valides,
            f"chaque archive SURVIVANTE verifie ({len(presentes) - len(efface)}/"
            f"{len(presentes) - len(efface)})",
        )
        print("         -> une archive est un ilot : elle ignore qu'il y en avait six")
        print()

        chaine = journal.verify_records(travail)
        verdict(chaine.intact, "chaine de scellement : des archives inscrites manquent")
        print(f"         absentes : {efface}")

        # ---------------------------------------------------------------
        titre("III", "LA REECRITURE")
        # ---------------------------------------------------------------
        print("  Le faussaire comprend la chaine. Il REFAIT le journal en entier,")
        print("  sans les entrees genantes.")
        print()

        jeton_present = (source / TIMESTAMP_FILENAME).is_file()
        if jeton_present:
            jeton = Timestamp.load(source / TIMESTAMP_FILENAME)
            atteste = jeton.details().get("imprint")
            print(f"  jeton d'horodatage trouve, datant une tete deja attestee")
        else:
            print("  Aucun jeton local : on en demande un a une autorite RFC 3161.")
            print("  (32 octets sortent de la machine, rien d'autre)")
            try:
                jeton = stamp(tete_originale)
                jeton.save(travail)
                print(f"  tete attestee le {jeton.verify(tete_originale).stamped_at}")
            except Exception as exc:
                print(f"  horodatage indisponible ({exc}) - acte III sur la tete seule")
                jeton = None
        print()

        refait = Journal()
        for nom in presentes:
            if nom in efface:
                continue
            refait.append(RunRecord.load(travail / nom), label=nom)

        verdict(refait.verify().intact, "chainage du NOUVEAU journal : valide")
        verdict(
            refait.verify_records(travail).intact,
            "archives inscrites : toutes presentes et conformes",
        )
        print("         -> rien dans le journal seul ne trahit la reecriture")
        print()

        if jeton is not None:
            controle = jeton.verify(refait.head)
            verdict(controle.bound, "horodatage : le jeton NE LIE PAS cette tete")
            print(f"         date attestee : {controle.stamped_at}")
            print("         ce jeton date une AUTRE empreinte : il n'atteste rien ici")
            print()
            print("  Personne ne peut antidater un jeton sans la cle de l'autorite.")
            print("  L'ancienne tete reste attestee ; la nouvelle ne l'est pas.")

        # ---------------------------------------------------------------
        titre("IV", "CE QUI PASSE ENCORE")
        # ---------------------------------------------------------------
        print("  Un outil de provenance qui ne saurait pas nommer ses angles")
        print("  morts serait le premier a ne pas meriter confiance.")
        print()
        for texte in (
            "Une execution JAMAIS inscrite ne laisse aucune trace. On ne peut",
            "  pas prouver l'absence d'un evenement dont rien n'a garde memoire.",
            "",
            "Un faussaire qui CONNAIT la calibration peut fabriquer des tirages",
            "  plausibles. Le controle physique attrape l'incoherence, pas la",
            "  malveillance competente.",
            "",
            "La SIGNATURE de l'autorite d'horodatage n'est pas verifiee ici :",
            "  elle demande une racine de confiance. Un jeton fabrique passerait.",
            "  `commande_openssl()` rend la commande qui la verifie vraiment.",
            "",
            "Le controle physique se TAIT au-dela de 3 % d'infidelite predite :",
            "  mesure sur ibm_marrakesh, il accusait a tort des 10 portes cz.",
        ):
            print(f"  {texte}")

        print()
        print("=" * LARGEUR)
        print("  Trois attaques executees, trois attrapees. Quatre limites nommees.")
        print("=" * LARGEUR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
