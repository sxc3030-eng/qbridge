"""Ne pas espionner la machine : l'interroger.

LE PROBLEME. Nos deux defauts les plus couteux de la journee ont la meme
racine : on demande a la machine de se decrire elle-meme, et on la croit.

    defaut 25 : la calibration utilisee datait de 43 heures
    defaut 26 : les chiffres publies etaient optimistes — 98.15 % annonces,
                96.19 % rendus

L'INSTINCT NATUREL est de vouloir un mouchard chez le fournisseur. Il ne
marcherait pas : il faudrait un acces qu'on n'aura jamais, et il ne ferait que
deplacer la question — qui verifie le mouchard ?

CE QU'ON FAIT A LA PLACE. On glisse des circuits DONT ON CONNAIT LA REPONSE. La
machine ne peut pas les distinguer des vrais. Ce qui leur manque pour atteindre
100 % est exactement l'erreur qu'elle infligera au vrai circuit — mesuree par
nous, au meme instant, sur les memes qubits.

C'est un domaine etabli : randomized benchmarking, circuits-pieges, et a
l'extreme le protocole de Mahadev (2018) ou un verificateur purement classique
valide un ordinateur quantique par la cryptographie.

LES TROIS PIEGES, choisis pour se completer :

    repos  : rien du tout        -> 000 certain. Mesure la LECTURE seule.
    plein  : X sur les trois     -> 111 certain. La lecture depuis |1>, donc
                                    l'ASYMETRIE que la relaxation cause.
    chaine : X puis les 2 CNOT   -> 111 certain. Lecture ET portes a deux
                                    qubits. Verifie : MEME nombre de cz que le
                                    GHZ, donc meme bruit dominant.

CE QUE CA CHANGE POUR LE VERDICT. La borne de plausibilite reposait sur une
fidelite PREDITE par la calibration publiee. Avec les pieges, elle repose sur
une fidelite MESUREE dans la meme session. Le verdict cesse de dependre de la
parole du fournisseur.
"""

from __future__ import annotations

import json
import time

import cirq
import numpy as np

from qbridge import capture
from qbridge.backends.ibm_runtime import IbmRuntimeBackend, backend_reel
from qbridge.calibration import CalibrationSnapshot
from qbridge.cli import _adoucir_les_flux
from qbridge.record import RunRecord
from qbridge.verdict import bitstring_counts

APPAREIL = "ibm_marrakesh"
QUBITS = [147, 148, 149]
TIRAGES = 1024
TOURS = 3
DOSSIER = "runs/pieges.json"


def circuits():
    q = cirq.LineQubit.range(3)
    mesure = cirq.measure(*q, key="m")
    return {
        "repos": (cirq.Circuit([cirq.I(q[0]), mesure]), {0b000}),
        "plein": (cirq.Circuit([cirq.X(q[0]), cirq.X(q[1]), cirq.X(q[2]), mesure]),
                  {0b111}),
        "chaine": (cirq.Circuit([cirq.X(q[0]), cirq.CNOT(q[0], q[1]),
                                 cirq.CNOT(q[1], q[2]), mesure]), {0b111}),
        "ghz": (cirq.Circuit([cirq.H(q[0]), cirq.CNOT(q[0], q[1]),
                              cirq.CNOT(q[1], q[2]), mesure]), {0b000, 0b111}),
    }


def main() -> int:
    _adoucir_les_flux()

    appareil = backend_reel(APPAREIL)._backend
    backend = IbmRuntimeBackend(appareil, initial_layout=QUBITS,
                                optimization_level=0)
    jeu = circuits()

    print(f"{TOURS} tours x {len(jeu)} circuits x {TIRAGES} tirages sur {QUBITS}")
    print("Les circuits sont ENTRELACES : la derive touche tout le monde pareil.")
    print()

    mesures = {nom: [] for nom in jeu}
    calibration = None
    for tour in range(TOURS):
        for nom, (circuit, attendu) in jeu.items():
            run = capture(circuit, backend=backend, seed=7, repetitions=TIRAGES)
            if calibration is None and run.manifest.calibration_json:
                calibration = CalibrationSnapshot.from_json(
                    run.manifest.calibration_json
                )
            comptes = bitstring_counts(run.samples["m"])
            total = sum(comptes.values())
            mesures[nom].append(sum(comptes.get(e, 0) for e in attendu) / total)
        print(f"  tour {tour + 1} : " + "  ".join(
            f"{nom}={100 * mesures[nom][-1]:.2f}%" for nom in jeu))

    moyennes = {nom: float(np.mean(v)) for nom, v in mesures.items()}
    incertitudes = {
        nom: (moyennes[nom] * (1 - moyennes[nom]) / (TIRAGES * TOURS)) ** 0.5
        for nom in jeu
    }

    print()
    print("=== MESURE contre DECLARATION ===")

    lecture = [calibration.qubits[f"q({q})"]["readout_error"].value for q in QUBITS]
    declaree_lecture = float(np.prod([1 - e for e in lecture]))
    print(f"  lecture, DECLAREE par IBM  : {100 * declaree_lecture:.2f} %")
    print(f"  lecture, MESUREE (repos)   : {100 * moyennes['repos']:.2f} %"
          f" +/- {100 * incertitudes['repos']:.2f}")
    print(f"  lecture, MESUREE (plein)   : {100 * moyennes['plein']:.2f} %"
          f" +/- {100 * incertitudes['plein']:.2f}")
    ecart = moyennes["repos"] - moyennes["plein"]
    print(f"  asymetrie |0> contre |1>   : {100 * ecart:+.2f} points")
    if abs(ecart) > 3 * (incertitudes["repos"] ** 2
                         + incertitudes["plein"] ** 2) ** 0.5:
        print("    -> significative : lire un 1 n'a pas le meme cout qu'un 0")

    print()
    print(f"  piege `chaine`, MESURE     : {100 * moyennes['chaine']:.2f} %"
          f" +/- {100 * incertitudes['chaine']:.2f}")
    print(f"  GHZ,            MESURE     : {100 * moyennes['ghz']:.2f} %"
          f" +/- {100 * incertitudes['ghz']:.2f}")

    print()
    print("=== UNE BORNE SANS LA PAROLE DU FOURNISSEUR ===")
    # Le piege `chaine` a le meme nombre de cz que le GHZ : sa fidelite mesuree
    # estime directement celle que le GHZ peut atteindre.
    f_piege = moyennes["chaine"]
    borne_mesuree = f_piege + (1 - f_piege) * (2 / 8)
    print(f"  fidelite du piege (mesuree) : {100 * f_piege:.2f} %")
    print(f"  borne qui en decoule        : {100 * borne_mesuree:.2f} %")
    print(f"  GHZ observe                 : {100 * moyennes['ghz']:.2f} %")
    if moyennes["ghz"] <= borne_mesuree:
        print("  -> le GHZ respecte la borne MESUREE. Aucune calibration publiee")
        print("     n'a servi a ce verdict.")
    else:
        print("  -> le GHZ DEPASSE la borne mesuree, ce qui demande une")
        print("     explication : piege plus bruite que le vrai circuit ?")

    with open(DOSSIER, "w", encoding="utf-8") as fichier:
        json.dump(
            {
                "qubits": QUBITS,
                "tours": TOURS,
                "tirages": TIRAGES,
                "mesures": mesures,
                "moyennes": moyennes,
                "lecture_declaree": declaree_lecture,
                "borne_mesuree": borne_mesuree,
            },
            fichier,
            indent=2,
        )
    print(f"\nmesures dans {DOSSIER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
