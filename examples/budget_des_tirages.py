"""TEST 3 : ou faut-il depenser son budget de tirages ?

LA QUESTION NAIVE. « Combien de tirages pour telle precision ? » La reponse est
connue et sans interet : l'incertitude statistique vaut sqrt(p(1-p)/n), donc la
diviser par deux coute QUATRE fois plus de tirages.

LA VRAIE QUESTION. Cette formule ne vaut que si la machine reste la meme entre
le premier tirage et le dernier. Elle ne le reste pas.

Indice releve sur la journee : six executions du MEME circuit sur les MEMES
qubits ont rendu 97.27, 98.05, 97.85, 96.78, 94.34 et 93.85 %. Quatre points
d'ecart, quand l'incertitude statistique vaut 0.5 point. Si la derive domine, un
budget depense en tirages sur UNE execution achete beaucoup moins qu'un budget
reparti sur PLUSIEURS.

L'EXPERIENCE. Le meme circuit, N fois, sur les memes qubits, a budget constant.
On compare deux incertitudes :

    intra-execution : sqrt(p(1-p)/tirages), ce que la statistique predit
    inter-execution : l'ecart-type reel des N resultats

Si la seconde depasse largement la premiere, la machine derive entre les
executions, et le conseil de budget change du tout au tout.

CE QUE LA MESURE A DONNE : L HYPOTHESE EST REFUTEE.

    10 executions x 512 tirages sur [147, 148, 149]
    intra (statistique) : 0.704 %
    inter (mesure)      : 0.726 %
    rapport             : 1.03x
    chi2/ddl            : 1.04 pour 9 degres de liberte

Les dix executions viennent d une SEULE loi. Sur deux minutes, la derive est
indetectable et la formule naive suffit : le budget se depense en tirages.

D OU VENAIENT ALORS MES QUATRE POINTS D ECART ? De DEUX confusions, pas d une
derive rapide.

1. Le PLACEMENT. La plupart des executions que je comparais tournaient sur des
   qubits differents. Le test 2 a mesure 3 points d ecart entre le placement
   par defaut et le meilleur : voila l essentiel du 4.

2. L ECHELLE DE TEMPS. Sur des HEURES, la derive est reelle. La fidelite que la
   machine PREDIT pour [0, 1, 2] a valu 97.38 %, puis 98.15 %, puis 95.69 % au
   fil de la journee, IBM republiant sa calibration entre-temps. L appareil se
   redecrit differemment d une heure a l autre.

LE CONSEIL DE BUDGET, DONC, DEPEND DE L ECHELLE :

    minutes  -> la statistique gouverne. Depenser en tirages.
    heures   -> la machine change. Un resultat vieux de quelques heures ne
                decrit plus le meme appareil, et repeter n y change rien.

Et la lecon de methode : j avais attribue a la DERIVE ce que causait le
PLACEMENT. Les deux effets valent chacun quelques points, ils se melangent dans
une serie non controlee, et seule une experience a placement FIXE les separe.
"""

from __future__ import annotations

import json
import time

import cirq
import numpy as np

from qbridge import capture
from qbridge.backends.ibm_runtime import IbmRuntimeBackend, backend_reel
from qbridge.cli import _adoucir_les_flux
from qbridge.record import RunRecord
from qbridge.verdict import bitstring_counts

APPAREIL = "ibm_marrakesh"
QUBITS = [147, 148, 149]  # la meilleure chaine, mesuree au test 2
EXECUTIONS = 10
TIRAGES = 512
DOSSIER = "runs/budget.json"


def ghz():
    q = cirq.LineQubit.range(3)
    return cirq.Circuit(
        [cirq.H(q[0]), cirq.CNOT(q[0], q[1]), cirq.CNOT(q[1], q[2]),
         cirq.measure(*q, key="m")]
    )


def main() -> int:
    _adoucir_les_flux()

    appareil = backend_reel(APPAREIL)._backend
    backend = IbmRuntimeBackend(appareil, initial_layout=QUBITS)
    circuit = ghz()

    print(f"{EXECUTIONS} executions x {TIRAGES} tirages sur {QUBITS}")
    print(f"budget total : {EXECUTIONS * TIRAGES} tirages")
    print()
    print(f"{'#':>3} {'fidelite':>10} {'+/- stat':>10} {'secondes':>9}")
    print("-" * 36)

    mesures = []
    for i in range(EXECUTIONS):
        debut = time.time()
        run = capture(circuit, backend=backend, seed=7, repetitions=TIRAGES)
        duree = time.time() - debut
        comptes = bitstring_counts(run.samples["m"])
        total = sum(comptes.values())
        f = (comptes.get(0, 0) + comptes.get(7, 0)) / total
        stat = (f * (1 - f) / total) ** 0.5
        mesures.append({"fidelite": f, "stat": stat, "secondes": duree})
        print(f"{i:>3} {100 * f:9.2f}% {100 * stat:9.2f}% {duree:8.1f}")

    f = np.array([m["fidelite"] for m in mesures])
    stat_moyenne = float(np.mean([m["stat"] for m in mesures]))
    inter = float(np.std(f, ddof=1))

    print()
    print("=== ce que le budget achete vraiment ===")
    print(f"  incertitude INTRA-execution (statistique) : {100 * stat_moyenne:.3f} %")
    print(f"  ecart-type INTER-executions (mesure)      : {100 * inter:.3f} %")
    print(f"  rapport : {inter / stat_moyenne:.1f}x")
    print(f"  etendue : {100 * f.min():.2f} % a {100 * f.max():.2f} %")

    print()
    if inter > 2 * stat_moyenne:
        print("  LA DERIVE DOMINE. Ajouter des tirages a UNE execution achete")
        print("  une precision que la machine ne tient pas entre deux jobs.")
        print()
        # Incertitude sur la moyenne, selon qu'on croit ou non a la stabilite.
        naif = stat_moyenne / (EXECUTIONS ** 0.5)
        reel = inter / (EXECUTIONS ** 0.5)
        print(f"  incertitude sur la moyenne, modele naif : {100 * naif:.3f} %")
        print(f"  incertitude sur la moyenne, mesuree     : {100 * reel:.3f} %")
        print(f"  le modele naif SOUS-ESTIME d'un facteur {reel / naif:.1f}")
        print()
        print("  Consequence pratique : a budget egal, repartir sur plusieurs")
        print("  executions donne une moyenne plus honnete qu'une seule longue.")
        print("  Une seule execution ne mesure PAS la machine, elle mesure la")
        print("  machine A CET INSTANT.")
    else:
        print("  La derive ne domine pas sur cette serie : la formule")
        print("  statistique suffit, et le budget se depense en tirages.")

    with open(DOSSIER, "w", encoding="utf-8") as fichier:
        json.dump(
            {
                "qubits": QUBITS,
                "executions": EXECUTIONS,
                "tirages": TIRAGES,
                "mesures": mesures,
                "stat_moyenne": stat_moyenne,
                "inter": inter,
            },
            fichier,
            indent=2,
        )
    print(f"\nmesures dans {DOSSIER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
