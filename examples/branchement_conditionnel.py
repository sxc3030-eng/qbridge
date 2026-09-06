"""Un « si » que la machine tranche elle-meme, a froid, en cours de route.

CE QUI N'A JAMAIS TOURNE SUR UN QPU. Un programme, au sens ordinaire. Les
machines quantiques executent des CIRCUITS : des sequences droites, figees,
sans boucle, sans branchement, sans memoire. Tout est deroule d'avance. C'est
plus proche d'un rouleau de piano mecanique que d'un programme.

Aucun calcul quantique coherent n'a jamais dure plus d'une milliseconde. Le
circuit le plus profond de cette journee faisait 44 microsecondes.

CE QUI EST TOUT DE MEME POSSIBLE. `ibm_marrakesh` expose `if_else` dans son jeu
d'operations natives. La machine sait mesurer un qubit AU MILIEU du circuit,
lire le resultat, et decider de la suite — pendant que les autres qubits sont
encore coherents. Quelques centaines de nanosecondes pour lire, decider, agir.

C'est le premier pas vers quelque chose qui ressemble a un programme, et c'est
ce qui rend la correction d'erreur possible : sans branchement temps reel, pas
de code de surface.

C'EST AUSSI LE SEUL CHEMIN QUE CE PROJET N'AVAIT JAMAIS EXERCE. Le mode
MIDCIRCUIT_SAMPLING est celui ou l'on a mesure, en simulation, que
`cpu_threads` change les bitstrings. Sur du vrai materiel, personne ici ne
savait ce qu'il faisait.

L'EXPERIENCE, ET SON TEMOIN.

    conditionnel : H sur q0, mesurer q0, SI 1 alors X sur q1, mesurer q1
    temoin       : le meme, SANS la condition

Si le branchement s'execute, q1 doit valoir q0 a chaque tirage. S'il est
silencieusement ignore — ce qui arriverait sans un temoin pour le voir — q1
resterait a 0, et la correlation tomberait a 50 %.

Le temoin n'est pas une precaution de style : sans lui, un branchement jamais
evalue produirait un resultat parfaitement lisible et parfaitement faux.
"""

from __future__ import annotations

import json
import pathlib

import cirq
import numpy as np

from qbridge import capture
from qbridge.backends.ibm_runtime import IbmRuntimeBackend, backend_reel
from qbridge.cli import _adoucir_les_flux
from qbridge.digest import canonical_json, sha256_of
from qbridge.record import RunRecord
from qbridge.timestamp import Timestamp, stamp

APPAREIL = "ibm_marrakesh"
TIRAGES = 1024
DOSSIER = "runs/branchement"


def conditionnel():
    """Le branchement : X sur q1 seulement si la mesure de q0 vaut 1."""
    q = cirq.LineQubit.range(2)
    return cirq.Circuit(
        [
            cirq.H(q[0]),
            cirq.measure(q[0], key="a"),
            cirq.X(q[1]).with_classical_controls("a"),
            cirq.measure(q[1], key="b"),
        ]
    )


def temoin():
    """Le meme circuit SANS condition. q1 doit rester a 0."""
    q = cirq.LineQubit.range(2)
    return cirq.Circuit(
        [
            cirq.H(q[0]),
            cirq.measure(q[0], key="a"),
            cirq.I(q[1]),
            cirq.measure(q[1], key="b"),
        ]
    )


def meilleure_paire(cible):
    """La paire connectee la plus fiable : lecture + cz."""
    lecture = {}
    for q in range(cible.num_qubits):
        inst = cible["measure"].get((q,))
        if inst is not None and inst.error is not None:
            lecture[q] = inst.error
    paires = []
    for (a, b), inst in cible["cz"].items():
        if inst is None or inst.error is None or a >= b:
            continue
        if a in lecture and b in lecture:
            paires.append(((a, b), inst.error + lecture[a] + lecture[b]))
    paires.sort(key=lambda x: x[1])
    return paires[0][0], lecture


def executer(appareil, circuit, layout, nom):
    backend = IbmRuntimeBackend(
        appareil, initial_layout=list(layout), optimization_level=0
    )
    run = capture(circuit, backend=backend, seed=7, repetitions=TIRAGES)
    dossier = f"{DOSSIER}/{nom}"
    RunRecord.from_capture(run).save(dossier)
    record = RunRecord.load(dossier)

    a = record.samples["a"][:, 0].astype(int)
    b = record.samples["b"][:, 0].astype(int)
    return {
        "nom": nom,
        "mode_scelle": record.manifest.mode,
        "accord": float((a == b).mean()),
        "q0_vaut_1": float(a.mean()),
        "q1_vaut_1": float(b.mean()),
        "tirages": len(a),
    }


def main() -> int:
    _adoucir_les_flux()

    appareil = backend_reel(APPAREIL)._backend
    paire, lecture = meilleure_paire(appareil.target)
    print(f"appareil : {APPAREIL}")
    print(f"paire retenue : {paire}, lecture "
          f"{lecture[paire[0]]:.3%} et {lecture[paire[1]]:.3%}")

    # ---- prediction, AVANT execution ----
    fiabilite = (1 - lecture[paire[0]]) * (1 - lecture[paire[1]])
    document = {
        "appareil": APPAREIL,
        "paire": list(paire),
        "tirages": TIRAGES,
        "attendu_conditionnel": fiabilite,
        "attendu_temoin": 0.5,
        "raisonnement": (
            "si le branchement s'execute, q1 == q0 a chaque tirage, aux "
            "erreurs de lecture pres. Sans branchement, q1 reste a 0 et "
            "l'accord tombe au hasard, soit 50 %."
        ),
    }
    empreinte = sha256_of(document)
    print()
    print("=== PREDICTION SCELLEE, avant execution ===")
    print(f"  accord attendu, conditionnel : {100 * fiabilite:.2f} %")
    print(f"  accord attendu, temoin       : 50 %")
    print(f"  empreinte : {empreinte[:48]}...")
    jeton = stamp(empreinte)
    print(f"  datee le  : {jeton.verify(empreinte).stamped_at}")

    pathlib.Path(DOSSIER).mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{DOSSIER}/prediction.json").write_text(
        canonical_json(document), encoding="utf-8"
    )
    jeton.save(f"{DOSSIER}/prediction.tsr")

    # ---- execution ----
    print()
    print("=== EXECUTION ===")
    resultats = [
        executer(appareil, conditionnel(), paire, "conditionnel"),
        executer(appareil, temoin(), paire, "temoin"),
    ]
    entete = f"{'circuit':>14} {'mode scelle':>22} {'q0=1':>7} {'q1=1':>7} {'accord':>9}"
    print(entete)
    print("-" * len(entete))
    for r in resultats:
        print(f"{r['nom']:>14} {r['mode_scelle']:>22} "
              f"{100 * r['q0_vaut_1']:6.1f}% {100 * r['q1_vaut_1']:6.1f}% "
              f"{100 * r['accord']:8.2f}%")

    cond, tem = resultats
    u = (0.25 / TIRAGES) ** 0.5

    print()
    print("=== LE BRANCHEMENT A-T-IL EU LIEU ? ===")
    ecart = (cond["accord"] - tem["accord"]) / (u * 2**0.5)
    print(f"  conditionnel : {100 * cond['accord']:.2f} % d'accord")
    print(f"  temoin       : {100 * tem['accord']:.2f} % d'accord")
    print(f"  ecart        : {ecart:.1f} sigma")
    print()
    if cond["accord"] > 0.9 and tem["accord"] < 0.6:
        print("  OUI. La machine a mesure un qubit, lu le resultat, et applique")
        print("  une porte en consequence — pendant que l'autre qubit etait")
        print("  encore coherent. Le temoin prouve que la condition a bien ete")
        print("  EVALUEE et non ignoree.")
    elif cond["accord"] < 0.6:
        print("  NON. Le conditionnel se comporte comme le temoin : la branche")
        print("  n'a pas ete evaluee, ou la conversion l'a perdue en route.")
    else:
        print("  RESULTAT AMBIGU : ni franc accord ni franc hasard.")

    resultat = {**document, "observe": resultats, "ecart_sigma": ecart}
    pathlib.Path(f"{DOSSIER}/resultat.json").write_text(
        canonical_json(resultat), encoding="utf-8"
    )

    print()
    relu = Timestamp.load(f"{DOSSIER}/prediction.tsr")
    controle = relu.verify(empreinte)
    print(f"=== OPPOSABLE ===")
    print(f"  prediction attestee le : {controle.stamped_at}")
    print(f"  liaison                : {controle.bound}")
    print(f"  mode scelle            : {cond['mode_scelle']}")
    print(f"\ntout est dans {DOSSIER}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
