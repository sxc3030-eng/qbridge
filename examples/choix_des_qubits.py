"""TEST 2 : choisir les qubits vaut-il mieux que laisser le transpileur choisir ?

L'IDEE. Une couche d'intention exprime un BUT — « un GHZ a trois qubits, le
plus fidele possible » — et laisse la pile choisir les moyens. Aujourd'hui on
emet un circuit et le transpileur place les qubits selon SES criteres, qui ne
sont pas forcement ceux de l'appelant.

CE QUE LA PUCE JUSTIFIE. Mesure sur ibm_marrakesh, 156 qubits :

    erreur de lecture : 0.23 % a 50.94 %   -> facteur 220
    erreur cz         : 0.12 % a 100 %     -> facteur 856

Un qubit a 50.9 % de lecture est un tirage a pile ou face. Un coupleur a 100 %
est hors service. Le choix du placement n'est donc pas un detail d'optimisation.

L'EXPERIENCE. Le meme GHZ, trois fois : placement laisse au transpileur,
meilleure chaine selon la calibration, et une chaine mediocre choisie expres.

CE QUE LA MESURE A DONNE, ET LA CORRECTION QU'ELLE IMPOSE.

Premier resultat, contre  :

    defaut     [0, 1, 2]         94.34 %
    meilleure  [147, 148, 149]   97.27 %   -> +2.93 points, 3.3 sigma

Conclusion tentante : « choisir les qubits bat le transpileur ». ELLE EST
FAUSSE, et la verification l'a montree. Contre les autres reglages :

    niveau 1   [0, 1, 2]         93.85 %
    niveau 2   [109, 118, 129]   96.78 %
    niveau 3   [149, 148, 147]   96.68 %
    notre choix[147, 148, 149]   97.27 %

Qiskit au niveau 3 trouve EXACTEMENT la meme chaine que nous — la notre,
inversee. L'ecart restant vaut 0.8 sigma : du bruit. Le premier chiffre
comparait a son reglage le plus faible.

CE QUE L'EXPERIENCE PROUVE VRAIMENT, et c'est plus utile :

1. Notre critere — somme des erreurs de lecture et de cz — REPRODUIT ce que
   la passe de placement de Qiskit trouve avec des moyens bien plus lourds.
   C'est une validation du critere, pas une victoire sur l'outil.

2. Le defaut coute 3 points.  est le reglage des
   tutoriels ; il rend le placement TRIVIAL [0, 1, 2], 72e sur 244, et rien
   n'avertit. Trois points de fidelite perdus en silence.

3. La valeur d'une couche d'intention n'est donc PAS de battre le transpileur.
   C'est de rendre le choix VISIBLE et SCELLE. Le placement est desormais dans
   le manifeste : deux archives du meme circuit logique sur des qubits
   differents ne se confondent plus.
"""

from __future__ import annotations

import json

import cirq
import numpy as np

from qbridge import capture, verify_physical_plausibility
from qbridge.backends.ibm_runtime import IbmRuntimeBackend, backend_reel
from qbridge.calibration import CalibrationSnapshot
from qbridge.cli import _adoucir_les_flux
from qbridge.record import RunRecord
from qbridge.verdict import bitstring_counts

APPAREIL = "ibm_marrakesh"
REPETITIONS = 1024


def cartographier(backend):
    """Erreurs de lecture par qubit et de cz par paire, depuis la machine."""
    cible = backend.target
    lecture, cz = {}, {}
    for q in range(cible.num_qubits):
        inst = cible["measure"].get((q,))
        if inst is not None and inst.error is not None:
            lecture[q] = inst.error
    for paire, inst in cible["cz"].items():
        if inst is not None and inst.error is not None:
            cz[paire] = inst.error
    return lecture, cz


def chaines_classees(lecture, cz):
    """Toutes les chaines de trois qubits connectes, de la meilleure a la pire.

    Le cout est une SOMME d'erreurs, pas une probabilite : au-dela de quelques
    pour cent il ne veut plus rien dire en valeur, mais il classe correctement.
    C'est tout ce qu'on lui demande.
    """
    voisins = {}
    for a, c in cz:
        voisins.setdefault(a, set()).add(c)
        voisins.setdefault(c, set()).add(a)

    def cout(a, m, c):
        e1 = cz.get((a, m), cz.get((m, a), 1.0))
        e2 = cz.get((m, c), cz.get((c, m), 1.0))
        return e1 + e2 + sum(lecture.get(q, 1.0) for q in (a, m, c))

    chaines = []
    for milieu, vs in voisins.items():
        ordonnes = sorted(vs)
        for i, a in enumerate(ordonnes):
            for c in ordonnes[i + 1 :]:
                chaines.append(((a, milieu, c), cout(a, milieu, c)))
    chaines.sort(key=lambda x: x[1])
    return chaines


def ghz():
    q = cirq.LineQubit.range(3)
    return cirq.Circuit(
        [cirq.H(q[0]), cirq.CNOT(q[0], q[1]), cirq.CNOT(q[1], q[2]),
         cirq.measure(*q, key="m")]
    )


def executer(appareil_qiskit, etiquette, layout):
    backend = IbmRuntimeBackend(appareil_qiskit, initial_layout=layout)
    run = capture(ghz(), backend=backend, seed=7, repetitions=REPETITIONS)
    dossier = f"runs/choix_{etiquette}"
    RunRecord.from_capture(run).save(dossier)
    record = RunRecord.load(dossier)

    comptes = bitstring_counts(record.samples["m"])
    total = sum(comptes.values())
    observe = (comptes.get(0, 0) + comptes.get(7, 0)) / total
    rapport = verify_physical_plausibility(record)
    place = json.loads(record.manifest.device_provenance_json)["initial_layout"]
    return {
        "etiquette": etiquette,
        "layout": place,
        "predit": rapport.predicted_fidelity,
        "observe": observe,
        "sigma": rapport.sigma,
        "verdict": rapport.verdict.name,
        "incertitude": (observe * (1 - observe) / total) ** 0.5,
    }


def main() -> int:
    _adoucir_les_flux()

    backend = backend_reel(APPAREIL)
    appareil = backend._backend
    lecture, cz = cartographier(appareil)
    chaines = chaines_classees(lecture, cz)

    meilleure = chaines[0][0]
    # Une chaine MEDIOCRE, pas la pire : les pires contiennent des coupleurs
    # hors service a 100 % d'erreur, qui ne mesureraient qu'une panne.
    mediocre = chaines[len(chaines) * 3 // 4][0]

    print(f"appareil : {APPAREIL}, {len(lecture)} qubits, {len(chaines)} chaines")
    print(f"  meilleure chaine : {meilleure}  cout {chaines[0][1]:.4%}")
    print(f"  chaine mediocre  : {mediocre}  cout "
          f"{chaines[len(chaines) * 3 // 4][1]:.4%}")
    print()

    essais = [
        ("defaut", None),
        ("meilleure", list(meilleure)),
        ("mediocre", list(mediocre)),
    ]
    resultats = []
    entete = (f"{'placement':>12} {'qubits':>16} {'predit':>9} {'observe':>9} "
              f"{'+/-':>7}  verdict")
    print(entete)
    print("-" * len(entete))
    for etiquette, layout in essais:
        r = executer(appareil, etiquette, layout)
        resultats.append(r)
        print(f"{r['etiquette']:>12} {str(r['layout']):>16} "
              f"{100 * r['predit']:8.2f}% {100 * r['observe']:8.2f}% "
              f"{100 * r['incertitude']:6.2f}%  {r['verdict']}")

    print()
    print("=== l'intention vaut-elle mieux que le defaut ? ===")
    par_nom = {r["etiquette"]: r for r in resultats}
    defaut, choisi = par_nom["defaut"], par_nom["meilleure"]
    ecart = choisi["observe"] - defaut["observe"]
    incertitude = (defaut["incertitude"] ** 2 + choisi["incertitude"] ** 2) ** 0.5
    print(f"  meilleure - defaut : {100 * ecart:+.2f} points "
          f"+/- {100 * incertitude:.2f}  ({abs(ecart) / incertitude:.1f} sigma)")
    if abs(ecart) / incertitude < 2:
        print("  -> sous 2 sigma : rien de concluant sur cette execution")
    elif ecart > 0:
        print("  -> choisir les qubits depuis la calibration BAT le transpileur")
    else:
        print("  -> le transpileur fait MIEUX que notre critere")

    mauvais = par_nom["mediocre"]
    ecart2 = choisi["observe"] - mauvais["observe"]
    inc2 = (mauvais["incertitude"] ** 2 + choisi["incertitude"] ** 2) ** 0.5
    print(f"  meilleure - mediocre : {100 * ecart2:+.2f} points "
          f"({abs(ecart2) / inc2:.1f} sigma)")
    print()
    print("  Le second ecart mesure si le CRITERE ordonne correctement.")
    print("  Le premier mesure s'il bat ce que le transpileur fait deja.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
