"""Predire AVANT de mesurer, et le prouver.

LE PROBLEME QUE CA REGLE. Tout ce qui precede compare une prediction a une
mesure — mais rien ne prouvait que la prediction venait EN PREMIER. Un modele
ajuste apres coup predit toujours bien. C'est le defaut le plus repandu de la
litterature scientifique, et aucune signature ne l'attrape : on peut signer un
mensonge sincere.

CE QUE FAIT CE SCRIPT, dans cet ordre strict :

    1. transpiler hors ligne les circuits, sans toucher au QPU
    2. predire la fidelite de chacun depuis la calibration publiee
    3. SCELLER ces predictions et faire DATER leur empreinte par une autorite
    4. SEULEMENT ENSUITE, executer sur la machine
    5. comparer, en pouvant prouver que 3 precede 4

L'horodatage rend l'ordre opposable. Le jeton porte l'empreinte des chiffres
predits ; il a ete emis quand les tirages n'existaient pas encore. Antidater
demanderait la cle de l'autorite.

CE QUE CA NE PROUVE PAS. Que le modele est bon — seulement qu'il n'a pas ete
ajuste. Un modele mauvais et pre-enregistre reste mauvais, et le dira.

RESULTAT, pre-enregistre le 2026-09-03 a 12:53:26 UTC par le TSA du DFN, cinq
chaines de qubits etalees sur la gamme de qualite d'ibm_marrakesh :

    chaine            predit   observe    ecart
    [4, 5, 6]         98.47 %  96.97 %   -1.50
    [102, 103, 104]   96.71 %  94.24 %   -2.47
    [126, 127, 128]   95.63 %  93.95 %   -1.69
    [77, 85, 84]      94.89 %  95.41 %   +0.52
    [58, 71, 72]      92.66 %  89.94 %   -2.72

CE QUE LE MODELE FAIT BIEN : il CLASSE. Correlation de Pearson 0.873, tau de
Kendall 0.60, huit paires correctement ordonnees sur dix. Choisir la meilleure
chaine depuis la calibration marche.

CE QU'IL FAIT MAL : il SURESTIME, systematiquement. Biais de -1.57 point, a
4.8 sigma, et quatre chaines sur cinq sous-performent. Ce n'est pas du bruit,
c'est un decalage constant.

POURQUOI C'EST LE RESULTAT LE PLUS UTILE DE LA SERIE. Le defaut 26 avait ete
corrige le matin meme, PAR PRINCIPE : la calibration publiee ne compte que les
erreurs declarees, tout ce qu'elle ignore — erreurs coherentes, diaphonie,
derive — ne peut que degrader, donc accuser un resultat « moins bon que
predit » est faux par construction.

Le principe est desormais MESURE, et la prediction etait horodatee avant que la
moindre donnee existe.

MAIS LA REPLICATION CORRIGE L'AMPLITUDE. Une SECONDE serie pre-enregistree,
memes chaines, vingt minutes plus tard :

    serie 1 (12:53) : biais -1.57 point,  5.0 sigma
    serie 2 (13:11) : biais -0.40 point,  1.3 sigma
    ecart entre les deux : 2.6 sigma

Les deux series ne mesurent pas le meme biais. Le « 1.57 point a 5 sigma » de
la premiere decrivait UNE session, pas la machine — exactement le genre de
chiffre qu'on cite ensuite comme une constante.

CE QUI TIENT APRES REPLICATION : le SIGNE, pas l'amplitude. Sur dix mesures
pre-enregistrees, sept sont negatives, et les dix ensemble donnent -0.99 point
a 4.5 sigma. Le modele penche du cote optimiste, ce qui suffit a justifier que
seule la borne superieure puisse accuser. L'amplitude, elle, derive avec
l'appareil.

C'est le pre-enregistrement qui rend cette correction possible : les deux
series etaient scellees et datees avant leurs donnees, donc aucune des deux ne
peut etre ecartee comme « un mauvais jour ».
"""

from __future__ import annotations

import json
import time

import cirq
import numpy as np

from qbridge import capture
from qbridge.backends.ibm_runtime import IbmRuntimeBackend, backend_reel
from qbridge.cli import _adoucir_les_flux
from qbridge.digest import canonical_json, sha256_of
from qbridge.journal import Journal
from qbridge.record import RunRecord
from qbridge.timestamp import Timestamp, stamp
from qbridge.verdict import bitstring_counts

APPAREIL = "ibm_marrakesh"
TIRAGES = 1024
CHAINES_VOULUES = 5
DOSSIER = "runs/prediction"


def etiquette(chaine):
    """Sujet commun a une prediction et a son execution."""
    return "chaine_" + "_".join(str(q) for q in chaine)


def ghz():
    q = cirq.LineQubit.range(3)
    return cirq.Circuit(
        [cirq.H(q[0]), cirq.CNOT(q[0], q[1]), cirq.CNOT(q[1], q[2]),
         cirq.measure(*q, key="m")]
    )


def cartographier(cible):
    lecture, cz = {}, {}
    for q in range(cible.num_qubits):
        inst = cible["measure"].get((q,))
        if inst is not None and inst.error is not None:
            lecture[q] = inst.error
    for paire, inst in cible["cz"].items():
        if inst is not None and inst.error is not None:
            cz[paire] = inst.error
    return lecture, cz


def predire(cible, chaine, operations):
    """Fidelite predite d'un GHZ, depuis la calibration SEULE.

    Meme modele multiplicatif que `fidelite_predite`, applique aux operations
    EXACTES issues de la transpilation — obtenues hors ligne, avant tout appel
    a la machine.
    """
    fidelite = 1.0
    for cle, compte in operations.items():
        nom, _, qubits = cle.partition(":")
        indices = tuple(int(x[2:-1]) for x in qubits.split(","))
        inst = cible[nom].get(indices) if nom in cible.operation_names else None
        erreur = getattr(inst, "error", None) if inst is not None else None
        if erreur:
            fidelite *= (1.0 - erreur) ** compte
    return fidelite


def main() -> int:
    _adoucir_les_flux()

    appareil = backend_reel(APPAREIL)._backend
    cible = appareil.target
    lecture, cz = cartographier(cible)

    # ---- chaines candidates, etalees sur toute la gamme de qualite ----
    voisins = {}
    for a, c in cz:
        voisins.setdefault(a, set()).add(c)
        voisins.setdefault(c, set()).add(a)
    toutes = []
    for milieu, vs in voisins.items():
        ordonnes = sorted(vs)
        for i, a in enumerate(ordonnes):
            for c in ordonnes[i + 1 :]:
                cout = (cz.get((a, milieu), cz.get((milieu, a), 1.0))
                        + cz.get((milieu, c), cz.get((c, milieu), 1.0))
                        + sum(lecture.get(q, 1.0) for q in (a, milieu, c)))
                toutes.append(((a, milieu, c), cout))
    toutes.sort(key=lambda x: x[1])
    # Reparties du meilleur au 60e centile : au-dela, les coupleurs hors
    # service ne mesureraient qu'une panne.
    pas = int(len(toutes) * 0.6) // (CHAINES_VOULUES - 1)
    chaines = [toutes[i * pas][0] for i in range(CHAINES_VOULUES)]

    # ---- 1 et 2 : transpiler et predire, HORS LIGNE ----
    print("=== 1. PREDICTION, avant tout appel a la machine ===")
    predictions = {}
    for chaine in chaines:
        backend = IbmRuntimeBackend(appareil, initial_layout=list(chaine),
                                    optimization_level=0)
        from qiskit import transpile

        physique = transpile(
            IbmRuntimeBackend._vers_qiskit(ghz()), backend=appareil,
            optimization_level=0, seed_transpiler=7, initial_layout=list(chaine),
        )
        operations = IbmRuntimeBackend._operations_physiques(physique)
        predictions[str(list(chaine))] = predire(cible, chaine, operations)

    for cle, f in predictions.items():
        print(f"  {cle:>18} -> {100 * f:.2f} %")

    # ---- 3 : inscrire CHAQUE prediction, puis dater la tete ----
    import pathlib

    pathlib.Path(DOSSIER).mkdir(parents=True, exist_ok=True)
    journal = Journal()

    print()
    print("=== 2. INSCRIPTION DE CHAQUE PREDICTION ===")
    for chaine in chaines:
        sujet = etiquette(chaine)
        journal.append(
            {
                "appareil": APPAREIL,
                "chaine": list(chaine),
                "tirages": TIRAGES,
                "modele": "multiplicatif, operations exactes, calibration publiee",
                "fidelite_predite": predictions[str(list(chaine))],
            },
            label=f"{sujet}.prediction",
            kind="prediction",
        )
    journal.save(DOSSIER)
    print(f"  {len(journal)} predictions inscrites, chainees")
    print("  Aucune ne peut plus etre retiree : la chaine casserait.")

    print()
    print("=== 3. HORODATAGE DE LA TETE ===")
    jeton = stamp(journal.head)
    jeton.save(DOSSIER)
    print(f"  tete : {journal.head[:48]}...")
    print(f"  datee par l'autorite le : {jeton.verify(journal.head).stamped_at}")
    print("  Les tirages n'existent pas encore. Un seul jeton date les cinq.")

    # ---- 4 : SEULEMENT MAINTENANT, executer ----
    print()
    print("=== 4. EXECUTION, apres l'horodatage ===")
    print(f"{'chaine':>18} {'predit':>9} {'observe':>9} {'+/-':>7} {'ecart':>9}")
    print("-" * 56)
    observations = {}
    for chaine in chaines:
        backend = IbmRuntimeBackend(appareil, initial_layout=list(chaine),
                                    optimization_level=0)
        run = capture(ghz(), backend=backend, seed=7, repetitions=TIRAGES)
        sujet = etiquette(chaine)
        record = RunRecord.from_capture(run)
        record.save(f"{DOSSIER}/{sujet}")
        journal.append(record, label=sujet, kind="execution")

        comptes = bitstring_counts(run.samples["m"])
        total = sum(comptes.values())
        f = (comptes.get(0, 0) + comptes.get(7, 0)) / total
        observations[str(list(chaine))] = f
        p = predictions[str(list(chaine))]
        u = (f * (1 - f) / total) ** 0.5
        print(f"{str(list(chaine)):>18} {100 * p:8.2f}% {100 * f:8.2f}% "
              f"{100 * u:6.2f}% {100 * (f - p):+8.2f}")
    journal.save(DOSSIER)

    # ---- 5 : verdict ----
    p = np.array([predictions[k] for k in predictions])
    o = np.array([observations[k] for k in predictions])
    u = np.sqrt(o * (1 - o) / TIRAGES)

    print()
    print("=== 4. LA PREDICTION TENAIT-ELLE ? ===")
    biais = float(np.mean(o - p))
    print(f"  biais moyen        : {100 * biais:+.2f} points")
    print(f"  erreur absolue moy : {100 * float(np.mean(np.abs(o - p))):.2f} points")
    if len(p) > 2:
        r = float(np.corrcoef(p, o)[0, 1])
        print(f"  correlation        : {r:.3f}")
        print(f"  ordre respecte     : "
              f"{'OUI' if list(np.argsort(p)) == list(np.argsort(o)) else 'NON'}")
    ecarts = np.abs(o - p) / u
    print(f"  ecarts en sigma    : {np.array2string(ecarts, precision=1)}")
    print(f"  chaines a moins de 3 sigma : {int((ecarts < 3).sum())} sur {len(p)}")

    print()
    print("=== 5. L'ORDRE EST-IL OPPOSABLE ? ===")
    relu = Journal.load(DOSSIER)
    natures = [f"{e.index}:{e.kind[:4]}" for e in relu.entries]
    print(f"  chaine : {' '.join(natures)}")
    print("  Les cinq predictions precedent les cinq executions, et le")
    print("  chainage l'etablit : aucune ne peut etre glissee apres coup.")

    orphelines = relu.predictions_sans_execution()
    print(f"  predictions sans execution : {len(orphelines)}")
    if orphelines:
        print(f"    {[e.label for e in orphelines]}")
    else:
        print("    aucune : rien n'a ete abandonne en route")

    tete_datee = None
    for entree in relu.entries:
        if entree.kind == "prediction":
            tete_datee = entree.entry_hash
    jeton_relu = Timestamp.load(f"{DOSSIER}/journal.tsr")
    controle = jeton_relu.verify(tete_datee)
    print(f"  jeton lie a la tete des predictions : {controle.bound}")
    print(f"  emis le                             : {controle.stamped_at}")
    print("  Retirer une prediction ratee changerait cette tete, et le jeton")
    print("  ne la couvrirait plus.")

    pathlib.Path(f"{DOSSIER}/observations.json").write_text(
        canonical_json(observations), encoding="utf-8"
    )
    print(f"\ntout est dans {DOSSIER}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
