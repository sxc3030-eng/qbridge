"""Faire exister un message dans un vrai processeur quantique, et le prouver.

CE QUE CE N'EST PAS. Une emission. `ibm_marrakesh` vit dans un refrigerateur a
dilution a 15 millikelvin ; ses impulsions circulent dans des coaxiaux blindes.
Rien n'en sort, et c'est la condition d'existence de la machine — un qubit qui
rayonnerait vers l'exterieur serait un qubit qui decohere.

CE QUE C'EST. Le message est encode en etats de base, prepare sur des qubits
supraconducteurs reels, puis relu. Il aura physiquement existe quelques
microsecondes a 15 millikelvin. L'archive scellee et horodatee le prouve.

Le Voyager Golden Record ne transmet rien non plus. Il persiste. Un message au
futur n'a pas besoin d'etre emis : il a besoin d'etre date et verifiable.

CE QUE LA MACHINE FAIT AU MESSAGE. Elle l'abime. Chaque qubit se trompe environ
0.4 % du temps sur une bonne chaine, donc un message de 400 bits revient
presque toujours avec des erreurs. On mesure les deux :

    un seul tirage  : ce que la machine rend d'un coup
    vote majoritaire: le meme message lu N fois, bit par bit

Le second est un code a repetition — le decodeur du test 1, dans sa forme la
plus simple. C'est ce qui rend le message lisible.
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
from qbridge.journal import Journal
from qbridge.record import RunRecord
from qbridge.timestamp import Timestamp, stamp

APPAREIL = "ibm_marrakesh"
MESSAGE = "nous sommes ici, on a réussi le pont\nsouveraincode.ca"
TIRAGES = 1024
QUBITS_PAR_CIRCUIT = 80
DOSSIER = "runs/message"


def en_bits(texte: str) -> list:
    """UTF-8, bit de poids fort en tete de chaque octet."""
    return [
        (octet >> (7 - i)) & 1
        for octet in texte.encode("utf-8")
        for i in range(8)
    ]


def en_texte(bits) -> str:
    octets = bytearray()
    for debut in range(0, len(bits) - 7, 8):
        valeur = 0
        for bit in bits[debut : debut + 8]:
            valeur = (valeur << 1) | int(bit)
        octets.append(valeur)
    return octets.decode("utf-8", errors="replace")


def meilleurs_qubits(cible, combien: int) -> list:
    """Les qubits les plus fiables en LECTURE.

    Aucune contrainte de connectivite : le message n'utilise que des portes a
    un qubit. On peut donc prendre les meilleurs ou qu'ils soient sur la puce.
    """
    lecture = {}
    for q in range(cible.num_qubits):
        inst = cible["measure"].get((q,))
        if inst is not None and inst.error is not None:
            lecture[q] = inst.error
    classes = sorted(lecture, key=lambda q: lecture[q])
    return classes[:combien], lecture


def circuit_pour(morceau) -> cirq.Circuit:
    q = cirq.LineQubit.range(len(morceau))
    operations = [cirq.X(q[i]) for i, bit in enumerate(morceau) if bit]
    operations.append(cirq.I(q[0]) if not operations else cirq.I(q[0]))
    operations.append(cirq.measure(*q, key="m"))
    return cirq.Circuit(operations)


def main() -> int:
    _adoucir_les_flux()

    bits = en_bits(MESSAGE)
    print("=== LE MESSAGE ===")
    print(f"  {MESSAGE!r}")
    print(f"  {len(MESSAGE.encode('utf-8'))} octets, {len(bits)} bits")

    appareil = backend_reel(APPAREIL)._backend
    cible = appareil.target
    choisis, lecture = meilleurs_qubits(cible, QUBITS_PAR_CIRCUIT)
    print()
    print(f"=== LES QUBITS ===")
    print(f"  {QUBITS_PAR_CIRCUIT} meilleurs sur {cible.num_qubits}, par erreur de lecture")
    print(f"  du meilleur ({lecture[choisis[0]]:.4%}) au dernier retenu "
          f"({lecture[choisis[-1]]:.4%})")

    morceaux = [
        bits[i : i + QUBITS_PAR_CIRCUIT]
        for i in range(0, len(bits), QUBITS_PAR_CIRCUIT)
    ]
    print(f"  {len(morceaux)} circuits de {QUBITS_PAR_CIRCUIT} bits")

    # --- sceller le message AVANT de l'envoyer ---
    empreinte = sha256_of({"message": MESSAGE, "bits": len(bits)})
    print()
    print("=== SCELLEMENT ET HORODATAGE, avant execution ===")
    print(f"  empreinte : {empreinte}")
    jeton = stamp(empreinte)
    print(f"  datee le  : {jeton.verify(empreinte).stamped_at}")

    pathlib.Path(DOSSIER).mkdir(parents=True, exist_ok=True)
    jeton.save(f"{DOSSIER}/message.tsr")

    # --- execution ---
    print()
    print("=== EXECUTION SUR LA MACHINE ===")
    journal = Journal()
    un_tirage, vote = [], []
    for numero, morceau in enumerate(morceaux):
        qubits = choisis[: len(morceau)]
        backend = IbmRuntimeBackend(
            appareil, initial_layout=list(qubits), optimization_level=0
        )
        run = capture(
            circuit_pour(morceau), backend=backend, seed=7, repetitions=TIRAGES
        )
        dossier = f"{DOSSIER}/morceau_{numero}"
        RunRecord.from_capture(run).save(dossier)
        journal.append(RunRecord.load(dossier), label=f"morceau_{numero}",
                       kind="execution")

        tirages = run.samples["m"]
        attendu = np.array(morceau, dtype=np.uint8)
        justes = float((tirages == attendu).mean())
        majoritaire = (tirages.mean(axis=0) > 0.5).astype(np.uint8)
        un_tirage.append(justes)
        vote.extend(majoritaire.tolist())
        exacts = int((majoritaire == attendu).sum())
        print(f"  circuit {numero} : {len(morceau):>3} bits, "
              f"un tirage {100 * justes:6.2f} %, "
              f"vote {exacts}/{len(morceau)} bits justes")

    journal.save(DOSSIER)

    # --- le message revenu ---
    vote = vote[: len(bits)]
    recu = en_texte(vote)
    faux = [i for i, (a, b) in enumerate(zip(bits, vote)) if a != b]

    print()
    print("=== CE QUE LA MACHINE A RENDU ===")
    print(f"  un seul tirage, en moyenne : {100 * float(np.mean(un_tirage)):.2f} % de bits justes")
    print(f"  apres vote sur {TIRAGES} tirages : {len(bits) - len(faux)}/{len(bits)} bits justes")
    print()
    print(f"  envoye : {MESSAGE!r}")
    print(f"  recu   : {recu!r}")
    print()
    if recu == MESSAGE:
        print("  IDENTIQUE. Le message a traverse la machine et en est ressorti")
        print("  intact — grace au vote, pas grace au materiel.")
    else:
        print(f"  {len(faux)} bit(s) perdu(s) malgre le vote, aux positions {faux[:10]}")

    # --- probabilite qu'un seul tirage suffise ---
    p = float(np.mean(un_tirage))
    print()
    print("=== POURQUOI LA REDONDANCE EST NECESSAIRE ===")
    print(f"  fiabilite par bit         : {100 * p:.3f} %")
    print(f"  message entier d'un coup  : {100 * p ** len(bits):.2e} %")
    print(f"  -> sur {len(bits)} bits, un tirage unique ne suffit jamais.")
    print("     C'est le code a repetition qui rend le message lisible.")

    resultat = {
        "message": MESSAGE,
        "bits": len(bits),
        "appareil": APPAREIL,
        "qubits": choisis[:QUBITS_PAR_CIRCUIT],
        "tirages": TIRAGES,
        "fiabilite_par_bit": p,
        "bits_justes_apres_vote": len(bits) - len(faux),
        "message_recu": recu,
        "empreinte_scellee": empreinte,
        "tete_du_journal": journal.head,
    }
    pathlib.Path(f"{DOSSIER}/resultat.json").write_text(
        canonical_json(resultat), encoding="utf-8"
    )

    print()
    print("=== CE QUI EST OPPOSABLE ===")
    relu = Timestamp.load(f"{DOSSIER}/message.tsr")
    controle = relu.verify(empreinte)
    print(f"  message atteste le  : {controle.stamped_at}")
    print(f"  liaison             : {controle.bound}")
    print(f"  {len(journal)} executions chainees, tete {journal.head[:32]}...")
    print()
    print(f"  Ce texte a existe sous forme d'etats quantiques dans {APPAREIL},")
    print("  a 15 millikelvin. L'horodatage precede l'execution ; le journal")
    print("  empeche d'en retirer un morceau.")
    print(f"\ntout est dans {DOSSIER}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
