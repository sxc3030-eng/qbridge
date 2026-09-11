"""Un programme qui trie lui-meme ce qui revient de bon.

LE CONSTAT QUI LE MOTIVE. Le message envoye dans la machine est revenu avec
cinq bits faux, tous sur le meme qubit, le 66 :

    ce que la machine DECLARE : 1.39 % d'erreur de lecture
    ce qu'il FAIT             : lit 1 au lieu de 0 dans 75 a 92 % des tirages

Je l'ai trouve A LA MAIN, apres coup, en remarquant que les erreurs tombaient
tous les 80 bits. Un outil ne peut pas compter la-dessus.

CE QUE FAIT CE PROGRAMME.

    1. INTERROGER chaque qubit avec une question dont on connait la reponse :
       tout a 0, puis tout a 1. Deux circuits suffisent pour tous les qubits.
    2. COMPARER ce que chaque qubit a dit a ce qu'il aurait du dire, et a ce
       que la machine declare de lui.
    3. ECARTER ceux qui mentent, et garder les meilleurs selon la MESURE, pas
       selon la declaration.
    4. PRE-ENREGISTRER ce choix et la prediction, horodates.
    5. RENVOYER le message sur les qubits retenus, et compter.

La difference avec le choix d'hier : on ne croit plus la calibration publiee,
on la verifie dans la meme session. C'est le circuit-piege, applique a toute la
puce d'un coup.
"""

from __future__ import annotations

import pathlib

import cirq
import numpy as np

from qbridge import capture
from qbridge.backends.ibm_runtime import IbmRuntimeBackend, backend_reel
from qbridge.cli import _adoucir_les_flux
from qbridge.digest import canonical_json, sha256_of
from qbridge.journal import Journal
from qbridge.record import RunRecord
from qbridge.timestamp import stamp

APPAREIL = "ibm_marrakesh"
MESSAGE = "nous sommes ici, on a réussi le pont\nsouveraincode.ca"
CANDIDATS = 120
RETENUS = 80
TIRAGES = 1024
DOSSIER = "runs/tri"


def en_bits(texte):
    return [(o >> (7 - i)) & 1 for o in texte.encode("utf-8") for i in range(8)]


def en_texte(bits):
    octets = bytearray()
    for d in range(0, len(bits) - 7, 8):
        v = 0
        for b in bits[d : d + 8]:
            v = (v << 1) | int(b)
        octets.append(v)
    return octets.decode("utf-8", errors="replace")


def circuit_de_bits(bits):
    q = cirq.LineQubit.range(len(bits))
    ops = [cirq.X(q[i]) for i, b in enumerate(bits) if b]
    ops.append(cirq.I(q[0]))
    ops.append(cirq.measure(*q, key="m"))
    return cirq.Circuit(ops)


def executer(appareil, bits, qubits, dossier, journal, etiquette):
    backend = IbmRuntimeBackend(appareil, initial_layout=list(qubits),
                                optimization_level=0)
    run = capture(circuit_de_bits(bits), backend=backend, seed=7,
                  repetitions=TIRAGES)
    RunRecord.from_capture(run).save(dossier)
    journal.append(RunRecord.load(dossier), label=etiquette, kind="execution")
    return run.samples["m"]


def main() -> int:
    _adoucir_les_flux()
    pathlib.Path(DOSSIER).mkdir(parents=True, exist_ok=True)
    journal = Journal()

    appareil = backend_reel(APPAREIL)._backend
    cible = appareil.target
    declare = {}
    for q in range(cible.num_qubits):
        inst = cible["measure"].get((q,))
        if inst is not None and inst.error is not None:
            declare[q] = inst.error
    candidats = sorted(declare, key=declare.get)[:CANDIDATS]

    # ---- 1. interroger chaque qubit ----
    print(f"=== 1. INTERROGATION de {CANDIDATS} qubits ===")
    zeros = executer(appareil, [0] * CANDIDATS, candidats,
                     f"{DOSSIER}/tout_a_0", journal, "tout_a_0")
    uns = executer(appareil, [1] * CANDIDATS, candidats,
                   f"{DOSSIER}/tout_a_1", journal, "tout_a_1")

    err0 = zeros.mean(axis=0)          # lu 1 alors que c'etait 0
    err1 = 1.0 - uns.mean(axis=0)      # lu 0 alors que c'etait 1
    mesuree = (err0 + err1) / 2

    # ---- 2. comparer a la declaration ----
    print()
    print("=== 2. DECLARE contre MESURE ===")
    menteurs = []
    for i, q in enumerate(candidats):
        d = declare[q]
        m = float(mesuree[i])
        sigma = (max(m, 1e-4) * (1 - m) / (2 * TIRAGES)) ** 0.5
        if m > 0.02 and (m - d) > 5 * sigma:
            menteurs.append((q, d, m))
    menteurs.sort(key=lambda x: -x[2])
    print(f"  {len(menteurs)} qubit(s) sur {CANDIDATS} font nettement pire que "
          "ce qu'ils declarent :")
    for q, d, m in menteurs[:12]:
        print(f"    q{q:<4} declare {d:7.3%}   mesure {m:7.3%}   "
              f"x{m / max(d, 1e-6):.0f}")
    if len(menteurs) > 12:
        print(f"    ... et {len(menteurs) - 12} autre(s)")
    ratio = np.median(mesuree / np.array([declare[q] for q in candidats]))
    print(f"  rapport median mesure/declare : {ratio:.2f}")

    # ---- 3. retenir par la MESURE ----
    ordre = np.argsort(mesuree)
    retenus = [candidats[i] for i in ordre[:RETENUS]]
    ecartes_dans_ancien = [
        q for q in sorted(declare, key=declare.get)[:RETENUS] if q not in retenus
    ]
    print()
    print(f"=== 3. SELECTION des {RETENUS} meilleurs par la MESURE ===")
    print(f"  pire erreur mesuree retenue : {float(mesuree[ordre[RETENUS - 1]]):.3%}")
    print(f"  qubits que le choix PAR DECLARATION aurait pris et que la mesure "
          f"ecarte : {len(ecartes_dans_ancien)}")
    print(f"    {ecartes_dans_ancien[:15]}")
    print(f"  le qubit 66 est-il retenu ? {'OUI' if 66 in retenus else 'non'}")

    # ---- 4. pre-enregistrer ----
    bits = en_bits(MESSAGE)
    p_bit = 1 - float(np.mean(mesuree[ordre[:RETENUS]]))
    document = {
        "appareil": APPAREIL,
        "message": MESSAGE,
        "qubits_retenus": retenus,
        "fiabilite_par_bit_predite": p_bit,
        "prediction": "tous les bits justes apres vote majoritaire",
    }
    empreinte = sha256_of(document)
    jeton = stamp(empreinte)
    jeton.save(f"{DOSSIER}/selection.tsr")
    pathlib.Path(f"{DOSSIER}/selection.json").write_text(
        canonical_json(document), encoding="utf-8")
    print()
    print("=== 4. SELECTION ET PREDICTION HORODATEES, avant l'envoi ===")
    print(f"  fiabilite par bit predite : {100 * p_bit:.3f} %")
    print(f"  datee le : {jeton.verify(empreinte).stamped_at}")

    # ---- 5. renvoyer le message ----
    print()
    print("=== 5. LE MESSAGE, sur les qubits qui ont dit vrai ===")
    vote, justes_un_tirage = [], []
    for n, debut in enumerate(range(0, len(bits), RETENUS)):
        morceau = bits[debut : debut + RETENUS]
        tirages = executer(appareil, morceau, retenus[: len(morceau)],
                           f"{DOSSIER}/message_{n}", journal, f"message_{n}")
        attendu = np.array(morceau, dtype=np.uint8)
        justes_un_tirage.append(float((tirages == attendu).mean()))
        vote.extend((tirages.mean(axis=0) > 0.5).astype(int).tolist())

    journal.save(DOSSIER)
    vote = vote[: len(bits)]
    faux = [i for i, (a, b) in enumerate(zip(bits, vote)) if a != b]
    recu = en_texte(vote)

    print(f"  un tirage   : {100 * float(np.mean(justes_un_tirage)):.2f} % de bits justes")
    print(f"  apres vote  : {len(bits) - len(faux)}/{len(bits)} bits justes")
    print(f"  recu : {recu!r}")
    print()
    print("  hier, qubits choisis sur DECLARATION : 427/432, qubit 66 inclus")
    print(f"  aujourd'hui, choisis sur MESURE      : {len(bits) - len(faux)}/{len(bits)}")
    print(f"\n  {len(journal)} executions chainees dans {DOSSIER}/journal.json")

    pathlib.Path(f"{DOSSIER}/resultat.json").write_text(canonical_json({
        **document,
        "menteurs": [{"qubit": q, "declare": d, "mesure": m} for q, d, m in menteurs],
        "bits_justes": len(bits) - len(faux),
        "message_recu": recu,
        "tete_journal": journal.head,
    }), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
