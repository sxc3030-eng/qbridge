"""Une vraie addition, faite par la machine quantique elle-meme.

POURQUOI UNE ADDITION. L'inclinaison de la Terre dans 40 ans se calcule avec
une formule astronomique : un portable la donne exactement en une microseconde.
Un ordinateur quantique n'y gagne rien. Mais on peut lui faire faire un
MORCEAU du calcul avec de vraies portes logiques — pas en rangeant la reponse
dans les qubits pour la relire, comme pour le message, mais en la CALCULANT.

LES DEUX ADDITIONS, tirees du calcul reel de l'obliquite (IAU 2006) : la Terre
perd environ 4.7 secondes d'arc par decennie.

    petite : 5 + 5 = 10    deux decennies, en secondes d'arc arrondies
    vraie  : 47 + 47 = 94  deux decennies, en DIXIEMES de seconde d'arc

LE CIRCUIT. Additionneur a propagation de retenue de Cuccaro (2004) : des
blocs MAJ calculent la retenue de proche en proche, un CNOT la sort, des blocs
UMA remettent tout en place en laissant la somme dans le second registre. Les
portes de Toffoli se decomposent chacune en six CNOT : c'est le prix d'une
addition reversible.

CE QU'ON ATTEND. Au test d'effondrement, un GHZ tombait a moitie juste vers
130 portes cz. Si une addition en demande autant, la machine se trompera plus
souvent qu'une calculatrice des annees 70 — et c'est exactement ce qu'on veut
voir, mesure plutot que suppose.
"""

from __future__ import annotations

import collections
import pathlib

import cirq

from qbridge import capture
from qbridge.backends.ibm_runtime import IbmRuntimeBackend, backend_reel
from qbridge.cli import _adoucir_les_flux
from qbridge.digest import canonical_json, sha256_of
from qbridge.journal import Journal
from qbridge.record import RunRecord
from qbridge.timestamp import stamp

APPAREIL = "ibm_marrakesh"
TIRAGES = 1024
DOSSIER = "runs/addition"
ADDITIONS = [
    ("petite", 5, 5, 3),    # nom, a, b, nombre de bits
    ("vraie", 47, 47, 6),
]


def additionneur(a: int, b: int, n: int) -> cirq.Circuit:
    """Cuccaro : charge a et b, additionne, mesure la somme (b + retenue)."""
    c0 = cirq.NamedQubit("c0")
    A = [cirq.NamedQubit(f"a{i}") for i in range(n)]
    B = [cirq.NamedQubit(f"b{i}") for i in range(n)]
    z = cirq.NamedQubit("z")

    def maj(x, y, w):
        return [cirq.CNOT(w, y), cirq.CNOT(w, x), cirq.TOFFOLI(x, y, w)]

    def uma(x, y, w):
        return [cirq.TOFFOLI(x, y, w), cirq.CNOT(w, x), cirq.CNOT(x, y)]

    ops = []
    ops += [cirq.X(A[i]) for i in range(n) if (a >> i) & 1]
    ops += [cirq.X(B[i]) for i in range(n) if (b >> i) & 1]

    precedents = [c0] + A[:-1]
    for i in range(n):
        ops += maj(precedents[i], B[i], A[i])
    ops.append(cirq.CNOT(A[-1], z))
    for i in reversed(range(n)):
        ops += uma(precedents[i], B[i], A[i])

    # La somme se lit bit de poids faible d'abord : b0 ... b(n-1), puis z.
    ops.append(cirq.measure(*B, z, key="s"))
    return cirq.Circuit(ops)


def lire(ligne) -> int:
    return sum(int(bit) << i for i, bit in enumerate(ligne))


def main() -> int:
    _adoucir_les_flux()
    pathlib.Path(DOSSIER).mkdir(parents=True, exist_ok=True)

    # ---- 1. le circuit est-il juste ? sur simulateur parfait d'abord ----
    print("=== 1. VERIFICATION SUR SIMULATEUR PARFAIT ===")
    circuits = {}
    for nom, a, b, n in ADDITIONS:
        c = additionneur(a, b, n)
        res = cirq.Simulator().run(c, repetitions=20).measurements["s"]
        valeurs = {lire(l) for l in res}
        ok = valeurs == {a + b}
        print(f"  {nom:7} {a} + {b} -> {sorted(valeurs)}  {'JUSTE' if ok else 'FAUX'}")
        if not ok:
            print("  le circuit est faux : on ne depense pas de temps machine.")
            return 1
        circuits[nom] = c

    # ---- 2. combien de portes sur la vraie puce ? ----
    appareil = backend_reel(APPAREIL)._backend
    from qiskit import transpile

    print()
    print("=== 2. COUT SUR LA PUCE, compte hors ligne ===")
    for nom, a, b, n in ADDITIONS:
        t = transpile(IbmRuntimeBackend._vers_qiskit(circuits[nom]),
                      backend=appareil, optimization_level=3, seed_transpiler=7)
        ops = t.count_ops()
        print(f"  {nom:7} {2 * n + 2:>2} qubits, {ops.get('cz', 0):>4} portes cz, "
              f"profondeur {t.depth()}")

    # ---- 3. pre-enregistrer ----
    document = {"appareil": APPAREIL, "tirages": TIRAGES,
                "attendu": {nom: a + b for nom, a, b, _ in ADDITIONS}}
    empreinte = sha256_of(document)
    jeton = stamp(empreinte)
    jeton.save(f"{DOSSIER}/attendu.tsr")
    pathlib.Path(f"{DOSSIER}/attendu.json").write_text(
        canonical_json(document), encoding="utf-8")
    print()
    print(f"=== 3. REPONSES ATTENDUES HORODATEES : "
          f"{jeton.verify(empreinte).stamped_at} ===")

    # ---- 4. executer ----
    print()
    print("=== 4. LA MACHINE ADDITIONNE ===")
    journal = Journal()
    resultats = {}
    for nom, a, b, n in ADDITIONS:
        backend = IbmRuntimeBackend(appareil, optimization_level=3)
        run = capture(circuits[nom], backend=backend, seed=7, repetitions=TIRAGES)
        dossier = f"{DOSSIER}/{nom}"
        RunRecord.from_capture(run).save(dossier)
        journal.append(RunRecord.load(dossier), label=nom, kind="execution")

        compte = collections.Counter(lire(l) for l in run.samples["s"])
        juste = compte.get(a + b, 0) / TIRAGES
        plus_frequent, fois = compte.most_common(1)[0]
        hasard = 1 / 2 ** (n + 1)
        resultats[nom] = {"juste": juste, "plus_frequent": plus_frequent,
                          "top": compte.most_common(5)}
        print()
        print(f"  {nom} : {a} + {b} = ?")
        print(f"    reponse juste ({a + b}) dans {100 * juste:.1f} % des tirages "
              f"(le hasard donnerait {100 * hasard:.1f} %)")
        print(f"    reponse la plus frequente : {plus_frequent} "
              f"({100 * fois / TIRAGES:.1f} %) -> "
              f"{'JUSTE' if plus_frequent == a + b else 'FAUX'}")
        print("    cinq reponses les plus frequentes : "
              + ", ".join(f"{v} ({100 * k / TIRAGES:.1f} %)"
                          for v, k in compte.most_common(5)))

    journal.save(DOSSIER)
    pathlib.Path(f"{DOSSIER}/resultat.json").write_text(
        canonical_json(resultats), encoding="utf-8")
    print(f"\n{len(journal)} executions chainees dans {DOSSIER}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
