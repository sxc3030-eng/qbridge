"""Jusqu'ou une archive prouve-t-elle encore quelque chose ?

LA QUESTION. Le verdict de plausibilite compare ce que l'etat d'appareil scelle
PREDIT a ce que les tirages scelles MONTRENT. Son modele est multiplicatif :
chaque porte reussit independamment, avec l'erreur publiee par la machine. Ce
modele ignore la DECOHERENCE — un circuit long laisse le temps aux qubits de se
degrader, quelles que soient les portes.

Il existe donc une profondeur au-dela de laquelle le modele sous-estime
l'erreur, declare IMPLAUSIBLE une archive honnete, et devient un piege. Ce
fichier la cherche, sur une vraie machine.

LE DESIGN, ET POURQUOI IL A FAILLI NE RIEN MESURER. Un GHZ suivi de N paires
CNOT.CNOT — qui valent l'identite. L'etat ideal reste {000, 111}, donc le
support ne bouge pas et seul le bruit s'accumule.

Mais `optimization_level=1` ANNULE ces paires : le circuit transpile rendait
2 `cz` a toute profondeur. Six executions identiques, et aucune mesure. D'ou
`optimization_level=0`, verifie avant toute soumission :

    paires |  cz | profondeur
         0 |   2 |  12
         4 |  10 |  65
        16 |  34 | 233
        64 | 130 | 905
"""

from __future__ import annotations

import json
import time

import cirq

from qbridge import capture, verify_physical_plausibility
from qbridge.backends.ibm_runtime import backend_reel
from qbridge.calibration import CalibrationSnapshot
from qbridge.cli import _adoucir_les_flux
from qbridge.record import RunRecord
from qbridge.verdict import bitstring_counts

APPAREIL = "ibm_marrakesh"
REPETITIONS = 1024
PAIRES = (0, 4, 16, 32, 64, 128)


def circuit_profond(paires: int) -> cirq.Circuit:
    """GHZ + `paires` paires CNOT.CNOT. L'etat ideal ne change jamais."""
    q = cirq.LineQubit.range(3)
    ops = [cirq.H(q[0]), cirq.CNOT(q[0], q[1]), cirq.CNOT(q[1], q[2])]
    for _ in range(paires):
        ops += [cirq.CNOT(q[0], q[1]), cirq.CNOT(q[0], q[1])]
    ops.append(cirq.measure(*q, key="m"))
    return cirq.Circuit(ops)


def main() -> int:
    _adoucir_les_flux()

    # optimization_level=0 : sans lui le transpileur efface l'experience.
    backend = backend_reel(APPAREIL, optimization_level=0)
    print(f"appareil : {backend.device_name}\n")

    entete = (
        f"{'paires':>7} {'cz':>5} {'prof':>6} {'duree_ns':>10} "
        f"{'predit':>8} {'observe':>8} {'sigma':>7}  verdict"
    )
    print(entete)
    print("-" * len(entete))

    lignes = []
    for paires in PAIRES:
        circuit = circuit_profond(paires)
        debut = time.time()
        run = capture(
            circuit, backend=backend, seed=7, repetitions=REPETITIONS
        )
        duree = time.time() - debut

        dossier = f"runs/profondeur_{paires:03d}"
        RunRecord.from_capture(run).save(dossier)
        record = RunRecord.load(dossier)
        rapport = verify_physical_plausibility(record)

        provenance = json.loads(run.manifest.device_provenance_json)
        n_cz = provenance["gate_counts"].get("cz", 0)

        # Duree physique du circuit, d'apres les durees de porte scellees.
        instantane = CalibrationSnapshot.from_json(run.manifest.calibration_json)
        duree_ns = 0.0
        for cle, compte in provenance["operations"].items():
            params = instantane.gates.get(cle) or {}
            if "gate_length_ns" in params:
                duree_ns += params["gate_length_ns"].value * compte

        comptes = bitstring_counts(run.samples["m"])
        total = sum(comptes.values())
        observe = (comptes.get(0, 0) + comptes.get(7, 0)) / total

        print(
            f"{paires:>7} {n_cz:>5} {provenance['depth']:>6} {duree_ns:>10.0f} "
            f"{100 * (rapport.predicted_fidelity or 0):>7.2f}% "
            f"{100 * observe:>7.2f}% "
            f"{rapport.sigma if rapport.sigma is not None else 0:>7.1f}  "
            f"{rapport.verdict.name}"
        )
        lignes.append(
            {
                "paires": paires,
                "cz": n_cz,
                "depth": provenance["depth"],
                "duree_ns": duree_ns,
                "predit": rapport.predicted_fidelity,
                "observe": observe,
                "sigma": rapport.sigma,
                "verdict": rapport.verdict.name,
                "secondes": duree,
            }
        )

    print()
    print("=== CE QUE LE MODELE IGNORE ===")
    t2_min = min(
        p["t2_us"].value for p in instantane.qubits.values() if "t2_us" in p
    )
    print(f"  T2 le plus court des qubits utilises : {t2_min:.1f} us")
    for ligne in lignes:
        part = ligne["duree_ns"] / 1000.0 / t2_min
        ecart = (ligne["observe"] - (ligne["predit"] or 0)) * 100
        print(
            f"  {ligne['cz']:>4} cz : circuit = {part:5.1%} de T2, "
            f"observe - predit = {ecart:+6.2f} points"
        )

    with open("runs/effondrement.json", "w", encoding="utf-8") as f:
        json.dump(lignes, f, indent=2)
    print("\nmesures dans runs/effondrement.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
