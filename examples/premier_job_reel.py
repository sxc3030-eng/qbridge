"""Le premier vrai job sur une vraie machine quantique.

C'EST LE SEUL CHEMIN QUE CE PROJET N'AVAIT JAMAIS EXERCE. Tout le reste est
teste hors ligne contre un backend factice : conversion, transpilation,
depaquetage, manifeste, verdicts. Ce qui n'avait jamais tourne, c'est la
soumission reelle — reseau, file d'attente, recuperation.

CE QU'ON ATTEND, ET CE QU'ON N'ATTEND PAS. Un GHZ a trois qubits devrait sortir
majoritairement 000 et 111. Sur une VRAIE machine, la proportion sera nettement
inferieure a 100 % : le bruit produit les six autres etats. Cet ecart n'est pas
un defaut, c'est la mesure elle-meme — et c'est precisement pour cela que le
verdict materiel est plafonne a « statistiquement compatible ».
"""

from __future__ import annotations

import time

import cirq

from qbridge import capture, verify_archival
from qbridge.cli import _adoucir_les_flux
from qbridge.backends.ibm_runtime import backend_reel
from qbridge.record import RunRecord
from qbridge.verdict import bitstring_counts

APPAREIL = "ibm_marrakesh"
REPETITIONS = 1024
DOSSIER = "runs/premier_job_reel"


def main() -> int:
    # Le diagramme de cirq est trace avec des caracteres Unicode qu'une console
    # Windows en cp1252 ne sait pas encoder. Sans cela, l'affichage du circuit
    # fait echouer le script AVANT meme la soumission — deja corrige dans la
    # CLI, jamais reporte ici.
    _adoucir_les_flux()

    q = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        [
            cirq.H(q[0]),
            cirq.CNOT(q[0], q[1]),
            cirq.CNOT(q[1], q[2]),
            cirq.measure(*q, key="m"),
        ]
    )
    print("circuit GHZ a 3 qubits :")
    print(circuit)

    # `backend_reel` verifie le plan AVANT de rendre l'appareil : sur un plan
    # payant sans accord explicite, il leve plutot que de laisser soumettre.
    print(f"\nouverture de {APPAREIL} (le plan est verifie ici)...")
    backend = backend_reel(APPAREIL)
    print(f"  appareil : {backend.device_name}")

    print(f"\nsoumission de {REPETITIONS} tirages. File d'attente possible...")
    debut = time.time()
    run = capture(circuit, backend=backend, seed=7, repetitions=REPETITIONS)
    duree = time.time() - debut
    print(f"  termine en {duree:.1f} s")

    trace = backend.derniere_transpilation
    print("\n=== provenance de la transpilation ===")
    print(f"  placement logique -> physique : {trace['initial_layout']}")
    print(f"  profondeur : {trace['depth']}")
    print(f"  portes     : {trace['gate_counts']}")

    comptes = bitstring_counts(run.samples["m"])
    total = sum(comptes.values())
    print(f"\n=== resultats ({total} tirages) ===")
    for etat in range(8):
        n = comptes.get(etat, 0)
        barre = "#" * round(60 * n / total)
        print(f"  {etat:03b}  {n:5}  {100 * n / total:5.1f}%  {barre}")

    fidelite = (comptes.get(0b000, 0) + comptes.get(0b111, 0)) / total
    print(f"\n  000 + 111 = {100 * fidelite:.1f}%")
    print("  (100 % sur un simulateur parfait ; l'ecart EST le bruit reel)")

    from qbridge.calibration import CalibrationSnapshot
    import json

    print()
    print('=== ETAT DE L APPAREIL SCELLE DANS L ARCHIVE ===')
    if run.manifest.calibration_json is None:
        print('  AUCUN — l archive ne pourra rien prouver sur le resultat.')
    else:
        cal = CalibrationSnapshot.from_json(run.manifest.calibration_json)
        print(f'  appareil : {cal.device_id} v{cal.device_version}')
        print(f'  etalement des mesures : {cal.temporal_spread_seconds()/3600:.1f} h')
        for cle in sorted(cal.qubits):
            d = cal.qubits[cle]
            print(f'    {cle}: T1={d["t1_us"].value:7.1f} us  '
                  f'T2={d["t2_us"].value:7.1f} us  '
                  f'lecture={d["readout_error"].value:.3e}')
        print(f'  portes scellees : {len(cal.gates)}')
    print(f'  noyau qsim scelle : {run.manifest.kernel} (vide = qsim absent)')
    prov = json.loads(run.manifest.device_provenance_json or 'null')
    print(f'  placement scelle  : {prov and prov["initial_layout"]}')

    record = RunRecord.from_capture(run)
    record.save(DOSSIER)
    print(f"\narchive scellee dans {DOSSIER}/")

    relu = RunRecord.load(DOSSIER)
    relu.verify_integrity()
    rapport = verify_archival(relu)
    print("\n=== verification archivistique (zero ressource quantique) ===")
    print(f"  manifest_intact : {rapport.manifest_intact}")
    print(f"  results_intact  : {rapport.results_intact}")
    print(f"  total_shots     : {rapport.total_shots}")

    print(f"\nbackend scelle : {relu.manifest.backend_name}")
    print(f"version scellee : {relu.manifest.backend_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
