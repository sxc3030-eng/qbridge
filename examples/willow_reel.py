"""Cycle qbridge complet sur la calibration REELLE de Willow."""

import pathlib
import tempfile

import cirq
from cirq_google.engine import create_device_from_processor_id

from qbridge import capture, replay, verify_archival
from qbridge.providers import from_google_calibration
from qbridge.record import RunRecord
from qbridge.signing import (
    Ed25519Signer,
    Ed25519Verifier,
    sign_record,
    verify_record_signature,
)
from qbridge.verdict import bitstring_counts

d = pathlib.Path(tempfile.mkdtemp())

snap, avert = from_google_calibration("willow_pink")
device = create_device_from_processor_id("willow_pink")

# chaine de 5 qubits reellement connectes sur la puce
graphe = device.metadata.nx_graph
depart = sorted(device.metadata.qubit_set)[0]
chaine = [depart]
while len(chaine) < 5:
    voisins = [v for v in graphe.neighbors(chaine[-1]) if v not in chaine]
    if not voisins:
        break
    chaine.append(sorted(voisins)[0])

print("=" * 74)
print("CYCLE COMPLET SUR CALIBRATION REELLE — google:willow_pink")
print("=" * 74)
print(f"  calibration du : {snap.device_version}")
print(f"  chaine physique: {chaine}")
print(f"  T1 de la chaine: {[round(snap.qubit_param(q, 't1_us'), 1) for q in chaine]}")

ghz = cirq.Circuit([cirq.H(chaine[0])])
for a, b in zip(chaine, chaine[1:]):
    ghz.append(cirq.CNOT(a, b))
ghz.append(cirq.measure(*chaine, key="m"))

print("\n1. CAPTURE sur hardware-sim, bruit derive de la calibration reelle")
run = capture(ghz, backend="hardware-sim", seed=7, repetitions=2000, calibration=snap)
m = run.manifest
print(f"   mode            : {m.mode}")
print(f"   backend         : {m.backend_name} {m.backend_version}")
print(f"   hash semantique : {m.semantic_hash[:32]}...")
print(f"   manifeste       : {len(m.circuit_json) + len(m.calibration_json):,} octets")

comptes = bitstring_counts(run.samples["m"])
n = len(chaine)
parfait = (0, (1 << n) - 1)
fidelite = sum(comptes.get(k, 0) for k in parfait) / sum(comptes.values())
print(f"   bitstrings vus  : {len(comptes)} sur {1 << n} possibles")
print(f"   |00000>+|11111> : {fidelite:.1%} des tirages")

print("\n2. ARCHIVE + SIGNATURE ed25519")
record = RunRecord.from_capture(run)
dossier = d / "willow"
record.save(dossier)
signeur, _priv, publique = Ed25519Signer.generate("simon")
signature = sign_record(record, signeur)
signature.save(dossier / "signature.json")
poids = sum(f.stat().st_size for f in dossier.iterdir())
print(f"   dossier         : {poids:,} octets")
print(f"   portee signature: {signature.scope}")

print("\n3. VERIFICATION ARCHIVISTIQUE — zero ressource quantique")
relu = RunRecord.load(dossier)
rap = verify_archival(relu)
print(f"   manifeste intact: {rap.manifest_intact}")
print(f"   tirages intacts : {rap.results_intact} ({rap.total_shots} tirages)")
sig = verify_record_signature(relu, signature, Ed25519Verifier(publique, "simon"))
print(f"   signature       : {sig.valid} | opposable : {sig.third_party_verifiable}")

print("\n4. REJEU")
r = replay(m)
print(f"   verdict         : {r.verdict.name}")
print(f"   detail          : {r.detail[:88]}")

print("\n5. LE PLAFOND TIENT-IL SUR DU MATERIEL REEL ?")
r2 = replay(m, backend="qsim")
print(f"   rejeu sur qsim  : {r2.verdict.name}  (jamais BIT_EXACT)")

print("\n" + "=" * 74)
print("AVERTISSEMENTS SCELLES AVEC L'INSTANTANE")
print("=" * 74)
for a in avert:
    print(f"  - {a}")
