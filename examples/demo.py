"""Demonstration : capturer, sceller sur disque, rejouer, comparer."""

from pathlib import Path

import cirq

from qbridge import capture, replay, replay_record, verify_archival
from qbridge.manifest import Manifest
from qbridge.record import RunRecord

SORTIE = Path(__file__).resolve().parent.parent / "runs"
SORTIE.mkdir(exist_ok=True)


def bandeau(titre: str) -> None:
    print(f"\n{'=' * 70}\n{titre}\n{'=' * 70}")


def main() -> None:
    qubits = cirq.GridQubit.rect(3, 4)
    circuit = cirq.experiments.random_rotations_between_grid_interaction_layers_circuit(
        qubits=qubits, depth=12, seed=2026
    )

    bandeau("1. Capture sur qsim — 8 threads, fusion 2, mode vecteur d'etat")
    run = capture(
        circuit,
        backend="qsim",
        seed=7,
        options={"cpu_threads": 8, "max_fused_gate_size": 2},
    )
    chemin = SORTIE / "demo_manifest.json"
    run.manifest.save(chemin)
    print(f"  mode               : {run.manifest.mode}")
    print(f"  noyau charge       : {run.manifest.kernel['qsim_kernel_module']}")
    print(f"  options PERFORMANCE: {run.manifest.performance_options}")
    print(f"  options NUMERIC    : {run.manifest.numeric_options}")
    print(f"  options SEMANTIC   : {run.manifest.semantic_options}")
    print(f"  hash semantique    : {run.manifest.semantic_hash[:40]}...")
    print(f"  hash resultat      : {run.result_hash[:40]}...")
    print(f"  manifeste          : {chemin.name} ({chemin.stat().st_size} octets)")

    bandeau("2. Rejeu a l'identique depuis le fichier")
    r = replay(Manifest.load(chemin))
    print(f"  verdict : {r.verdict.name} — {r.detail}")

    bandeau("3. Rejeu avec 1 seul thread (PERFORMANCE dans ce mode)")
    r = replay(Manifest.load(chemin), override_performance={"cpu_threads": 1})
    print(f"  verdict : {r.verdict.name} — {r.detail}")
    print("  -> le nombre de coeurs de la machine de rejeu n'a aucune importance")

    bandeau("4. Rejeu sur l'oracle Cirq — moteur entierement different")
    r = replay(Manifest.load(chemin), backend="cirq-reference")
    print(f"  verdict      : {r.verdict.name} — {r.detail}")
    print(f"  noyau change : {r.kernel_changed}")

    bandeau("5. Mesures intermediaires : les threads deviennent SEMANTIQUES")
    q = cirq.LineQubit.range(8)
    mid = cirq.Circuit([cirq.H(x) for x in q])
    mid.append(cirq.measure(*q[:4], key="mid"))
    mid.append([cirq.X(x) ** 0.37 for x in q])
    mid.append(cirq.measure(*q, key="fin"))
    assert not mid.are_all_measurements_terminal()

    run_mid = capture(
        mid, backend="qsim", seed=7, repetitions=200, options={"cpu_threads": 4}
    )
    print(f"  mode              : {run_mid.manifest.mode}")
    print(f"  options SEMANTIC  : {run_mid.manifest.semantic_options}")
    print(f"  verdict du rejeu  : {replay(run_mid.manifest).verdict.name}")
    try:
        replay(run_mid.manifest, override_performance={"cpu_threads": 1})
        print("  !! la surcharge aurait du etre refusee")
    except ValueError as e:
        print(f"  surcharge refusee : {e}")

    bandeau("6. Detection d'une alteration du manifeste (hash NON signe)")
    import json

    data = json.loads(chemin.read_text(encoding="utf-8"))
    data["seed"] = 999
    falsifie = SORTIE / "demo_manifest_falsifie.json"
    falsifie.write_text(json.dumps(data), encoding="utf-8")
    try:
        replay(Manifest.load(falsifie))
        print("  !! l'alteration n'a pas ete detectee")
    except ValueError as e:
        print(f"  refuse : {e}")
    print("  NB : le hash n'est pas signe. Il detecte une corruption, pas un")
    print("       adversaire — celui-ci recalculerait le hash apres coup.")

    bandeau("7. Archive complete : la recette PLUS les tirages bruts")
    run_arch = capture(
        cirq.Circuit(
            cirq.H(cirq.LineQubit(0)),
            cirq.CX(cirq.LineQubit(0), cirq.LineQubit(1)),
            cirq.measure(cirq.LineQubit(0), cirq.LineQubit(1), key="m"),
        ),
        backend="qsim",
        seed=7,
        repetitions=500,
    )
    dossier = SORTIE / "archive_bell"
    record = RunRecord.from_capture(run_arch)
    record.save(dossier)
    poids = sum(f.stat().st_size for f in dossier.iterdir())
    print(f"  dossier      : {dossier.name}/ ({poids} octets)")
    print(f"  fichiers     : {sorted(f.name for f in dossier.iterdir())}")
    print(f"  comptages    : {record.bitstring_counts('m')}")
    print("  -> les comptages sont DERIVES des tirages, jamais stockes a cote")

    bandeau("8. Verification archivistique — ZERO ressource quantique")
    rapport = verify_archival(RunRecord.load(dossier))
    print(f"  manifeste intact : {rapport.manifest_intact}")
    print(f"  resultats intacts: {rapport.results_intact}")
    print(f"  detail           : {rapport.detail}")
    print("  -> aucun simulateur n'a tourne : c'est la garantie qui survit 10 ans")

    bandeau("9. Rejeu compare a l'ARCHIVE, pas a une re-execution")
    r = replay_record(RunRecord.load(dossier))
    print(f"  verdict : {r.verdict.name} — {r.detail}")
    print("  -> la reference vient du disque, scellee avant : pas de tautologie")

    bandeau("10. Un seul bit de tirage falsifie")
    corrompu = RunRecord.load(dossier)
    corrompu.samples["m"][0][0] ^= 1
    rapport = verify_archival(corrompu)
    print(f"  resultats intacts : {rapport.results_intact}")
    print(f"  detail            : {rapport.detail[:110]}...")


if __name__ == "__main__":
    main()
