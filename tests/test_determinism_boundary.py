"""Verrouillage des faits empiriques qui fondent OPTION_TIERS.

Chaque test correspond a une ligne du tableau de faits du plan. Un echec ici
signifie que qsim a change de comportement et que la table doit etre revue —
ce n'est PAS le test qui est faux.
"""

import cirq
import numpy as np
import pytest
import qsimcirq

from qbridge.digest import sha256_of_array


@pytest.fixture(scope="module")
def circuit():
    qubits = cirq.GridQubit.rect(4, 5)  # 20 qubits : ParallelFor reellement engage
    return cirq.experiments.random_rotations_between_grid_interaction_layers_circuit(
        qubits=qubits, depth=12, seed=1234
    )


@pytest.fixture(scope="module")
def circuit_midcircuit():
    q = cirq.LineQubit.range(20)
    c = cirq.Circuit([cirq.H(x) for x in q])
    c.append(cirq.measure(*q[:10], key="mid"))
    c.append([cirq.X(x) ** 0.37 for x in q])
    c.append(cirq.measure(*q, key="fin"))
    assert not c.are_all_measurements_terminal()
    return c


def _sv(circuit, **opts):
    sim = qsimcirq.QSimSimulator(qsimcirq.QSimOptions(**opts))
    return sim.simulate(circuit).state_vector()


@pytest.mark.parametrize("threads", [1, 2, 4, 8])
def test_cpu_threads_ne_change_pas_le_vecteur_detat(circuit, threads):
    """FAIT : cpu_threads est neutre pour l'application de portes."""
    ref = sha256_of_array(_sv(circuit, cpu_threads=1, max_fused_gate_size=2))
    got = sha256_of_array(_sv(circuit, cpu_threads=threads, max_fused_gate_size=2))
    assert got == ref, (
        f"cpu_threads={threads} change le vecteur d'etat. OPTION_TIERS classe "
        "cpu_threads en PERFORMANCE pour STATE_VECTOR : ce n'est plus vrai."
    )


def test_cpu_threads_change_les_mesures_intermediaires(circuit_midcircuit):
    """FAIT : t=1 et t>=2 donnent des bitstrings differents. VirtualMeasure lit
    un vecteur de normes partielles dont la longueur EST num_threads."""

    def ech(t):
        sim = qsimcirq.QSimSimulator(qsimcirq.QSimOptions(cpu_threads=t), seed=5)
        return sha256_of_array(
            sim.run(circuit_midcircuit, repetitions=200).measurements["fin"]
        )

    assert ech(1) != ech(4), (
        "cpu_threads n'affecte plus les mesures intermediaires : il pourrait "
        "etre reclasse en PERFORMANCE pour MIDCIRCUIT_SAMPLING."
    )


def test_max_fused_gate_size_change_le_vecteur_detat(circuit):
    """FAIT : la fusion >=3 perturbe l'arrondi. Justifie le niveau NUMERIC."""
    a = _sv(circuit, cpu_threads=1, max_fused_gate_size=2)
    b = _sv(circuit, cpu_threads=1, max_fused_gate_size=4)
    assert sha256_of_array(a) != sha256_of_array(b)


def test_l_ecart_du_a_la_fusion_reste_un_arrondi(circuit):
    """L'ecart doit rester du bruit d'arrondi, pas une erreur de calcul.
    Calibre INFIDELITY_TOLERANCE dans verdict.py."""
    a = _sv(circuit, cpu_threads=1, max_fused_gate_size=2)
    b = _sv(circuit, cpu_threads=1, max_fused_gate_size=4)
    infidelite = abs(1.0 - abs(np.vdot(a, b)) ** 2)
    assert infidelite < 1e-4, f"infidelite {infidelite:.3e} trop grande pour un arrondi"
    assert np.abs(a - b).max() < 1e-6


def test_deux_instances_fraiches_au_meme_seed_concordent():
    """FAIT : le rejeu est possible a condition de construire une instance neuve."""
    q = cirq.LineQubit.range(6)
    c = cirq.Circuit([cirq.H(x) for x in q] + [cirq.measure(*q, key="m")])

    def ech():
        sim = qsimcirq.QSimSimulator(qsimcirq.QSimOptions(), seed=42)
        return sim.run(c, repetitions=64).measurements["m"].tobytes()

    assert ech() == ech()


def test_reutiliser_une_instance_casse_la_reproductibilite():
    """FAIT : _prng avance a chaque appel. Justifie l'instance fraiche dans
    QsimBackend._simulateur()."""
    q = cirq.LineQubit.range(6)
    c = cirq.Circuit([cirq.H(x) for x in q] + [cirq.measure(*q, key="m")])
    sim = qsimcirq.QSimSimulator(qsimcirq.QSimOptions(), seed=42)
    a = sim.run(c, repetitions=64).measurements["m"].tobytes()
    b = sim.run(c, repetitions=64).measurements["m"].tobytes()
    assert a != b, "QSimSimulator n'est plus a etat : la note de qsim.py peut sauter."


def test_le_round_trip_json_de_cirq_est_exact(circuit):
    """FAIT : base de l'attestation par hash."""
    js = cirq.to_json(circuit)
    assert cirq.read_json(json_text=js) == circuit
    assert cirq.to_json(cirq.read_json(json_text=js)) == js


def test_le_vecteur_detat_est_en_complex64(circuit):
    """Justifie la calibration des tolerances sur eps(float32)."""
    assert np.asarray(_sv(circuit, cpu_threads=1)).dtype == np.complex64


def test_le_noyau_simd_est_identifiable():
    """FAIT : la wheel embarque 4 noyaux et choisit par CPUID a l'import."""
    assert qsimcirq.qsim.__name__.startswith("qsimcirq.qsim")
    assert qsimcirq.qsim_decide.detect_instructions() in (0, 1, 2, 3)
