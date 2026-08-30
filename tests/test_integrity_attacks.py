"""Non-regression sur les failles d'integrite trouvees en revue.

Chaque test reproduit une attaque qui FONCTIONNAIT avant correction. Ce ne sont
pas des tests hypothetiques : chacun a ete verifie comme exploitable.
"""

import json

import cirq
import numpy as np
import pytest

from qbridge import capture, replay
from qbridge.manifest import MANIFEST_SCHEMA_VERSION, Manifest
from qbridge.modes import ExecutionMode, detect_mode
from qbridge.verdict import Verdict, compare_samples


def _bell():
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1))


def _mid():
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit(
        cirq.H(q[0]),
        cirq.CX(q[0], q[1]),
        cirq.measure(q[0], key="a"),
        cirq.X(q[0]) ** 0.5,
        cirq.measure(q[0], q[1], key="b"),
    )
    assert not c.are_all_measurements_terminal()
    return c


# ---------- CRITIQUE 1 : substitution complete du circuit ----------


def test_le_circuit_ne_peut_pas_etre_remplace():
    """AVANT correction : circuit_hash etait calcule a la construction et plus
    jamais verifie. On remplacait circuit_json en entier, le hash semantique
    restait valide (il ne couvre que circuit_hash), et replay certifiait
    BIT_EXACT un circuit totalement different."""
    m = capture(_bell(), backend="qsim", seed=7).manifest
    q0, q1 = cirq.LineQubit.range(2)
    autre = cirq.Circuit(cirq.X(q0), cirq.X(q1))

    substitue = Manifest.from_dict({**m.to_dict(), "circuit_json": cirq.to_json(autre)})

    with pytest.raises(ValueError, match="circuit"):
        substitue.verify_self()
    with pytest.raises(ValueError, match="circuit"):
        replay(substitue)


def test_le_hash_du_circuit_est_verifie_meme_sans_rejeu():
    m = capture(_bell(), backend="qsim", seed=7).manifest
    falsifie = Manifest.from_dict({**m.to_dict(), "circuit_hash": "0" * 64})
    with pytest.raises(ValueError, match="circuit"):
        falsifie.verify_self()


# ---------- CRITIQUE 2 : contrebande par le seau PERFORMANCE ----------


def test_une_option_verrouillee_ne_peut_pas_passer_par_le_seau_performance():
    """AVANT correction : performance_options n'etait pas couvert par le hash
    ET l'emportait dans all_options(). Ecrire cpu_threads dans ce seau en mode
    midcircuit contournait silencieusement le verrou que override_performance
    refuse bruyamment."""
    run = capture(_mid(), backend="qsim", seed=7, repetitions=50, options={"cpu_threads": 1})
    m = run.manifest
    assert m.semantic_options == {"cpu_threads": 1}

    contrebande = Manifest.from_dict(
        {**m.to_dict(), "performance_options": {"cpu_threads": 8}}
    )
    with pytest.raises(ValueError, match="seau|options"):
        contrebande.verify_self()
    with pytest.raises(ValueError, match="seau|options"):
        replay(contrebande)


def test_les_options_semantiques_l_emportent_sur_le_seau_performance():
    """Meme si une option se retrouvait dans les deux seaux, la valeur
    verrouillee doit gagner."""
    run = capture(_mid(), backend="qsim", seed=7, repetitions=50, options={"cpu_threads": 1})
    bricole = Manifest.from_dict(
        {**run.manifest.to_dict(), "performance_options": {"cpu_threads": 8}}
    )
    assert bricole.all_options()["cpu_threads"] == 1


def test_la_voie_bruyante_reste_refusee():
    run = capture(_mid(), backend="qsim", seed=7, repetitions=50, options={"cpu_threads": 1})
    with pytest.raises(ValueError, match="PERFORMANCE"):
        replay(run.manifest, override_performance={"cpu_threads": 8})


# ---------- CRITIQUE 4 : le chi2 dans le regime clairseme ----------


def test_deux_jeux_independants_ne_sont_pas_declares_compatibles():
    """AVANT correction : a 20 qubits avec 200 tirages, presque chaque bitstring
    est unique, chaque categorie contribue 1.0 au chi2, et p tendait vers 0.5
    QUELLES QUE SOIENT les donnees. Deux jeux sans aucun rapport ressortaient
    STATISTICALLY_COMPATIBLE avec p=0.476."""
    rng = np.random.default_rng(0)
    a = {"m": rng.integers(0, 2, size=(200, 20), dtype=np.uint8)}
    b = {"m": rng.integers(0, 2, size=(200, 20), dtype=np.uint8)}

    r = compare_samples(a, b)
    assert r.verdict is Verdict.INDETERMINATE, (
        f"deux jeux independants declares {r.verdict.name} — le chi2 ment "
        "dans le regime clairseme"
    )
    assert "tirages" in r.detail


def test_indetermine_n_est_jamais_accepte_comme_une_reussite():
    """L'ordre de l'enum doit garantir que `verdict <= NUMERICALLY_EQUIVALENT`
    ne laisse jamais passer INDETERMINATE."""
    assert Verdict.INDETERMINATE > Verdict.DIVERGENT
    assert not (Verdict.INDETERMINATE <= Verdict.NUMERICALLY_EQUIVALENT)


def test_le_regime_dense_reste_decidable():
    """La garde ne doit pas rendre l'outil inutile quand il y a assez de
    tirages : 2 categories et 4000 tirages, le chi2 est valide."""
    rng = np.random.default_rng(0)
    a = {"m": rng.integers(0, 2, size=(4000, 1), dtype=np.uint8)}
    b = {"m": rng.integers(0, 2, size=(4000, 1), dtype=np.uint8)}
    assert compare_samples(a, b).verdict is Verdict.STATISTICALLY_COMPATIBLE


def test_des_echantillons_identiques_restent_bit_exact_meme_clairsemes():
    """La garde s'applique APRES le test d'egalite : une archive identique doit
    rester BIT_EXACT quel que soit le nombre de qubits."""
    rng = np.random.default_rng(1)
    s = {"m": rng.integers(0, 2, size=(200, 20), dtype=np.uint8)}
    assert compare_samples(s, {"m": s["m"].copy()}).verdict is Verdict.BIT_EXACT


def test_compare_samples_distingue_le_dtype():
    a = {"m": np.zeros((10, 2), dtype=np.int8)}
    b = {"m": np.zeros((10, 2), dtype=np.uint8)}
    assert compare_samples(a, b).verdict is not Verdict.BIT_EXACT


# ---------- detect_mode : mesures sans repetitions ----------


def test_un_circuit_qui_mesure_sans_repetitions_n_est_pas_du_vecteur_detat():
    """`simulate()` sur un circuit qui mesure echantillonne ces mesures et
    effondre l'etat. Le classer STATE_VECTOR rendait cpu_threads surchargeable
    sur un resultat qui depend de tirages."""
    assert detect_mode(_mid(), repetitions=None) is ExecutionMode.MIDCIRCUIT_SAMPLING
    assert detect_mode(_bell(), repetitions=None) is ExecutionMode.STATE_VECTOR


# ---------- robustesse du manifeste ----------


def test_to_dict_ne_partage_pas_ses_dicts_internes():
    """AVANT correction : to_dict() rendait une copie de surface, donc muter le
    resultat invalidait un manifeste pourtant `frozen`, sans rien signaler."""
    m = capture(_bell(), backend="qsim", seed=7).manifest
    d = m.to_dict()
    d["kernel"]["qsim_instruction_set"] = 999
    assert m.kernel["qsim_instruction_set"] != 999
    m.verify_self()  # ne doit pas lever


def test_from_dict_refuse_une_version_de_schema_inconnue():
    m = capture(_bell(), backend="qsim", seed=7).manifest
    with pytest.raises(ValueError, match="schema"):
        Manifest.from_dict({**m.to_dict(), "schema_version": "99.0"})


def test_from_dict_refuse_un_champ_inconnu():
    m = capture(_bell(), backend="qsim", seed=7).manifest
    with pytest.raises(ValueError, match="inconnus"):
        Manifest.from_dict({**m.to_dict(), "champ_invente": 1})


def test_from_dict_refuse_un_champ_manquant():
    m = capture(_bell(), backend="qsim", seed=7).manifest
    d = m.to_dict()
    del d["seed"]
    with pytest.raises(ValueError, match="absents"):
        Manifest.from_dict(d)


def test_le_mode_expectation_est_rejete_faute_d_implementation():
    """Il est classe dans OPTION_TIERS mais capture/replay ne le produisent
    pas. Un manifeste qui le declare serait execute par le chemin
    d'echantillonnage tout en se voyant appliquer les niveaux d'EXPECTATION."""
    m = capture(_bell(), backend="qsim", seed=7).manifest
    fabrique = Manifest.from_dict({**m.to_dict(), "mode": "expectation"})
    with pytest.raises(NotImplementedError, match="EXPECTATION"):
        fabrique.execution_mode()


def test_le_manifeste_sur_disque_reste_verifiable(tmp_path):
    m = capture(_bell(), backend="qsim", seed=7).manifest
    chemin = tmp_path / "m.json"
    m.save(chemin)
    Manifest.load(chemin).verify_self()

    data = json.loads(chemin.read_text(encoding="utf-8"))
    data["circuit_json"] = cirq.to_json(cirq.Circuit(cirq.X(cirq.LineQubit(0))))
    chemin.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="circuit"):
        Manifest.load(chemin).verify_self()
