import dataclasses

import pytest
import qsimcirq

from qbridge.modes import ExecutionMode
from qbridge.tiers import Tier, known_options, split_options, tier_of


def test_cpu_threads_est_perf_en_mode_vecteur_detat():
    assert tier_of("cpu_threads", ExecutionMode.STATE_VECTOR) is Tier.PERFORMANCE


def test_cpu_threads_est_perf_en_echantillonnage_terminal():
    assert tier_of("cpu_threads", ExecutionMode.TERMINAL_SAMPLING) is Tier.PERFORMANCE


def test_cpu_threads_est_semantique_en_midcircuit():
    # MESURE : 20 qubits, seed fixe, t=1 et t=2 donnent des bitstrings differents.
    assert tier_of("cpu_threads", ExecutionMode.MIDCIRCUIT_SAMPLING) is Tier.SEMANTIC


def test_cpu_threads_est_numerique_en_mode_expectation():
    assert tier_of("cpu_threads", ExecutionMode.EXPECTATION) is Tier.NUMERIC


def test_max_fused_gate_size_est_numerique_dans_tous_les_modes():
    for mode in ExecutionMode:
        assert tier_of("max_fused_gate_size", mode) is Tier.NUMERIC


def test_verbosity_est_perf_dans_tous_les_modes():
    for mode in ExecutionMode:
        assert tier_of("verbosity", mode) is Tier.PERFORMANCE


def test_option_inconnue_leve_une_erreur():
    with pytest.raises(KeyError, match="inconnue"):
        tier_of("option_qui_nexiste_pas", ExecutionMode.STATE_VECTOR)


def test_toutes_les_options_de_la_version_installee_sont_classees():
    champs = {f.name for f in dataclasses.fields(qsimcirq.QSimOptions)}
    manquantes = champs - known_options()
    assert not manquantes, (
        f"Options QSimOptions non classees : {sorted(manquantes)}. "
        "qsimcirq a change de version — revoir la table OPTION_TIERS."
    )


def test_split_options_repartit_par_niveau():
    parts = split_options(
        {"cpu_threads": 8, "max_fused_gate_size": 3, "verbosity": 0},
        ExecutionMode.STATE_VECTOR,
    )
    assert parts[Tier.PERFORMANCE] == {"cpu_threads": 8, "verbosity": 0}
    assert parts[Tier.NUMERIC] == {"max_fused_gate_size": 3}
    assert parts[Tier.SEMANTIC] == {}


def test_split_options_deplace_cpu_threads_selon_le_mode():
    opts = {"cpu_threads": 8}
    perf = split_options(opts, ExecutionMode.STATE_VECTOR)
    mid = split_options(opts, ExecutionMode.MIDCIRCUIT_SAMPLING)
    assert perf[Tier.PERFORMANCE] == {"cpu_threads": 8}
    assert mid[Tier.SEMANTIC] == {"cpu_threads": 8}


def test_split_options_renvoie_les_trois_niveaux_meme_vides():
    parts = split_options({}, ExecutionMode.STATE_VECTOR)
    assert set(parts) == {Tier.SEMANTIC, Tier.NUMERIC, Tier.PERFORMANCE}
