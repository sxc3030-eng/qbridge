# qbridge — Harnais de capture/replay déterministe — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construire un harnais qui scelle une exécution de circuit quantique dans un manifeste autonome, et sait la rejouer plus tard sur une autre machine ou un autre backend en rendant un verdict de conformité gradué et justifié.

**Architecture:** Un manifeste JSON canonique. `capture()` exécute et scelle ; `replay()` relit, ré-exécute via une interface `Backend` abstraite, et compare selon une échelle de verdicts (BIT_EXACT → NUMERICALLY_EQUIVALENT → STATISTICALLY_COMPATIBLE → DIVERGENT). Le classement d'une option en « influence le résultat » ou non dépend du **mode d'exécution** : c'est un fait mesuré, verrouillé par des tests de frontière qui échoueront si qsim change.

**Tech Stack:** Python 3.13, cirq 1.7.0, qsimcirq 0.22.0, numpy, pytest. Sérialisation `cirq.to_json`. Hash SHA-256 sur JSON canonique. Aucune dépendance à scipy.

---

## Faits établis (mesurés le 2026-08-30 — machine 20 cœurs, AVX2, OpenMP actif)

Toute la conception repose sur ces mesures. Elles ont été obtenues sur `qsimcirq.qsim_avx2`, `detect_instructions()==1`, `detect_gpu()==10` (pas de GPU).

### Ce qui ne change JAMAIS le résultat

| Fait | Mesure |
|---|---|
| `cpu_threads` 1→16 en mode vecteur d'état | bit-pour-bit identique jusqu'à **25 qubits** |
| `cpu_threads` 1→16 en échantillonnage **terminal** | bitstrings identiques |
| `cpu_threads` 1→16 sur valeurs moyennes (20 q) | valeur `float` strictement identique |
| `verbosity` | journalisation seulement |

OpenMP est bien actif : ×1.86 de gain à 25 qubits entre 1 et 8 threads. La neutralité mesurée n'est donc pas l'artefact d'un threading inerte.

### Ce qui CHANGE le résultat

| Fait | Mesure | Niveau |
|---|---|---|
| **`cpu_threads` avec mesures intermédiaires** | **20 q : `t=1` → `747f4099…` ; `t≥2` → `d38f4a9c…`** | **critique** |
| `max_fused_gate_size` ≥ 3 | vecteur différent, max\|Δ\|≈3.2e-9, infidélité ≈1.4e-5 | arrondi |
| `use_gpu` / `gpu_mode` | noyaux CUDA entièrement distincts | non testé ici |
| `ev_noisy_repetitions` | nombre de trajectoires moyennées | sémantique |
| `denormals_are_zeros` | FTZ/DAZ ; sans effet sur les circuits testés | prudence |
| **module SIMD** (AVX512/AVX2/SSE/basic) | choisi par CPUID **à l'import**, pas figé dans la wheel | critique |

**La cause de la ligne critique :** `statespace.h::VirtualMeasure` appelle `PartialNorms`, dont le vecteur retourné a **exactement `num_threads` éléments**. Le tirage parcourt ce vecteur (`while (r > partial_norms[m]) ++m;`). Le nombre de threads fait donc partie de l'algorithme de mesure, pas seulement de sa vitesse.

### Autres faits vérifiés

| Fait | Conséquence |
|---|---|
| `QSimSimulator` est **à état** : `_prng` avance à chaque appel | il faut construire une instance **fraîche** par exécution |
| 2 instances fraîches, même seed → résultats identiques | le rejeu est possible |
| `seed=None` → non reproductible | `capture()` doit refuser un seed nul |
| `repetitions == 1` bascule sur un autre chemin C++ | à consigner dans le manifeste |
| round-trip `cirq.to_json`/`read_json` | exact, hash re-sérialisé stable |
| `cirq.to_json(QSimCircuit)` | **lève `ValueError`** → sérialiser le `cirq.Circuit` nu |
| `StateVectorTrialResult` | **non sérialisable** en JSON par Cirq |
| vecteur d'état | `complex64` → tolérances calibrées sur eps(float32)=1.19e-7 |

### API vérifiée

```python
qsimcirq.QSimOptions(max_fused_gate_size=2, cpu_threads=1, ev_noisy_repetitions=1,
                     use_gpu=False, gpu_mode=0, gpu_state_threads=512,
                     gpu_data_blocks=16, verbosity=0, denormals_are_zeros=False)
qsimcirq.QSimSimulator(qsim_options=None, seed=None, noise=None, circuit_memoization_size=0)
qsimcirq.qsim.__name__                      # 'qsimcirq.qsim_avx2'  <- noyau reellement charge
qsimcirq.qsim_decide.detect_instructions()  # 0=AVX512F 1=AVX2 2=SSE4.1 3=BASIC
qsimcirq.qsim_decide.detect_gpu()           # 10 = pas de GPU
```

**Attention version :** qsimcirq 0.22.x expose **9** champs `QSimOptions` ; la branche `main` (0.23.0.dev0) en expose **11** (`gpu_cusvex_log_buf_size`, `gpu_cusvex_network_type`). Le code doit énumérer `dataclasses.fields(QSimOptions)` à l'exécution, jamais coder la liste en dur.

---

## Le modèle central : le niveau dépend du mode

Une option n'a pas un niveau absolu. Elle en a un **par mode d'exécution**.

| Mode | Déclencheur | `cpu_threads` |
|---|---|---|
| `STATE_VECTOR` | `repetitions is None` | PERFORMANCE ✅ mesuré |
| `TERMINAL_SAMPLING` | `run()`, `are_all_measurements_terminal()` vrai, `repetitions > 1` | PERFORMANCE ✅ mesuré |
| `MIDCIRCUIT_SAMPLING` | `run()`, mesures non terminales, **ou `repetitions == 1`** | **SEMANTIC** ⚠️ mesuré différent |
| `EXPECTATION` | `simulate_expectation_values()` | NUMERIC (prudence : le source partitionne par threads même si stable à 20 q) |

C'est ce tableau que `tiers.py` encode, et que `tests/test_determinism_boundary.py` re-mesure à chaque exécution.

---

## Structure des fichiers

| Fichier | Responsabilité unique |
|---|---|
| `pyproject.toml` | paquet + config pytest |
| `src/qbridge/digest.py` | JSON canonique + SHA-256. Aucune notion de quantique. |
| `src/qbridge/modes.py` | `ExecutionMode` et sa détection depuis un circuit |
| `src/qbridge/tiers.py` | table (option × mode) → niveau |
| `src/qbridge/fingerprint.py` | empreinte machine + **noyau SIMD réellement chargé** |
| `src/qbridge/manifest.py` | `Manifest` + sérialisation + hash sémantique |
| `src/qbridge/backends/base.py` | protocole `Backend` — la frontière destinée à durer |
| `src/qbridge/backends/cirq_ref.py` | oracle indépendant `cirq.Simulator` |
| `src/qbridge/backends/qsim.py` | backend qsim, instance fraîche par appel |
| `src/qbridge/capture.py` | `capture()` |
| `src/qbridge/verdict.py` | comparaisons et verdicts, χ² sans scipy |
| `src/qbridge/replay.py` | `replay()` |
| `src/qbridge/__init__.py` | API publique |

---

## Task 1: Squelette du paquet

**Files:** Create `pyproject.toml`, `src/qbridge/__init__.py`, `tests/test_smoke.py`

- [ ] **Step 1: Écrire le test qui échoue**

```python
# tests/test_smoke.py
def test_package_importable():
    import qbridge
    assert qbridge.__version__ == "0.1.0"
```

- [ ] **Step 2: Vérifier l'échec**

Run: `.venv/Scripts/python.exe -m pytest tests/test_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qbridge'`

- [ ] **Step 3: Créer le paquet**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "qbridge"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["cirq>=1.7", "qsimcirq>=0.22", "numpy>=1.26"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

```python
# src/qbridge/__init__.py
__version__ = "0.1.0"
```

- [ ] **Step 4: Vérifier que ça passe**

Run: `.venv/Scripts/python.exe -m pytest tests/test_smoke.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/qbridge/__init__.py tests/test_smoke.py .gitignore && git commit -m "feat: squelette du paquet qbridge"
```

---

## Task 2: Hash canonique (`digest.py`)

**Files:** Create `src/qbridge/digest.py`, Test `tests/test_digest.py`

- [ ] **Step 1: Écrire les tests qui échouent**

```python
# tests/test_digest.py
import numpy as np
import pytest
from qbridge.digest import canonical_json, sha256_of, sha256_of_text, sha256_of_array

def test_canonical_json_insensible_a_l_ordre_des_cles():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})

def test_canonical_json_compact_et_trie():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'

def test_sha256_stable():
    assert sha256_of({"a": 1}) == sha256_of({"a": 1})

def test_sha256_distingue():
    assert sha256_of({"a": 1}) != sha256_of({"a": 2})

def test_sha256_renvoie_64_hex():
    h = sha256_of({"a": 1})
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)

def test_floats_non_finis_refuses():
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})

def test_sha256_of_array_distingue_le_dtype():
    a = np.array([1, 0], dtype=np.complex64)
    b = np.array([1, 0], dtype=np.complex128)
    assert sha256_of_array(a) != sha256_of_array(b)

def test_sha256_of_array_distingue_la_forme():
    a = np.zeros((2, 2), dtype=np.uint8)
    b = np.zeros((4,), dtype=np.uint8)
    assert sha256_of_array(a) != sha256_of_array(b)

def test_sha256_of_text_est_stable():
    assert sha256_of_text("abc") == sha256_of_text("abc")
```

- [ ] **Step 2: Vérifier l'échec**

Run: `.venv/Scripts/python.exe -m pytest tests/test_digest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qbridge.digest'`

- [ ] **Step 3: Implémenter**

```python
# src/qbridge/digest.py
"""JSON canonique et empreintes SHA-256. Aucune dependance au domaine quantique."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def canonical_json(obj: Any) -> str:
    """Serialise en JSON deterministe : cles triees, sans espaces, ASCII.

    `allow_nan=False` fait echouer NaN/Infinity, qui ne sont pas du JSON valide
    et casseraient la stabilite du hash entre implementations.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def sha256_of(obj: Any) -> str:
    """SHA-256 hexadecimal de la forme canonique de `obj`."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_of_text(text: str) -> str:
    """SHA-256 d'une chaine deja serialisee (ex. sortie de cirq.to_json)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_of_array(array: np.ndarray) -> str:
    """SHA-256 des octets bruts d'un tableau, dtype et forme inclus.

    On hache les octets et non une representation texte : c'est la seule facon
    de detecter une difference bit-pour-bit dans un vecteur d'etat complex64.
    """
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("utf-8"))
    h.update(str(a.shape).encode("utf-8"))
    h.update(a.tobytes())
    return h.hexdigest()
```

- [ ] **Step 4: Vérifier que ça passe**

Run: `.venv/Scripts/python.exe -m pytest tests/test_digest.py -v` → PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qbridge/digest.py tests/test_digest.py && git commit -m "feat: hash canonique deterministe"
```

---

## Task 3: Modes d'exécution (`modes.py`)

**Files:** Create `src/qbridge/modes.py`, Test `tests/test_modes.py`

- [ ] **Step 1: Écrire les tests qui échouent**

```python
# tests/test_modes.py
import cirq
import pytest
from qbridge.modes import ExecutionMode, detect_mode

def _q(n):
    return cirq.LineQubit.range(n)

def test_sans_repetitions_c_est_le_vecteur_detat():
    c = cirq.Circuit(cirq.H(_q(1)[0]))
    assert detect_mode(c, repetitions=None) is ExecutionMode.STATE_VECTOR

def test_mesures_terminales_c_est_l_echantillonnage_terminal():
    q = _q(2)
    c = cirq.Circuit(cirq.H(q[0]), cirq.CX(q[0], q[1]), cirq.measure(*q, key="m"))
    assert detect_mode(c, repetitions=100) is ExecutionMode.TERMINAL_SAMPLING

def test_mesures_intermediaires_c_est_le_mode_midcircuit():
    q = _q(1)
    c = cirq.Circuit(cirq.H(q[0]), cirq.measure(q[0], key="a"),
                     cirq.H(q[0]), cirq.measure(q[0], key="b"))
    assert detect_mode(c, repetitions=100) is ExecutionMode.MIDCIRCUIT_SAMPLING

def test_une_seule_repetition_bascule_en_midcircuit():
    # qsim change de chemin C++ quand repetitions == 1 : on refuse de le traiter
    # comme un echantillonnage terminal, meme si les mesures le sont.
    q = _q(2)
    c = cirq.Circuit(cirq.H(q[0]), cirq.measure(*q, key="m"))
    assert detect_mode(c, repetitions=1) is ExecutionMode.MIDCIRCUIT_SAMPLING

def test_repetitions_nulle_ou_negative_est_refusee():
    q = _q(1)
    c = cirq.Circuit(cirq.H(q[0]), cirq.measure(q[0], key="m"))
    with pytest.raises(ValueError, match="repetitions"):
        detect_mode(c, repetitions=0)

def test_les_modes_sont_serialisables_en_chaine():
    assert ExecutionMode.STATE_VECTOR.value == "state_vector"
    assert ExecutionMode("state_vector") is ExecutionMode.STATE_VECTOR
```

- [ ] **Step 2: Vérifier l'échec**

Run: `.venv/Scripts/python.exe -m pytest tests/test_modes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qbridge.modes'`

- [ ] **Step 3: Implémenter**

```python
# src/qbridge/modes.py
"""Modes d'execution.

Le mode n'est pas cosmetique : qsim emprunte des chemins C++ differents selon
le mode, et la neutralite d'une option comme `cpu_threads` en depend
directement. Voir `tiers.py`.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import cirq


class ExecutionMode(str, Enum):
    STATE_VECTOR = "state_vector"
    """simulate() : vecteur d'etat complet, aucune mesure echantillonnee."""

    TERMINAL_SAMPLING = "terminal_sampling"
    """run() avec toutes les mesures terminales et repetitions > 1.
    qsim evolue l'etat une fois puis echantillonne l'etat final."""

    MIDCIRCUIT_SAMPLING = "midcircuit_sampling"
    """run() avec au moins une mesure non terminale, OU repetitions == 1.
    qsim re-execute le circuit par repetition et appelle VirtualMeasure."""

    EXPECTATION = "expectation"
    """simulate_expectation_values() : reduction sur tout le vecteur d'etat."""


def detect_mode(
    circuit: cirq.Circuit, *, repetitions: Optional[int]
) -> ExecutionMode:
    """Determine le mode d'execution a partir du circuit et des repetitions."""
    if repetitions is None:
        return ExecutionMode.STATE_VECTOR
    if repetitions < 1:
        raise ValueError(f"repetitions doit valoir au moins 1, recu {repetitions}")
    # repetitions == 1 bascule qsim sur la boucle par repetition (chemin B),
    # meme quand toutes les mesures sont terminales. On aligne le mode dessus.
    if repetitions == 1 or not circuit.are_all_measurements_terminal():
        return ExecutionMode.MIDCIRCUIT_SAMPLING
    return ExecutionMode.TERMINAL_SAMPLING
```

- [ ] **Step 4: Vérifier que ça passe**

Run: `.venv/Scripts/python.exe -m pytest tests/test_modes.py -v` → PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qbridge/modes.py tests/test_modes.py && git commit -m "feat: detection du mode d'execution"
```

---

## Task 4: Niveaux par (option × mode) (`tiers.py`)

**Files:** Create `src/qbridge/tiers.py`, Test `tests/test_tiers.py`

- [ ] **Step 1: Écrire les tests qui échouent**

```python
# tests/test_tiers.py
import dataclasses
import pytest
import qsimcirq
from qbridge.modes import ExecutionMode
from qbridge.tiers import Tier, tier_of, split_options, known_options

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
```

- [ ] **Step 2: Vérifier l'échec**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tiers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qbridge.tiers'`

- [ ] **Step 3: Implémenter**

```python
# src/qbridge/tiers.py
"""Classification (option x mode) -> niveau d'influence sur le resultat.

CETTE TABLE EST EMPIRIQUE. Mesuree le 2026-08-30 avec cirq 1.7.0 /
qsimcirq 0.22.0 sur qsim_avx2, OpenMP actif.
`tests/test_determinism_boundary.py` la re-mesure et echouera si qsim change.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Set

from qbridge.modes import ExecutionMode


class Tier(str, Enum):
    SEMANTIC = "semantic"
    """Change le resultat qualitativement. Identique obligatoire pour tout rejeu."""

    NUMERIC = "numeric"
    """Ne change le resultat qu'au niveau de l'arrondi flottant. Identique
    obligatoire pour un rejeu BIT_EXACT ; peut varier pour NUMERICALLY_EQUIVALENT."""

    PERFORMANCE = "performance"
    """Sans effet mesurable sur le resultat. Libre de varier d'une machine a l'autre."""


_TOUS_LES_MODES = tuple(ExecutionMode)

# Chaque entree : nom d'option -> {mode: niveau}.
OPTION_TIERS: Dict[str, Dict[ExecutionMode, Tier]] = {
    # MESURE : bit-identique t=1..16 en vecteur d'etat (jusqu'a 25 qubits) et en
    # echantillonnage terminal. MESURE DIFFERENT en midcircuit a 20 qubits :
    # VirtualMeasure lit un vecteur de normes partielles de longueur num_threads.
    # En mode expectation, RunReduce partitionne par thread : NUMERIC par prudence
    # (empiriquement stable a 20 qubits, mais le source montre la dependance).
    "cpu_threads": {
        ExecutionMode.STATE_VECTOR: Tier.PERFORMANCE,
        ExecutionMode.TERMINAL_SAMPLING: Tier.PERFORMANCE,
        ExecutionMode.MIDCIRCUIT_SAMPLING: Tier.SEMANTIC,
        ExecutionMode.EXPECTATION: Tier.NUMERIC,
    },
    # MESURE : f>=3 change le vecteur d'etat (max|delta| 3.2e-9, infidelite 1.4e-5).
    "max_fused_gate_size": {m: Tier.NUMERIC for m in _TOUS_LES_MODES},
    # FTZ/DAZ : sans effet sur les circuits testes, mais modifie la semantique
    # flottante en presence de denormaux. NUMERIC par prudence.
    "denormals_are_zeros": {m: Tier.NUMERIC for m in _TOUS_LES_MODES},
    # Noyaux CUDA entierement distincts des noyaux AVX.
    "use_gpu": {m: Tier.NUMERIC for m in _TOUS_LES_MODES},
    "gpu_mode": {m: Tier.NUMERIC for m in _TOUS_LES_MODES},
    # Reductions CUDA non tracees : ne pas supposer neutres.
    "gpu_state_threads": {m: Tier.NUMERIC for m in _TOUS_LES_MODES},
    "gpu_data_blocks": {m: Tier.NUMERIC for m in _TOUS_LES_MODES},
    # Nombre de trajectoires moyennees : change la valeur estimee.
    "ev_noisy_repetitions": {m: Tier.SEMANTIC for m in _TOUS_LES_MODES},
    # Journalisation seulement. La seule option franchement neutre.
    "verbosity": {m: Tier.PERFORMANCE for m in _TOUS_LES_MODES},
}


def known_options() -> Set[str]:
    """Noms d'options couverts par la table."""
    return set(OPTION_TIERS)


def tier_of(option_name: str, mode: ExecutionMode) -> Tier:
    """Niveau d'une option pour un mode donne. Leve KeyError si inconnue."""
    try:
        par_mode = OPTION_TIERS[option_name]
    except KeyError:
        raise KeyError(
            f"Option qsim inconnue : {option_name!r}. "
            f"Options classees : {sorted(OPTION_TIERS)}"
        ) from None
    return par_mode[mode]


def split_options(
    options: Dict[str, Any], mode: ExecutionMode
) -> Dict[Tier, Dict[str, Any]]:
    """Repartit un dict d'options en trois dicts, un par niveau, pour ce mode."""
    parts: Dict[Tier, Dict[str, Any]] = {t: {} for t in Tier}
    for name, value in options.items():
        parts[tier_of(name, mode)][name] = value
    return parts
```

- [ ] **Step 4: Vérifier que ça passe**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tiers.py -v` → PASS (11 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qbridge/tiers.py tests/test_tiers.py && git commit -m "feat: niveaux d'influence par couple (option, mode)"
```

---

## Task 5: Empreinte d'environnement (`fingerprint.py`)

Doit capturer le **noyau SIMD réellement chargé** : la wheel embarque AVX512/AVX2/SSE/basic et choisit par CPUID **à l'import**. Un rejeu sur un autre CPU peut donc charger un autre noyau et produire d'autres amplitudes.

**Files:** Create `src/qbridge/fingerprint.py`, Test `tests/test_fingerprint.py`

- [ ] **Step 1: Écrire les tests qui échouent**

```python
# tests/test_fingerprint.py
from qbridge.digest import canonical_json
from qbridge.fingerprint import environment_fingerprint, kernel_fingerprint

def test_contient_les_versions():
    fp = environment_fingerprint()
    assert fp["cirq_version"] and fp["qsimcirq_version"] and fp["numpy_version"]

def test_contient_la_plateforme():
    fp = environment_fingerprint()
    for cle in ("python_version", "platform", "machine", "processor", "cpu_count"):
        assert cle in fp

def test_contient_le_noyau_simd_reellement_charge():
    fp = environment_fingerprint()
    assert fp["qsim_kernel_module"].startswith("qsimcirq.qsim")
    assert isinstance(fp["qsim_instruction_set"], int)

def test_le_noyau_est_isolable():
    k = kernel_fingerprint()
    assert set(k) == {"qsim_kernel_module", "qsim_instruction_set", "qsim_gpu_mode"}

def test_serialisable_en_json_canonique():
    canonical_json(environment_fingerprint())

def test_stable_dans_un_meme_processus():
    assert environment_fingerprint() == environment_fingerprint()

def test_aucune_valeur_none():
    assert all(v is not None for v in environment_fingerprint().values())
```

- [ ] **Step 2: Vérifier l'échec**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fingerprint.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qbridge.fingerprint'`

- [ ] **Step 3: Implémenter**

```python
# src/qbridge/fingerprint.py
"""Empreinte de l'environnement d'execution.

L'empreinte generale n'entre PAS dans le hash semantique : elle sert a
expliquer une divergence, pas a la provoquer. Le noyau SIMD est traite a part
parce qu'il PEUT changer le resultat : la wheel qsimcirq embarque quatre
noyaux (AVX512F / AVX2 / SSE4.1 / basic) et en choisit un par CPUID a l'import.
"""
from __future__ import annotations

import os
import platform
import sys
from typing import Any, Dict


def kernel_fingerprint() -> Dict[str, Any]:
    """Identifie le noyau de calcul reellement charge par qsimcirq."""
    import qsimcirq

    module = getattr(qsimcirq.qsim, "__name__", "unknown")
    try:
        jeu = int(qsimcirq.qsim_decide.detect_instructions())
    except Exception:
        jeu = -1
    try:
        gpu = int(qsimcirq.qsim_decide.detect_gpu())
    except Exception:
        gpu = -1
    return {
        "qsim_kernel_module": module,
        "qsim_instruction_set": jeu,  # 0=AVX512F 1=AVX2 2=SSE4.1 3=basic -1=inconnu
        "qsim_gpu_mode": gpu,  # 10 = pas de GPU
    }


def environment_fingerprint() -> Dict[str, Any]:
    """Capture machine, bibliotheques et noyau de calcul."""
    import cirq
    import numpy
    import qsimcirq

    fp: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count() or 0,
        "cirq_version": cirq.__version__,
        "qsimcirq_version": qsimcirq.__version__,
        "numpy_version": numpy.__version__,
    }
    fp.update(kernel_fingerprint())
    return fp
```

- [ ] **Step 4: Vérifier que ça passe**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fingerprint.py -v` → PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qbridge/fingerprint.py tests/test_fingerprint.py && git commit -m "feat: empreinte d'environnement avec noyau SIMD"
```

---

## Task 6: Protocole `Backend` et oracle Cirq

**Files:** Create `src/qbridge/backends/__init__.py`, `base.py`, `cirq_ref.py`, Test `tests/test_backends.py`

- [ ] **Step 1: Écrire les tests qui échouent**

```python
# tests/test_backends.py
import cirq
import numpy as np
import pytest
from qbridge.backends.cirq_ref import CirqReferenceBackend

@pytest.fixture
def bell():
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1))

@pytest.fixture
def bell_mesure():
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1), cirq.measure(q0, q1, key="m"))

def test_nom_et_version():
    b = CirqReferenceBackend()
    assert b.name == "cirq-reference"
    assert b.version

def test_simulate_renvoie_un_vecteur_detat(bell):
    sv = CirqReferenceBackend().simulate(bell, seed=7, options={})
    assert sv.shape == (4,)
    assert np.isclose(abs(sv[0]) ** 2, 0.5) and np.isclose(abs(sv[3]) ** 2, 0.5)

def test_simulate_deterministe_a_seed_fixe(bell):
    b = CirqReferenceBackend()
    assert b.simulate(bell, seed=7, options={}).tobytes() == \
           b.simulate(bell, seed=7, options={}).tobytes()

def test_sample_deterministe_a_seed_fixe(bell_mesure):
    b = CirqReferenceBackend()
    s1 = b.sample(bell_mesure, repetitions=50, seed=7, options={})
    s2 = b.sample(bell_mesure, repetitions=50, seed=7, options={})
    assert s1["m"].tobytes() == s2["m"].tobytes()

def test_sample_change_avec_le_seed(bell_mesure):
    b = CirqReferenceBackend()
    s1 = b.sample(bell_mesure, repetitions=200, seed=7, options={})
    s2 = b.sample(bell_mesure, repetitions=200, seed=8, options={})
    assert s1["m"].tobytes() != s2["m"].tobytes()

def test_refuse_toute_option(bell):
    with pytest.raises(ValueError, match="n'accepte aucune option"):
        CirqReferenceBackend().simulate(bell, seed=7, options={"cpu_threads": 4})

def test_est_rejouable_bit_pour_bit():
    assert CirqReferenceBackend().is_bit_exact_replayable() is True
```

- [ ] **Step 2: Vérifier l'échec**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backends.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qbridge.backends'`

- [ ] **Step 3: Implémenter**

```python
# src/qbridge/backends/base.py
"""Le protocole `Backend` : la frontiere destinee a durer.

Aujourd'hui elle est implementee par des simulateurs. Le jour ou une machine
reelle est disponible, un `HardwareBackend` implemente le meme protocole et le
manifeste ne change pas de forme — seul le verdict de rejeu se degrade de
BIT_EXACT vers STATISTICALLY_COMPATIBLE.
"""
from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable

import cirq
import numpy as np


@runtime_checkable
class Backend(Protocol):
    """Executeur de circuit. Doit etre sans etat entre deux appels."""

    name: str
    version: str

    def simulate(
        self, circuit: cirq.Circuit, *, seed: int, options: Dict[str, Any]
    ) -> np.ndarray:
        """Renvoie le vecteur d'etat final complet.

        Un backend materiel levera `NotImplementedError` : on ne peut pas lire
        le vecteur d'etat d'une machine reelle (mesure destructive, no-cloning).
        """
        ...

    def sample(
        self,
        circuit: cirq.Circuit,
        *,
        repetitions: int,
        seed: int,
        options: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Renvoie les mesures, indexees par cle de mesure."""
        ...

    def is_bit_exact_replayable(self) -> bool:
        """Vrai si ce backend peut reproduire un resultat bit-pour-bit.
        Faux pour tout materiel reel : le bruit physique n'est pas rejouable."""
        ...
```

```python
# src/qbridge/backends/cirq_ref.py
"""Backend de reference : le simulateur natif de Cirq.

Sert d'oracle independant. Il est lent, mais il ne partage aucune ligne de code
avec qsim : si les deux concordent, la probabilite d'un bug commun est faible.
"""
from __future__ import annotations

from typing import Any, Dict

import cirq
import numpy as np


class CirqReferenceBackend:
    """Enveloppe `cirq.Simulator`. Instance fraiche a chaque appel."""

    name = "cirq-reference"

    def __init__(self) -> None:
        self.version = cirq.__version__

    @staticmethod
    def _rejeter_les_options(options: Dict[str, Any]) -> None:
        if options:
            raise ValueError(
                f"Le backend {CirqReferenceBackend.name} n'accepte aucune option "
                f"d'execution ; recu : {sorted(options)}"
            )

    def simulate(
        self, circuit: cirq.Circuit, *, seed: int, options: Dict[str, Any]
    ) -> np.ndarray:
        self._rejeter_les_options(options)
        return cirq.Simulator(seed=seed).simulate(circuit).state_vector()

    def sample(
        self,
        circuit: cirq.Circuit,
        *,
        repetitions: int,
        seed: int,
        options: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        self._rejeter_les_options(options)
        return dict(cirq.Simulator(seed=seed).run(circuit, repetitions=repetitions).measurements)

    def is_bit_exact_replayable(self) -> bool:
        return True
```

```python
# src/qbridge/backends/__init__.py
from qbridge.backends.base import Backend
from qbridge.backends.cirq_ref import CirqReferenceBackend

BACKENDS = {CirqReferenceBackend.name: CirqReferenceBackend}

__all__ = ["Backend", "CirqReferenceBackend", "BACKENDS"]
```

- [ ] **Step 4: Vérifier que ça passe**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backends.py -v` → PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qbridge/backends/ tests/test_backends.py && git commit -m "feat: protocole Backend et oracle Cirq"
```

---

## Task 7: Backend qsim

Point critique : `QSimSimulator` est **à état** (`_prng` avance à chaque appel). Le backend doit construire une instance **fraîche** à chaque exécution, et un test doit le prouver.

**Files:** Create `src/qbridge/backends/qsim.py`, Modify `src/qbridge/backends/__init__.py`, Test `tests/test_backend_qsim.py`

- [ ] **Step 1: Écrire les tests qui échouent**

```python
# tests/test_backend_qsim.py
import cirq
import numpy as np
import pytest
from qbridge.backends.cirq_ref import CirqReferenceBackend
from qbridge.backends.qsim import QsimBackend
from qbridge.modes import ExecutionMode

@pytest.fixture
def bell():
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1))

@pytest.fixture
def bell_mesure():
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1), cirq.measure(q0, q1, key="m"))

def test_nom_et_version():
    b = QsimBackend()
    assert b.name == "qsim" and b.version

def test_concorde_avec_l_oracle_cirq(bell):
    a = QsimBackend().simulate(bell, seed=7, options={})
    b = CirqReferenceBackend().simulate(bell, seed=7, options={})
    assert np.allclose(a, b, atol=1e-6)

def test_simulate_deterministe(bell):
    b = QsimBackend()
    o = {"cpu_threads": 4, "max_fused_gate_size": 2}
    assert b.simulate(bell, seed=7, options=o).tobytes() == \
           b.simulate(bell, seed=7, options=o).tobytes()

def test_appels_repetes_sur_la_meme_instance_restent_reproductibles(bell_mesure):
    # QSimSimulator est a etat : son _prng avance. Le backend doit construire
    # une instance fraiche par appel, sinon deux appels identiques divergent.
    b = QsimBackend()
    s1 = b.sample(bell_mesure, repetitions=100, seed=7, options={})
    s2 = b.sample(bell_mesure, repetitions=100, seed=7, options={})
    assert s1["m"].tobytes() == s2["m"].tobytes(), (
        "Le backend reutilise un simulateur a etat : le rejeu est casse."
    )

def test_sample_change_avec_le_seed(bell_mesure):
    b = QsimBackend()
    s1 = b.sample(bell_mesure, repetitions=200, seed=7, options={})
    s2 = b.sample(bell_mesure, repetitions=200, seed=8, options={})
    assert s1["m"].tobytes() != s2["m"].tobytes()

def test_refuse_une_option_inconnue(bell):
    with pytest.raises(KeyError, match="inconnue"):
        QsimBackend().simulate(bell, seed=7, options={"pas_une_option": 1})

def test_est_rejouable_bit_pour_bit():
    assert QsimBackend().is_bit_exact_replayable() is True
```

- [ ] **Step 2: Vérifier l'échec**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backend_qsim.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qbridge.backends.qsim'`

- [ ] **Step 3: Implémenter**

```python
# src/qbridge/backends/qsim.py
"""Backend qsim.

Deux exigences non negociables :

1. Une instance `QSimSimulator` FRAICHE par execution. Son `_prng` avance a
   chaque appel — c'est teste en amont chez qsim
   (`test_sampling_nondeterminism`). Reutiliser une instance casserait le rejeu.
2. Toute option est validee contre la table des niveaux. Une option inconnue
   leve KeyError plutot que d'etre silencieusement ignoree : une option ignoree
   est exactement le genre de derive qui rend un rejeu faussement rassurant.
"""
from __future__ import annotations

from typing import Any, Dict

import cirq
import numpy as np
import qsimcirq

from qbridge.tiers import known_options


class QsimBackend:
    """Enveloppe `qsimcirq.QSimSimulator`."""

    name = "qsim"

    def __init__(self) -> None:
        self.version = qsimcirq.__version__

    @staticmethod
    def _valider(options: Dict[str, Any]) -> qsimcirq.QSimOptions:
        connues = known_options()
        for cle in options:
            if cle not in connues:
                raise KeyError(
                    f"Option qsim inconnue : {cle!r}. Options classees : {sorted(connues)}"
                )
        return qsimcirq.QSimOptions(**options)

    def _simulateur(
        self, options: Dict[str, Any], seed: int, noise: cirq.NoiseModel | None = None
    ) -> qsimcirq.QSimSimulator:
        """Toujours une instance neuve — voir la note 1 du module."""
        return qsimcirq.QSimSimulator(self._valider(options), seed=seed, noise=noise)

    def simulate(
        self, circuit: cirq.Circuit, *, seed: int, options: Dict[str, Any]
    ) -> np.ndarray:
        return self._simulateur(options, seed).simulate(circuit).state_vector()

    def sample(
        self,
        circuit: cirq.Circuit,
        *,
        repetitions: int,
        seed: int,
        options: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        sim = self._simulateur(options, seed)
        return dict(sim.run(circuit, repetitions=repetitions).measurements)

    def is_bit_exact_replayable(self) -> bool:
        return True
```

Remplacer `src/qbridge/backends/__init__.py` par exactement :

```python
# src/qbridge/backends/__init__.py
from qbridge.backends.base import Backend
from qbridge.backends.cirq_ref import CirqReferenceBackend
from qbridge.backends.qsim import QsimBackend

BACKENDS = {
    QsimBackend.name: QsimBackend,
    CirqReferenceBackend.name: CirqReferenceBackend,
}

__all__ = ["Backend", "CirqReferenceBackend", "QsimBackend", "BACKENDS"]
```

- [ ] **Step 4: Vérifier que ça passe**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backend_qsim.py -v` → PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qbridge/backends/qsim.py src/qbridge/backends/__init__.py tests/test_backend_qsim.py && git commit -m "feat: backend qsim, instance fraiche par execution"
```

---

## Task 8: Le manifeste

**Files:** Create `src/qbridge/manifest.py`, Test `tests/test_manifest.py`

- [ ] **Step 1: Écrire les tests qui échouent**

```python
# tests/test_manifest.py
import cirq
import pytest
from qbridge.manifest import MANIFEST_SCHEMA_VERSION, Manifest
from qbridge.modes import ExecutionMode

def _bell():
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1))

def _mid():
    q = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.H(q[0]), cirq.measure(q[0], key="a"),
                        cirq.H(q[1]), cirq.measure(q[1], key="b"))

def _build(circuit=None, seed=7, repetitions=None, options=None):
    return Manifest.build(
        circuit=circuit if circuit is not None else _bell(),
        backend_name="qsim", backend_version="0.22.0", seed=seed,
        repetitions=repetitions, options=options or {}, noise_json=None,
    )

def test_porte_une_version_de_schema():
    assert _build().schema_version == MANIFEST_SCHEMA_VERSION

def test_enregistre_le_mode_d_execution():
    assert _build().mode == ExecutionMode.STATE_VECTOR.value
    assert _build(repetitions=100).mode == ExecutionMode.TERMINAL_SAMPLING.value

def test_options_reparties_par_niveau_en_mode_vecteur_detat():
    m = _build(options={"cpu_threads": 8, "max_fused_gate_size": 3})
    assert m.performance_options == {"cpu_threads": 8}
    assert m.numeric_options == {"max_fused_gate_size": 3}

def test_cpu_threads_devient_semantique_en_midcircuit():
    m = _build(circuit=_mid(), repetitions=100, options={"cpu_threads": 8})
    assert m.semantic_options == {"cpu_threads": 8}
    assert m.performance_options == {}

def test_hash_semantique_ignore_les_options_de_performance():
    a = _build(options={"cpu_threads": 8, "max_fused_gate_size": 3})
    b = _build(options={"cpu_threads": 1, "max_fused_gate_size": 3})
    assert a.semantic_hash == b.semantic_hash

def test_hash_semantique_prend_en_compte_cpu_threads_en_midcircuit():
    a = _build(circuit=_mid(), repetitions=100, options={"cpu_threads": 8})
    b = _build(circuit=_mid(), repetitions=100, options={"cpu_threads": 1})
    assert a.semantic_hash != b.semantic_hash

def test_hash_semantique_change_avec_les_options_numeriques():
    assert _build(options={"max_fused_gate_size": 2}).semantic_hash != \
           _build(options={"max_fused_gate_size": 3}).semantic_hash

def test_hash_semantique_change_avec_le_seed():
    assert _build(seed=7).semantic_hash != _build(seed=8).semantic_hash

def test_hash_semantique_inclut_le_noyau_simd():
    m = _build()
    altere = Manifest.from_dict({**m.to_dict(),
                                 "kernel": {**m.kernel, "qsim_instruction_set": 3}})
    assert altere._compute_semantic_hash() != m.semantic_hash

def test_round_trip_json(tmp_path):
    m = _build(options={"cpu_threads": 4})
    chemin = tmp_path / "m.json"
    m.save(chemin)
    relu = Manifest.load(chemin)
    assert relu.semantic_hash == m.semantic_hash
    assert relu.circuit() == m.circuit()
    assert relu.environment == m.environment
    assert relu.kernel == m.kernel

def test_circuit_reconstruit_a_l_identique():
    m = _build()
    assert Manifest.from_dict(m.to_dict()).circuit() == m.circuit()

def test_seed_nul_refuse():
    with pytest.raises(ValueError, match="seed"):
        _build(seed=None)
```

- [ ] **Step 2: Vérifier l'échec**

Run: `.venv/Scripts/python.exe -m pytest tests/test_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qbridge.manifest'`

- [ ] **Step 3: Implémenter**

```python
# src/qbridge/manifest.py
"""Le manifeste : la recette scellee d'une execution.

Le manifeste ne contient JAMAIS d'etat quantique. Le no-cloning l'interdit sur
materiel reel, et accepter un etat depuis un simulateur creerait une API qui ne
peut pas survivre au passage au materiel. On ne scelle que la recette.

Le `semantic_hash` couvre : le circuit, le seed, le mode, les options de niveau
SEMANTIC et NUMERIC pour ce mode, et le noyau SIMD. Il EXCLUT les options de
niveau PERFORMANCE et le reste de l'environnement — parce qu'il est mesure
qu'elles ne changent pas le resultat.
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import cirq

from qbridge.digest import sha256_of, sha256_of_text
from qbridge.fingerprint import environment_fingerprint, kernel_fingerprint
from qbridge.modes import ExecutionMode, detect_mode
from qbridge.tiers import Tier, split_options

MANIFEST_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class Manifest:
    """Description complete et rejouable d'une execution de circuit."""

    schema_version: str
    created_at: str
    circuit_json: str
    circuit_hash: str
    backend_name: str
    backend_version: str
    mode: str
    seed: int
    repetitions: Optional[int]
    noise_json: Optional[str]
    semantic_options: Dict[str, Any]
    numeric_options: Dict[str, Any]
    performance_options: Dict[str, Any]
    kernel: Dict[str, Any]
    environment: Dict[str, Any]
    semantic_hash: str = field(default="")

    @classmethod
    def build(
        cls,
        *,
        circuit: cirq.Circuit,
        backend_name: str,
        backend_version: str,
        seed: Optional[int],
        repetitions: Optional[int],
        options: Dict[str, Any],
        noise_json: Optional[str],
    ) -> "Manifest":
        if seed is None:
            raise ValueError(
                "Un seed explicite est obligatoire : sans lui l'execution n'est "
                "pas reproductible et le manifeste serait mensonger."
            )
        mode = detect_mode(circuit, repetitions=repetitions)
        parts = split_options(options, mode)
        circuit_json = cirq.to_json(circuit)
        brut = cls(
            schema_version=MANIFEST_SCHEMA_VERSION,
            created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
            circuit_json=circuit_json,
            circuit_hash=sha256_of_text(circuit_json),
            backend_name=backend_name,
            backend_version=backend_version,
            mode=mode.value,
            seed=int(seed),
            repetitions=repetitions,
            noise_json=noise_json,
            semantic_options=parts[Tier.SEMANTIC],
            numeric_options=parts[Tier.NUMERIC],
            performance_options=parts[Tier.PERFORMANCE],
            kernel=kernel_fingerprint(),
            environment=environment_fingerprint(),
        )
        return cls(**{**brut.__dict__, "semantic_hash": brut._compute_semantic_hash()})

    def _compute_semantic_hash(self) -> str:
        """Hash de tout ce qui influence le resultat."""
        return sha256_of(
            {
                "schema_version": self.schema_version,
                "circuit_hash": self.circuit_hash,
                "backend_name": self.backend_name,
                "mode": self.mode,
                "seed": self.seed,
                "repetitions": self.repetitions,
                "noise_json": self.noise_json,
                "semantic_options": self.semantic_options,
                "numeric_options": self.numeric_options,
                "kernel": self.kernel,
            }
        )

    def execution_mode(self) -> ExecutionMode:
        return ExecutionMode(self.mode)

    def circuit(self) -> cirq.Circuit:
        """Reconstruit le circuit depuis le JSON scelle."""
        return cirq.read_json(json_text=self.circuit_json)

    def noise(self) -> Optional[cirq.NoiseModel]:
        if self.noise_json is None:
            return None
        return cirq.read_json(json_text=self.noise_json)

    def all_options(self) -> Dict[str, Any]:
        return {**self.semantic_options, **self.numeric_options, **self.performance_options}

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Manifest":
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
```

- [ ] **Step 4: Vérifier que ça passe**

Run: `.venv/Scripts/python.exe -m pytest tests/test_manifest.py -v` → PASS (12 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qbridge/manifest.py tests/test_manifest.py && git commit -m "feat: manifeste scelle, hash semantique dependant du mode"
```

---

## Task 9: Verdicts de conformité

**Files:** Create `src/qbridge/verdict.py`, Test `tests/test_verdict.py`

- [ ] **Step 1: Écrire les tests qui échouent**

```python
# tests/test_verdict.py
import numpy as np
from qbridge.verdict import (Verdict, chi2_homogeneity_pvalue, compare_samples,
                             compare_state_vectors)

def test_vecteurs_identiques_donnent_bit_exact():
    a = np.array([0.7071068, 0, 0, 0.7071068], dtype=np.complex64)
    assert compare_state_vectors(a, a.copy()).verdict is Verdict.BIT_EXACT

def test_ecart_d_arrondi_donne_numeriquement_equivalent():
    a = np.array([0.7071068, 0, 0, 0.7071068], dtype=np.complex64)
    b = a.copy()
    b[0] = np.complex64(a[0] + np.complex64(3e-9))
    r = compare_state_vectors(a, b)
    assert r.verdict is Verdict.NUMERICALLY_EQUIVALENT and r.infidelity < 1e-6

def test_vecteurs_differents_donnent_divergent():
    a = np.array([1, 0, 0, 0], dtype=np.complex64)
    b = np.array([0, 0, 0, 1], dtype=np.complex64)
    assert compare_state_vectors(a, b).verdict is Verdict.DIVERGENT

def test_formes_incompatibles_donnent_divergent():
    a = np.array([1, 0], dtype=np.complex64)
    b = np.array([1, 0, 0, 0], dtype=np.complex64)
    assert compare_state_vectors(a, b).verdict is Verdict.DIVERGENT

def test_echantillons_identiques_donnent_bit_exact():
    s = {"m": np.array([[0, 0], [1, 1]], dtype=np.uint8)}
    assert compare_samples(s, {"m": s["m"].copy()}).verdict is Verdict.BIT_EXACT

def test_memes_distributions_donnent_statistiquement_compatible():
    rng = np.random.default_rng(0)
    a = {"m": rng.integers(0, 2, size=(4000, 1), dtype=np.uint8)}
    b = {"m": rng.integers(0, 2, size=(4000, 1), dtype=np.uint8)}
    assert compare_samples(a, b).verdict is Verdict.STATISTICALLY_COMPATIBLE

def test_distributions_differentes_donnent_divergent():
    a = {"m": np.zeros((4000, 1), dtype=np.uint8)}
    b = {"m": np.random.default_rng(1).integers(0, 2, size=(4000, 1), dtype=np.uint8)}
    assert compare_samples(a, b).verdict is Verdict.DIVERGENT

def test_cles_de_mesure_differentes_donnent_divergent():
    a = {"m": np.zeros((10, 1), dtype=np.uint8)}
    b = {"autre": np.zeros((10, 1), dtype=np.uint8)}
    assert compare_samples(a, b).verdict is Verdict.DIVERGENT

def test_les_verdicts_sont_ordonnes():
    assert Verdict.BIT_EXACT < Verdict.NUMERICALLY_EQUIVALENT \
        < Verdict.STATISTICALLY_COMPATIBLE < Verdict.DIVERGENT

def test_chi2_sur_deux_echantillons_identiques_donne_p_eleve():
    c = {0: 500, 1: 500}
    assert chi2_homogeneity_pvalue(c, dict(c)) > 0.9

def test_chi2_sur_deux_echantillons_opposes_donne_p_faible():
    assert chi2_homogeneity_pvalue({0: 1000, 1: 0}, {0: 0, 1: 1000}) < 1e-10
```

- [ ] **Step 2: Vérifier l'échec**

Run: `.venv/Scripts/python.exe -m pytest tests/test_verdict.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qbridge.verdict'`

- [ ] **Step 3: Implémenter**

```python
# src/qbridge/verdict.py
"""Verdicts de conformite d'un rejeu.

L'echelle est graduee parce que l'egalite bit-pour-bit est le mauvais critere
des qu'on quitte un simulateur deterministe. Un backend materiel ne pourra
jamais faire mieux que STATISTICALLY_COMPATIBLE — c'est une propriete de la
physique, pas un defaut du harnais.

Le chi2 est implemente ici plutot que via scipy : une dependance de moins pour
un harnais cense rester lisible et executable dans dix ans.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional

import numpy as np

INFIDELITY_TOLERANCE = 1e-4
"""Seuil d'infidelite sous lequel deux vecteurs d'etat sont juges equivalents.
Calibre au-dessus de l'ecart mesure entre max_fused_gate_size=2 et 4
(infidelite ~1.4e-5) et bien au-dessus du bruit d'arrondi de complex64."""

CHI2_ALPHA = 0.001
"""Seuil de p-value. Volontairement bas : on veut detecter une vraie
divergence, pas signaler du bruit d'echantillonnage normal."""


class Verdict(IntEnum):
    """Du plus fort au plus faible. L'ordre entier permet de comparer."""

    BIT_EXACT = 0
    NUMERICALLY_EQUIVALENT = 1
    STATISTICALLY_COMPATIBLE = 2
    DIVERGENT = 3


@dataclass(frozen=True)
class ComparisonResult:
    verdict: Verdict
    detail: str
    infidelity: Optional[float] = None
    p_value: Optional[float] = None


def compare_state_vectors(a: np.ndarray, b: np.ndarray) -> ComparisonResult:
    """Compare deux vecteurs d'etat."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return ComparisonResult(
            Verdict.DIVERGENT, f"formes incompatibles : {a.shape} vs {b.shape}"
        )
    if a.dtype == b.dtype and a.tobytes() == b.tobytes():
        return ComparisonResult(Verdict.BIT_EXACT, "octets identiques", 0.0)

    infidelite = abs(1.0 - float(abs(np.vdot(a, b)) ** 2))
    if infidelite <= INFIDELITY_TOLERANCE:
        return ComparisonResult(
            Verdict.NUMERICALLY_EQUIVALENT,
            f"infidelite {infidelite:.3e} <= {INFIDELITY_TOLERANCE:.0e}",
            infidelite,
        )
    return ComparisonResult(
        Verdict.DIVERGENT,
        f"infidelite {infidelite:.3e} > {INFIDELITY_TOLERANCE:.0e}",
        infidelite,
    )


def _bitstring_counts(samples: np.ndarray) -> Dict[int, int]:
    """Convertit un tableau (repetitions, n_qubits) en comptage par entier."""
    arr = np.asarray(samples)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    poids = 1 << np.arange(arr.shape[1] - 1, -1, -1)
    valeurs = (arr.astype(np.int64) * poids).sum(axis=1)
    uniques, comptes = np.unique(valeurs, return_counts=True)
    return {int(u): int(c) for u, c in zip(uniques, comptes)}


def _regularized_gamma_p(a: float, x: float) -> float:
    """Fonction gamma incomplete inferieure regularisee P(a, x)."""
    if x <= 0 or a <= 0:
        return 0.0
    if x < a + 1.0:
        terme = 1.0 / a
        somme = terme
        n = a
        for _ in range(1000):
            n += 1.0
            terme *= x / n
            somme += terme
            if abs(terme) < abs(somme) * 1e-15:
                break
        return somme * math.exp(-x + a * math.log(x) - math.lgamma(a))
    minuscule = 1e-300
    b = x + 1.0 - a
    c = 1.0 / minuscule
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < minuscule:
            d = minuscule
        c = b + an / c
        if abs(c) < minuscule:
            c = minuscule
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return 1.0 - h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _chi2_survival(x: float, k: int) -> float:
    """P(X > x) pour X ~ chi2(k)."""
    if x <= 0:
        return 1.0
    return 1.0 - _regularized_gamma_p(k / 2.0, x / 2.0)


def chi2_homogeneity_pvalue(
    counts_a: Dict[int, int], counts_b: Dict[int, int]
) -> float:
    """p-value d'un test du chi2 d'homogeneite entre deux echantillons."""
    n_a = sum(counts_a.values())
    n_b = sum(counts_b.values())
    if n_a == 0 or n_b == 0:
        return 0.0

    chi2 = 0.0
    categories = 0
    for k in sorted(set(counts_a) | set(counts_b)):
        o_a = counts_a.get(k, 0)
        o_b = counts_b.get(k, 0)
        total = o_a + o_b
        if total == 0:
            continue
        e_a = total * n_a / (n_a + n_b)
        e_b = total * n_b / (n_a + n_b)
        if e_a > 0:
            chi2 += (o_a - e_a) ** 2 / e_a
        if e_b > 0:
            chi2 += (o_b - e_b) ** 2 / e_b
        categories += 1
    return _chi2_survival(chi2, max(categories - 1, 1))


def compare_samples(
    a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]
) -> ComparisonResult:
    """Compare deux jeux de mesures."""
    if set(a) != set(b):
        return ComparisonResult(
            Verdict.DIVERGENT,
            f"cles de mesure differentes : {sorted(a)} vs {sorted(b)}",
        )
    if all(a[k].shape == b[k].shape and a[k].tobytes() == b[k].tobytes() for k in a):
        return ComparisonResult(Verdict.BIT_EXACT, "echantillons identiques")

    p_min, cle_min = 1.0, ""
    for k in sorted(a):
        p = chi2_homogeneity_pvalue(_bitstring_counts(a[k]), _bitstring_counts(b[k]))
        if p < p_min:
            p_min, cle_min = p, k
    if p_min >= CHI2_ALPHA:
        return ComparisonResult(
            Verdict.STATISTICALLY_COMPATIBLE,
            f"chi2 p={p_min:.4f} >= {CHI2_ALPHA} (cle la plus faible : {cle_min!r})",
            p_value=p_min,
        )
    return ComparisonResult(
        Verdict.DIVERGENT,
        f"chi2 p={p_min:.2e} < {CHI2_ALPHA} sur la cle {cle_min!r}",
        p_value=p_min,
    )
```

- [ ] **Step 4: Vérifier que ça passe**

Run: `.venv/Scripts/python.exe -m pytest tests/test_verdict.py -v` → PASS (11 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qbridge/verdict.py tests/test_verdict.py && git commit -m "feat: verdicts gradues, chi2 sans scipy"
```

---

## Task 10: capture() et replay()

**Files:** Create `src/qbridge/capture.py`, `src/qbridge/replay.py`, Modify `src/qbridge/__init__.py`, Test `tests/test_capture_replay.py`

- [ ] **Step 1: Écrire les tests qui échouent**

```python
# tests/test_capture_replay.py
import json
import cirq
import pytest
from qbridge import capture, replay
from qbridge.manifest import Manifest
from qbridge.verdict import Verdict

def _bell():
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1))

def _bell_mesure():
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1), cirq.measure(q0, q1, key="m"))

def test_capture_puis_replay_donne_bit_exact():
    run = capture(_bell(), backend="qsim", seed=7)
    assert replay(run.manifest).verdict is Verdict.BIT_EXACT

def test_capture_avec_mesures_donne_bit_exact():
    run = capture(_bell_mesure(), backend="qsim", seed=7, repetitions=100)
    assert replay(run.manifest).verdict is Verdict.BIT_EXACT

def test_replay_survit_a_un_changement_de_threads_en_vecteur_detat():
    run = capture(_bell(), backend="qsim", seed=7, options={"cpu_threads": 1})
    assert replay(run.manifest, override_performance={"cpu_threads": 8}).verdict \
        is Verdict.BIT_EXACT

def test_replay_refuse_de_changer_les_threads_en_midcircuit():
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.H(q[0]), cirq.measure(q[0], key="a"),
                     cirq.H(q[1]), cirq.measure(q[1], key="b"))
    run = capture(c, backend="qsim", seed=7, repetitions=50, options={"cpu_threads": 1})
    with pytest.raises(ValueError, match="PERFORMANCE"):
        replay(run.manifest, override_performance={"cpu_threads": 8})

def test_replay_sur_l_oracle_cirq_reste_au_moins_numeriquement_equivalent():
    run = capture(_bell(), backend="qsim", seed=7)
    assert replay(run.manifest, backend="cirq-reference").verdict \
        <= Verdict.NUMERICALLY_EQUIVALENT

def test_replay_detecte_un_manifeste_altere(tmp_path):
    run = capture(_bell(), backend="qsim", seed=7)
    chemin = tmp_path / "m.json"
    run.manifest.save(chemin)
    data = json.loads(chemin.read_text(encoding="utf-8"))
    data["seed"] = 999
    chemin.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="integrite"):
        replay(Manifest.load(chemin))

def test_replay_depuis_un_fichier(tmp_path):
    run = capture(_bell(), backend="qsim", seed=7)
    chemin = tmp_path / "m.json"
    run.manifest.save(chemin)
    assert replay(Manifest.load(chemin)).verdict is Verdict.BIT_EXACT

def test_capture_refuse_un_backend_inconnu():
    with pytest.raises(KeyError, match="inconnu"):
        capture(_bell(), backend="backend_imaginaire", seed=7)

def test_capture_expose_un_hash_de_resultat():
    a = capture(_bell(), backend="qsim", seed=7)
    b = capture(_bell(), backend="qsim", seed=7)
    assert a.result_hash == b.result_hash
```

- [ ] **Step 2: Vérifier l'échec**

Run: `.venv/Scripts/python.exe -m pytest tests/test_capture_replay.py -v`
Expected: FAIL — `ImportError: cannot import name 'capture' from 'qbridge'`

- [ ] **Step 3: Implémenter**

```python
# src/qbridge/capture.py
"""capture() : executer un circuit et sceller tout ce qu'il faut pour le rejouer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import cirq
import numpy as np

from qbridge.backends import BACKENDS
from qbridge.digest import sha256_of_array, sha256_of_text
from qbridge.manifest import Manifest


@dataclass(frozen=True)
class CaptureRun:
    """Le manifeste plus les resultats obtenus lors de la capture initiale."""

    manifest: Manifest
    state_vector: Optional[np.ndarray]
    samples: Optional[Dict[str, np.ndarray]]
    result_hash: str


def _hash_samples(samples: Dict[str, np.ndarray]) -> str:
    return sha256_of_text("".join(f"{k}:{sha256_of_array(samples[k])}" for k in sorted(samples)))


def capture(
    circuit: cirq.Circuit,
    *,
    backend: str = "qsim",
    seed: int,
    repetitions: Optional[int] = None,
    options: Optional[Dict[str, Any]] = None,
    noise: Optional[cirq.NoiseModel] = None,
) -> CaptureRun:
    """Execute `circuit` et renvoie un `CaptureRun` scelle.

    Si `repetitions` est fourni on echantillonne ; sinon on calcule le vecteur
    d'etat complet.
    """
    if backend not in BACKENDS:
        raise KeyError(f"Backend inconnu : {backend!r}. Disponibles : {sorted(BACKENDS)}")
    options = dict(options or {})
    impl = BACKENDS[backend]()

    manifest = Manifest.build(
        circuit=circuit,
        backend_name=impl.name,
        backend_version=impl.version,
        seed=seed,
        repetitions=repetitions,
        options=options,
        noise_json=cirq.to_json(noise) if noise is not None else None,
    )

    if repetitions is None:
        sv = impl.simulate(circuit, seed=seed, options=options)
        return CaptureRun(manifest, sv, None, sha256_of_array(sv))

    samples = impl.sample(circuit, repetitions=repetitions, seed=seed, options=options)
    return CaptureRun(manifest, None, samples, _hash_samples(samples))
```

```python
# src/qbridge/replay.py
"""replay() : re-executer depuis un manifeste et rendre un verdict."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from qbridge.backends import BACKENDS
from qbridge.capture import capture
from qbridge.manifest import Manifest
from qbridge.tiers import Tier, split_options
from qbridge.verdict import (ComparisonResult, Verdict, compare_samples,
                             compare_state_vectors)


@dataclass(frozen=True)
class ReplayReport:
    verdict: Verdict
    detail: str
    comparison: ComparisonResult
    original_backend: str
    replay_backend: str
    kernel_changed: bool
    environment_drift: Dict[str, Any]


def _verifier_integrite(manifest: Manifest) -> None:
    """Refuse un manifeste dont le hash ne correspond plus au contenu."""
    attendu = manifest._compute_semantic_hash()
    if attendu != manifest.semantic_hash:
        raise ValueError(
            "Echec du controle d'integrite : le manifeste a ete modifie. "
            f"hash stocke={manifest.semantic_hash[:16]}... "
            f"hash recalcule={attendu[:16]}..."
        )


def _derive_environnement(manifest: Manifest) -> Dict[str, Any]:
    from qbridge.fingerprint import environment_fingerprint

    actuel = environment_fingerprint()
    return {
        cle: {"capture": manifest.environment.get(cle), "replay": actuel.get(cle)}
        for cle in sorted(set(manifest.environment) | set(actuel))
        if manifest.environment.get(cle) != actuel.get(cle)
    }


def replay(
    manifest: Manifest,
    *,
    backend: Optional[str] = None,
    override_performance: Optional[Dict[str, Any]] = None,
) -> ReplayReport:
    """Rejoue l'execution decrite par `manifest`.

    `override_performance` ne peut contenir que des options qui sont de niveau
    PERFORMANCE POUR LE MODE DU MANIFESTE. C'est ce qui permet de rejouer sur
    une machine ayant un autre nombre de coeurs — sauf en mode midcircuit, ou
    `cpu_threads` fait partie de l'algorithme de mesure.
    """
    _verifier_integrite(manifest)

    nom = backend or manifest.backend_name
    if nom not in BACKENDS:
        raise KeyError(f"Backend inconnu : {nom!r}. Disponibles : {sorted(BACKENDS)}")

    mode = manifest.execution_mode()
    options = dict(manifest.all_options())

    if override_performance:
        parts = split_options(override_performance, mode)
        interdits = {**parts[Tier.SEMANTIC], **parts[Tier.NUMERIC]}
        if interdits:
            raise ValueError(
                f"En mode {mode.value}, ces options ne sont pas de niveau "
                f"PERFORMANCE et ne peuvent pas etre surchargees : {sorted(interdits)}"
            )
        options.update(override_performance)

    if nom == "cirq-reference":
        options = {}  # l'oracle n'accepte aucune option d'execution

    impl = BACKENDS[nom]()
    circuit = manifest.circuit()

    if manifest.repetitions is None:
        rejoue = impl.simulate(circuit, seed=manifest.seed, options=options)
    else:
        rejoue = impl.sample(
            circuit, repetitions=manifest.repetitions, seed=manifest.seed, options=options
        )

    origine = capture(
        circuit,
        backend=manifest.backend_name,
        seed=manifest.seed,
        repetitions=manifest.repetitions,
        options=manifest.all_options(),
        noise=manifest.noise(),
    )

    if manifest.repetitions is None:
        comparaison = compare_state_vectors(origine.state_vector, rejoue)
    else:
        comparaison = compare_samples(origine.samples, rejoue)

    from qbridge.fingerprint import kernel_fingerprint

    return ReplayReport(
        verdict=comparaison.verdict,
        detail=comparaison.detail,
        comparison=comparaison,
        original_backend=manifest.backend_name,
        replay_backend=nom,
        kernel_changed=kernel_fingerprint() != manifest.kernel,
        environment_drift=_derive_environnement(manifest),
    )
```

Remplacer `src/qbridge/__init__.py` par :

```python
# src/qbridge/__init__.py
"""qbridge — harnais de capture/replay pour executions de circuits quantiques."""
from qbridge.capture import CaptureRun, capture
from qbridge.manifest import Manifest
from qbridge.modes import ExecutionMode
from qbridge.replay import ReplayReport, replay
from qbridge.tiers import Tier
from qbridge.verdict import Verdict

__version__ = "0.1.0"
__all__ = [
    "capture", "replay", "Manifest", "CaptureRun", "ReplayReport",
    "Verdict", "Tier", "ExecutionMode", "__version__",
]
```

- [ ] **Step 4: Vérifier que ça passe**

Run: `.venv/Scripts/python.exe -m pytest tests/test_capture_replay.py -v` → PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qbridge/capture.py src/qbridge/replay.py src/qbridge/__init__.py tests/test_capture_replay.py && git commit -m "feat: capture() et replay() avec controle d'integrite"
```

---

## Task 11: Tests de frontière du déterminisme

Ces tests **re-mesurent** les faits qui fondent la conception. S'ils échouent après une mise à jour de cirq ou qsimcirq, ce n'est pas le test qui est faux : c'est `OPTION_TIERS` qui doit être révisé.

**Files:** Create `tests/test_determinism_boundary.py`

- [ ] **Step 1: Écrire les tests**

```python
# tests/test_determinism_boundary.py
"""Verrouillage des faits empiriques qui fondent OPTION_TIERS.

Chaque test correspond a une ligne du tableau de faits du plan. Un echec ici
signifie que qsim a change de comportement et que la table doit etre revue.
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
    return qsimcirq.QSimSimulator(qsimcirq.QSimOptions(**opts)).simulate(circuit).state_vector()


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
    """FAIT : t=1 et t>=2 donnent des bitstrings differents (VirtualMeasure lit
    un vecteur de normes partielles de longueur num_threads)."""
    def ech(t):
        sim = qsimcirq.QSimSimulator(qsimcirq.QSimOptions(cpu_threads=t), seed=5)
        r = sim.run(circuit_midcircuit, repetitions=200)
        return sha256_of_array(r.measurements["fin"])

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
```

- [ ] **Step 2: Lancer les tests de frontière**

Run: `.venv/Scripts/python.exe -m pytest tests/test_determinism_boundary.py -v`
Expected: PASS — 12 passed (4 paramétrages + 8 tests)

- [ ] **Step 3: Lancer la suite complète**

Run: `.venv/Scripts/python.exe -m pytest -v`
Expected: toute la suite verte

- [ ] **Step 4: Commit**

```bash
git add tests/test_determinism_boundary.py && git commit -m "test: verrouillage des faits empiriques du determinisme qsim"
```

---

## Task 12: Démonstration de bout en bout et README

**Files:** Create `examples/demo.py`, `README.md`

- [ ] **Step 1: Écrire la démo**

```python
# examples/demo.py
"""Demonstration : capturer, sceller sur disque, rejouer, comparer."""
from pathlib import Path

import cirq

from qbridge import capture, replay
from qbridge.manifest import Manifest

SORTIE = Path(__file__).resolve().parent.parent / "runs"
SORTIE.mkdir(exist_ok=True)


def bandeau(titre: str) -> None:
    print(f"\n{'=' * 68}\n{titre}\n{'=' * 68}")


def main() -> None:
    qubits = cirq.GridQubit.rect(3, 4)
    circuit = cirq.experiments.random_rotations_between_grid_interaction_layers_circuit(
        qubits=qubits, depth=12, seed=2026
    )

    bandeau("1. Capture sur qsim (8 threads, fusion 2, mode vecteur d'etat)")
    run = capture(
        circuit, backend="qsim", seed=7,
        options={"cpu_threads": 8, "max_fused_gate_size": 2},
    )
    chemin = SORTIE / "demo_manifest.json"
    run.manifest.save(chemin)
    print(f"  mode             : {run.manifest.mode}")
    print(f"  noyau            : {run.manifest.kernel['qsim_kernel_module']}")
    print(f"  hash semantique  : {run.manifest.semantic_hash[:32]}...")
    print(f"  hash resultat    : {run.result_hash[:32]}...")
    print(f"  options perf     : {run.manifest.performance_options}")
    print(f"  options numeriques: {run.manifest.numeric_options}")
    print(f"  manifeste        : {chemin} ({chemin.stat().st_size} octets)")

    bandeau("2. Rejeu a l'identique")
    r = replay(Manifest.load(chemin))
    print(f"  verdict : {r.verdict.name} — {r.detail}")

    bandeau("3. Rejeu avec 1 seul thread (PERFORMANCE dans ce mode)")
    r = replay(Manifest.load(chemin), override_performance={"cpu_threads": 1})
    print(f"  verdict : {r.verdict.name} — {r.detail}")

    bandeau("4. Rejeu sur l'oracle Cirq (moteur entierement different)")
    r = replay(Manifest.load(chemin), backend="cirq-reference")
    print(f"  verdict : {r.verdict.name} — {r.detail}")
    print(f"  noyau change : {r.kernel_changed}")

    bandeau("5. Meme circuit avec mesures intermediaires : threads verrouilles")
    q = cirq.LineQubit.range(8)
    mid = cirq.Circuit([cirq.H(x) for x in q])
    mid.append(cirq.measure(*q[:4], key="mid"))
    mid.append([cirq.X(x) ** 0.37 for x in q])
    mid.append(cirq.measure(*q, key="fin"))
    run_mid = capture(mid, backend="qsim", seed=7, repetitions=200,
                      options={"cpu_threads": 4})
    print(f"  mode              : {run_mid.manifest.mode}")
    print(f"  options semantiques: {run_mid.manifest.semantic_options}")
    print(f"  verdict rejeu      : {replay(run_mid.manifest).verdict.name}")
    try:
        replay(run_mid.manifest, override_performance={"cpu_threads": 1})
    except ValueError as e:
        print(f"  surcharge refusee  : {e}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lancer la démo**

Run: `.venv/Scripts/python.exe examples/demo.py`
Expected: étapes 2 et 3 en `BIT_EXACT` ; étape 4 en `BIT_EXACT` ou `NUMERICALLY_EQUIVALENT` ; étape 5 rejeu `BIT_EXACT` et surcharge refusée avec un message explicite.

- [ ] **Step 3: Écrire le README**

Le `README.md` doit contenir, dans cet ordre :
1. Le principe en trois phrases : on scelle la recette, jamais l'état — le no-cloning interdit de copier un état quantique, la mesure est destructive, donc « reconstituer un environnement » ne peut vouloir dire que « rejouer la recette et prouver la conformité ».
2. Le tableau complet des faits mesurés, recopié depuis ce plan.
3. Le tableau (mode × `cpu_threads`).
4. L'échelle des quatre verdicts avec, pour chacun, quand il est atteignable.
5. Installation : `python -m venv .venv` puis `.venv/Scripts/python.exe -m pip install -e .`
6. Un avertissement : la table `OPTION_TIERS` est valable pour cirq 1.7.0 / qsimcirq 0.22.0 sur AVX2 ; lancer `pytest tests/test_determinism_boundary.py` après toute mise à jour.

- [ ] **Step 4: Commit**

```bash
git add examples/demo.py README.md && git commit -m "docs: demonstration de bout en bout et README"
```

---

## Hors périmètre pour cette version

Volontairement exclu, à traiter dans un plan séparé une fois le noyau validé :

- `HardwareBackend` et l'intégration de snapshots de calibration réels
- Export vers OpenQASM 3 / QIR (portabilité entre écosystèmes)
- Capture du pré/post-traitement classique — le versant « environnement IA » du pont
- Compression du `circuit_json` (180 Ko pour 20 qubits × depth 16 ; deviendra un problème)
- Mode `EXPECTATION` de bout en bout (le niveau est classé, le chemin `capture`/`replay` ne l'implémente pas encore)
- Interface en ligne de commande

## Limites connues et assumées

- `replay()` ré-exécute la capture d'origine pour comparer, au lieu de comparer à un hash stocké. C'est plus lent mais cela détecte aussi une dérive de l'environnement local. Une comparaison au `result_hash` seul serait une optimisation ultérieure.
- La classification de `cpu_threads` en `NUMERIC` pour le mode `EXPECTATION` est **conservatrice** : le code source de qsim partitionne la réduction par nombre de threads, mais aucune divergence n'a été mesurée à 20 qubits. Le coût d'une erreur dans ce sens est nul ; l'inverse serait un rejeu faussement validé.
- La reproductibilité entre deux CPU du **même** palier SIMD (deux machines AVX2 différentes) n'est pas vérifiée. Le `kernel_fingerprint` la détecterait comme identique alors qu'elle pourrait diverger. À mesurer le jour où une seconde machine est disponible.
