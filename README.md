# qbridge

Harnais de capture/replay pour exécutions de circuits quantiques.

## Le principe

**On scelle la recette, jamais l'état.** Le théorème de non-clonage interdit de
copier un état quantique et la mesure est destructive : « reconstituer un
environnement quantique » ne peut donc rien vouloir dire d'autre que *rejouer la
recette et prouver la conformité du résultat*.

```python
run = capture(circuit, backend="qsim", seed=7)   # exécute et scelle
run.manifest.save("run.json")                    # un fichier autonome
...
rapport = replay(Manifest.load("run.json"))      # rejoue et compare
rapport.verdict                                  # BIT_EXACT
```

Le manifeste est un JSON autonome : circuit sérialisé, seed, mode d'exécution,
options réparties par niveau d'influence, noyau SIMD, empreinte d'environnement,
et un hash sémantique qui détecte toute falsification.

## Ce qui rend ce harnais différent

`Backend` est un protocole abstrait. Aujourd'hui il est implémenté par `qsim` et
par le simulateur de référence de Cirq. Le jour où une machine réelle est
disponible, un `HardwareBackend` implémente le même protocole : **le manifeste ne
change pas de forme**, seul le verdict atteignable se dégrade.

| Verdict | Signification | Atteignable par |
|---|---|---|
| `BIT_EXACT` | octets identiques | simulateur, options identiques |
| `NUMERICALLY_EQUIVALENT` | infidélité ≤ 1e-4 | simulateur, options numériques différentes |
| `STATISTICALLY_COMPATIBLE` | χ² p ≥ 0.001 | **plafond du matériel réel** |
| `DIVERGENT` | au-delà | dérive à expliquer |

## Le modèle central : le niveau dépend du mode

Une option d'exécution n'a pas un niveau d'influence absolu. Elle en a un **par
mode d'exécution**. C'est mesuré, pas déclaré.

| Mode | `cpu_threads` |
|---|---|
| `STATE_VECTOR` | PERFORMANCE — bit-identique de 1 à 16 threads, jusqu'à 25 qubits |
| `TERMINAL_SAMPLING` | PERFORMANCE — mêmes bitstrings |
| `MIDCIRCUIT_SAMPLING` | **SEMANTIC — change les bitstrings** |
| `EXPECTATION` | NUMERIC (prudence) |

La cause de la ligne critique : `statespace.h::VirtualMeasure` construit un
vecteur de normes partielles **dont la longueur est le nombre de threads**, et le
tirage aléatoire parcourt ce vecteur. Le nombre de threads fait donc partie de
l'algorithme de mesure, pas seulement de sa vitesse.

## Faits mesurés

Mesurés le 2026-08-30 — cirq 1.7.0, qsimcirq 0.22.0, `qsim_avx2`, OpenMP actif
(×1.86 à 25 qubits, donc la neutralité constatée n'est pas l'artefact d'un
threading inerte).

| Fait | Mesure |
|---|---|
| `cpu_threads` 1→16, vecteur d'état | bit-pour-bit identique jusqu'à 25 qubits |
| `cpu_threads` 1→16, échantillonnage terminal | bitstrings identiques |
| **`cpu_threads` 1 vs ≥2, mesures intermédiaires** | **hash `747f4099…` vs `d38f4a9c…` — différent** |
| `max_fused_gate_size` ≥ 3 | vecteur différent, max\|Δ\|≈3.2e-9, infidélité ≈1.4e-5 |
| `denormals_are_zeros` | sans effet sur les circuits testés |
| `seed` fixe, instance fraîche | reproductible |
| `seed` fixe, instance **réutilisée** | **non reproductible** — `_prng` avance |
| `seed=None` | non reproductible — `capture()` le refuse |
| round-trip `cirq.to_json` | exact, hash re-sérialisé stable |
| noyau SIMD | choisi par CPUID **à l'import**, pas figé dans la wheel |
| vecteur d'état | `complex64`, eps = 1.19e-7 |

Ces faits sont verrouillés par `tests/test_determinism_boundary.py`. **Si un de
ces tests échoue après une mise à jour, ce n'est pas le test qui est faux : c'est
la table `OPTION_TIERS` qui doit être révisée.**

## Installation

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install -e .
```

Puis la suite complète et la démonstration :

```bash
.venv/Scripts/python.exe -m pytest -q
```

```bash
.venv/Scripts/python.exe examples/demo.py
```

## Avertissement de portée

La table `OPTION_TIERS` est valable pour **cirq 1.7.0 / qsimcirq 0.22.0 sur
AVX2**. Relancer `pytest tests/test_determinism_boundary.py` après toute mise à
jour de l'un ou l'autre.

La reproductibilité entre deux CPU du **même** palier SIMD (deux machines AVX2
différentes) n'est pas vérifiée : `kernel_fingerprint()` les verrait identiques
alors qu'elles pourraient diverger. À mesurer le jour où une seconde machine est
disponible.

## Hors périmètre pour cette version

- `HardwareBackend` et l'intégration de snapshots de calibration réels
- **Rejeu archivistique** : régénérer les chiffres publiés depuis les bitstrings
  bruts, sans aucune ressource quantique. C'est la garantie que les utilisateurs
  exerceront réellement dans cinq ans — prochaine priorité.
- Export vers OpenQASM 3 / QIR
- Capture du pré/post-traitement classique
- Compression du `circuit_json` (180 Ko pour 20 qubits × depth 16)
- Mode `EXPECTATION` de bout en bout (classé, mais pas encore dans `capture`)
- Interface en ligne de commande

## État de l'art

Aucun format existant ne scelle une exécution complète. QIR et OpenQASM 3
scellent le *programme* ; les `fake_provider` de Qiskit scellent un *appareil* ;
`results.json` de Braket scelle un *enregistrement de job* ;
`cirq_google.workflow` est le squelette le plus proche mais re-résout l'appareil
au moment du rejeu. Chaque emplacement « environnement » existant est une
référence par identifiant, pas un sceau.
