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
et un hash sémantique.

> **Portée exacte du hash — à ne pas surestimer.** Le hash sémantique est
> **non signé** : il est public et déterministe, donc quiconque modifie le
> fichier peut le recalculer en deux lignes. Il détecte une **corruption ou une
> altération accidentelle**, ainsi qu'une modification faite sans connaître
> l'outil. Il ne détecte **pas** un adversaire délibéré. Pour ça il faudrait une
> signature (HMAC avec clé, ou ed25519 sur le JSON canonique) — non implémenté,
> et c'est la première chose à ajouter si l'archive doit servir de preuve
> opposable.

## Les trois garanties

Elles sont distinctes et il faut les séparer, parce que les confondre est
exactement là où ce genre d'outil échoue.

| Garantie | Ce qu'elle prouve | Plafond |
|---|---|---|
| **Bit-exact** | même simulateur, mêmes options → mêmes octets | simulateur seulement |
| **Statistique** | distributions compatibles (χ²) | **plafond du matériel réel** |
| **Archivistique** | les tirages archivés sont bien ceux qui ont été scellés, et tous les agrégats publiés s'en recalculent | **zéro ressource quantique** |

La troisième est la plus importante et la plus solide : elle ne dépend d'aucun
matériel, d'aucun simulateur, ni même de qsim installé. C'est celle que
quelqu'un exercera réellement dans cinq ans.

```python
record = RunRecord.from_capture(capture(circuit, seed=7, repetitions=500))
record.save("runs/mon_experience")           # manifest.json + samples.npz + record.json
...
verify_archival(RunRecord.load("runs/mon_experience"))   # sans rien exécuter
replay_record(RunRecord.load("runs/mon_experience"))     # compare à l'archive
```

**`replay_record` ≠ `replay`.** `replay(manifest)` ré-exécute la capture d'origine
pour obtenir sa référence : il ne peut donc pas détecter un bug présent dans les
*deux* exécutions. `replay_record` lit la référence sur disque, scellée avant.
C'est la seule des deux comparaisons qui prouve quelque chose sur la durée.

Les bitstrings bruts sont **la seule donnée non regénérable** de toute la chaîne
— le seul enregistrement physique de l'événement quantique. Ils sont archivés
tels quels, jamais agrégés : les agrégats se recalculent, les tirages non. Une
archive de 500 tirages sur un état de Bell pèse 3,1 Ko.

## Ligne de commande

```bash
qbridge capture circuit.json --seed 7 --repetitions 300 --option cpu_threads=4 --out runs/essai
```

```bash
qbridge verify runs/essai
```

Puis `qbridge replay`, `qbridge info` et `qbridge diff` (tous acceptent `--json`).
`verify`, `info` et `diff` n'exécutent **aucun circuit** — c'est vérifié par des
tests qui cassent volontairement les deux backends.

Codes de sortie : `0` conforme · `1` compatible seulement statistiquement ·
`2` divergent · `3` erreur · `4` indéterminé. Un verdict que la table ne connaît
pas retombe sur `3`, jamais sur `0` : une CI croirait à une réussite.

## Capture du versant classique

Un vrai calcul quantique est une boucle hybride : du code classique construit le
circuit, du code classique interprète les tirages. Sceller le milieu ne suffit
pas.

```python
ctx = capture_classical(callables={"reduce": ma_reduction}, input_data=probleme)
...
rapport = verify_source_unchanged(ctx, {"reduce": ma_reduction})
rapport.has_drift   # True si le code a change depuis le scellement
```

Chaque champ porte un **marqueur de preuve** — `captured`, `derived`,
`unavailable`, `not_applicable` — parce qu'un champ absent est autrement
ambigu entre « non capturé », « inapplicable » et « capture échouée ». Une
fonction définie dans un REPL ou un builtin est enregistrée `unavailable` avec
sa cause, jamais silencieusement ignorée.

C'est ce qui complète la garantie archivistique : tirages bruts + code de
réduction + environnement épinglé = tous les chiffres publiés regénérables,
sans aucune ressource quantique.

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
- ~~Rejeu archivistique~~ — **fait en v0.2** (`verify_archival`, `replay_record`)
- Export vers OpenQASM 3 / QIR
- ~~Capture du pré/post-traitement classique~~ — **fait** (`capture_classical`), pas encore intégrée au `Manifest`
- Compression du `circuit_json` (180 Ko pour 20 qubits × depth 16)
- Mode `EXPECTATION` de bout en bout (classé, mais pas encore dans `capture`)
- ~~Interface en ligne de commande~~ — **faite** (`qbridge capture/verify/replay/info/diff`)

## État de l'art

Aucun format existant ne scelle une exécution complète. QIR et OpenQASM 3
scellent le *programme* ; les `fake_provider` de Qiskit scellent un *appareil* ;
`results.json` de Braket scelle un *enregistrement de job* ;
`cirq_google.workflow` est le squelette le plus proche mais re-résout l'appareil
au moment du rejeu. Chaque emplacement « environnement » existant est une
référence par identifiant, pas un sceau.
