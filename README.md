# qbridge

**Prouver, dans cinq ans, qu'un calcul quantique a bien donné le résultat publié.**

---

## Le problème, en une minute

Vous lancez un calcul sur un ordinateur quantique. Vous publiez un chiffre.
Trois ans plus tard, quelqu'un demande : *« comment savez-vous que c'est bien
ce que la machine a répondu ? »*

Vous ne pouvez pas simplement relancer le calcul. Trois raisons, et aucune
n'est un détail technique :

- **La machine a changé.** Un ordinateur quantique se recalibre en permanence.
  Celui de 2026 n'est pas celui de 2029, même sous le même nom.
- **Le résultat n'est pas reproductible.** Le bruit physique n'est pas
  rejouable. Deux exécutions identiques donnent des tirages différents.
- **On ne peut pas photographier un état quantique.** Le théorème de
  non-clonage l'interdit, et la mesure le détruit. Il n'y a rien à sauvegarder.

Autrement dit : **la preuve ne peut pas être « je refais ».**

## Ce que fait qbridge

Il scelle la **recette** et les **résultats bruts** dans un dossier autonome,
signé, qui reste vérifiable sans machine quantique.

```
mon_experience/
  manifest.json    la recette : circuit, graine, état daté de l'appareil, environnement
  samples.npz      les tirages bruts — la seule donnée qu'on ne peut pas recréer
  signature.json   qui a scellé tout ça, et quand
```

Trois garanties, volontairement distinctes parce que les confondre est
exactement là où ce genre d'outil échoue :

| | Ce que ça prouve | Ce qu'il faut pour le vérifier |
|---|---|---|
| **Bit-exact** | même simulateur, mêmes octets | un simulateur |
| **Statistique** | distributions compatibles | une machine — c'est son plafond |
| **Archivistique** | les tirages archivés sont bien ceux qui ont été scellés | **rien du tout** |
| **Plausibilité physique** | les tirages sont cohérents avec la machine déclarée | un simulateur |

La troisième est la plus solide, et c'est celle qui compte vraiment. Elle ne
dépend d'aucun matériel, d'aucun simulateur, ni même de qsim installé.
Quelqu'un qui ouvre l'archive dans dix ans peut recalculer tous les chiffres
publiés à partir des tirages bruts — et vérifier qu'ils n'ont pas bougé.

## Un exemple concret

```python
from qbridge import capture, verify_archival, RunRecord
from qbridge.providers import from_google_calibration

# l'état réel d'une puce Willow de Google, 105 qubits, daté du 16 août 2024
etat_machine, avertissements = from_google_calibration("willow_pink")

run = capture(mon_circuit, backend="hardware-sim", seed=7,
              repetitions=2000, calibration=etat_machine)

RunRecord.from_capture(run).save("runs/mon_experience")
```

Et trois ans plus tard, sur n'importe quelle machine :

```python
rapport = verify_archival(RunRecord.load("runs/mon_experience"))
rapport.results_intact   # True — les tirages sont bien ceux qui ont été scellés
```

## Ce que ce projet a découvert en chemin

Ce ne sont pas des suppositions : chaque point a été **mesuré**, et les mesures
sont verrouillées par des tests qui échoueront si le comportement change.

**Le nombre de threads change les résultats.** Sur qsim, le simulateur de
Google, `cpu_threads=1` et `cpu_threads=2` donnent des bitstrings **différents**
dès qu'il y a une mesure en cours de circuit — à graine identique. La cause est
dans leur code C++ : le tirage aléatoire parcourt un tableau dont la longueur
*est* le nombre de threads. **Ce fait n'est documenté nulle part chez Google.**

**Un « instantané » de calibration n'en est pas un.** Le fichier d'IBM pour la
machine `ibm_fez`, étiqueté du 26 février 2025, contient 4 060 mesures dont la
plus ancienne date du **28 décembre 2024**. Soixante jours d'écart sous une
seule date. Qui le lit comme l'état de la machine à cette date se trompe de
deux mois sur certains paramètres.

**Les vraies machines sont très inégales.** Sur une chaîne de cinq qubits de
Willow, les temps de cohérence valent `[71.9, 71.9, 38.0, 74.6, 67.2] µs`. Un
qubit a la moitié de la durée de vie de ses voisins. Une moyenne effacerait
exactement ce qui compte.

## Ce que qbridge ne fait pas

Autant le dire tout de suite :

- **Il ne transporte pas d'état quantique.** C'est impossible, pas difficile.
- **Il ne rend pas les ordinateurs quantiques utiles.** C'est de la plomberie
  de provenance, pas un algorithme.
- **Il ne gère aucune clé.** Où vit votre clé privée, qui y accède, comment on
  révoque : hors périmètre, et c'est écrit tel quel dans le code.
- **Il ne corrige aucune erreur quantique.** Il enregistre le bruit, il ne le
  combat pas.

## Il a tourné sur une vraie machine

Le 1er septembre 2026, sur **`ibm_marrakesh`** — un QPU IBM de 156 qubits.
C'était le seul chemin du projet qui n'avait jamais été exercé : réseau, file
d'attente, récupération. Tout le reste était validé hors ligne.

Un état GHZ à trois qubits, 1024 tirages, 12 secondes :

| Résultat | Tirages | Part |
|---|---|---|
| `000` | 509 | 49,7 % |
| `111` | 487 | 47,6 % |
| les six autres | 28 | 2,7 % |

**97,3 % ± 0,5 %** de fidélité au GHZ. Sur un simulateur parfait ce serait
100 % : les 2,7 % qui manquent *sont* le bruit physique de la machine, et c'est
exactement pourquoi un verdict matériel est plafonné à « statistiquement
compatible » — jamais bit-exact.

L'archive a ensuite été vérifiée **sans aucune ressource quantique** :
manifeste intact, résultats intacts, 1024 tirages. C'est la garantie
archivistique qui fonctionne pour de vrai.

Une tentation, et pourquoi j'y ai résisté : `000` sort 509 fois contre 487 pour
`111`, ce qui ressemble à de la relaxation T1 (un `|1⟩` retombe vers `|0⟩`).
Vérification faite, l'écart vaut **0,7 σ** — c'est du bruit d'échantillonnage,
pas un effet. L'excès d'erreurs voisines de `111` plutôt que de `000` (20
contre 8) est plus intéressant à **2,3 σ**, mais reste sous le seuil : c'est
suggestif, pas démontré. Il faudrait plus de tirages pour trancher.

## Attraper un faux que rien d'autre n'attrape

Voici une archive fabriquée de toutes pièces : un GHZ **parfait**, 100 % sur
`000` et `111`, sur une machine dont la calibration scellée dit qu'elle ne peut
pas dépasser 97,4 %. Elle est scellée dans les règles — empreintes recalculées,
tout est cohérent.

```
$ qbridge verify runs/le_faux
  manifeste             : intact
  resultats             : intacts          <- l'integrite est irreprochable
  plausibilite physique : IMPLAUSIBLE
    predit par l'appareil : 97.38 %
    observe dans l'archive: 100.00 %
    ecart                 : 5.2 sigma
  code de sortie : 0
```

**Le code de sortie reste 0.** L'intégrité de ce faux est irréprochable : les
octets sont exactement ceux que son auteur a scellés. Aucun hash, aucune
signature ne peut le dire. Seule la physique le trahit.

C'est le quatrième verdict, et il ne coûte rien à vérifier — un simulateur, pas
un QPU. Sur la vraie archive d'`ibm_marrakesh` :

```
  plausibilite physique : PLAUSIBLE
    predit par l'appareil : 97.38 %
    observe dans l'archive: 98.05 %
    ecart                 : 1.3 sigma
    budget d'erreur       : measure 74 %, cz 20 %, sx 6 %
```

**Trois choses qu'il faut dire clairement :**

- **Ce n'est pas une preuve d'authenticité.** Un faussaire qui connaît la
  calibration peut fabriquer des tirages plausibles. Ce verdict attrape
  l'incohérence, pas la malveillance compétente ; la signature détachée reste
  le seul mécanisme d'opposabilité.
- **Sa résolution est limitée.** À 1024 tirages, l'écart type vaut 0,5 point :
  il faut environ **2 points** d'écart pour déclencher `IMPLAUSIBLE`. Un faux
  annonçant 98,5 % passerait. Plus de tirages, plus de finesse.
- **Il ne change pas le code de sortie par défaut.** La plausibilité est un
  jugement de physique, pas d'intégrité. Les confondre ferait lire « archive
  falsifiée » sur une exécution simplement surprenante. `--physique-stricte`
  le rend bloquant pour qui le veut.

Et quand il ne peut rien conclure, il le dit : pas de calibration scellée, trop
peu de tirages, ou un circuit dont la loi idéale couvre tous les bitstrings —
là, un résultat totalement dépolarisé tomberait déjà dans le support, et se
taire est la seule réponse honnête.

### Pourquoi compter les portes ne suffisait pas

Ma première version prédisait la fidélité à partir de `gate_counts` : *deux*
portes `cz`, *trois* mesures. Le problème est qu'un compte ne dit jamais
**lesquelles**. Sur `ibm_marrakesh`, les paires `cz` scellées vont de `1,65e-3`
à `3,63e-3` — un facteur **2,2**. Moyenner revient à supposer que la
transpilation a eu une chance moyenne.

À trois qubits c'est bénin. Mesuré sur la calibration réelle, avec le pire
placement possible :

| Portes à 2 qubits | Moyenne | Exact | Biais |
|---|---|---|---|
| 2 | 99,47 % | 99,28 % | 0,9 σ |
| 10 | 97,39 % | 96,43 % | 1,9 σ |
| **50** | 87,61 % | 83,37 % | **4,1 σ** |
| 200 | 58,92 % | 48,32 % | 6,9 σ |

**À cinquante portes à deux qubits, le seul choix moyenne/exact franchit le
seuil `IMPLAUSIBLE`.** L'approximation ne perdait pas en précision : elle
aurait déclaré fausse une archive parfaitement honnête.

L'archive scelle donc maintenant les opérations exactes, avec leurs qubits
physiques — `cz:q(0),q(1)` — dans le format même des clés de calibration, pour
que la prédiction soit une lecture directe et non une traduction. Une
traduction serait un endroit de plus où se tromper.

Les archives scellées avant ce changement retombent sur la moyenne, et le
rapport le **dit** au lieu de le taire.

### Il a attrapé IBM au passage

Le premier cas nominal du test échouait. Enquête faite : `FakeManilaV2`, le
simulateur factice livré par IBM, **déclare** des erreurs de lecture de 3,5 %,
2,2 % et 9,6 % — et rend 1024/1024 tirages **parfaits**. Il publie du bruit et
simule sans.

Ce n'était pas un défaut de qbridge : le verdict faisait son travail, sur
l'outil d'IBM lui-même. C'est un test à part entière maintenant.

## Ce que ce premier vrai job a révélé du code

Il a fallu un vrai QPU pour voir un trou que six vagues de relecture avaient
manqué : **l'archive ne scellait aucun état physique de la machine.**
`calibration_json` valait `None`. Elle disait « ça a tourné sur
`ibm_marrakesh` » — un nom — et rien d'autre. Ni T1/T2, ni erreurs de porte,
ni erreurs de lecture, ni **quels qubits physiques** avaient porté le calcul.

L'ironie est complète : ce projet avait construit `CalibrationSnapshot` avec
dates par paramètre, découvert l'étalement de 60 jours chez IBM — et le chemin
matériel n'en scellait rien. Le manifeste enregistrait même `qsim_gpu_mode`
pour une exécution qui n'a jamais touché qsim.

C'est corrigé. Une archive matérielle scelle désormais :

| | |
|---|---|
| **État daté de l'appareil** | T1, T2, erreurs de lecture et de porte, avec la date de **chaque** mesure |
| **Opérations exactes** | quelles portes, sur **quels** qubits physiques — pas des totaux |
| **Rien sur qsim** | l'empreinte du simulateur disparaît quand aucun simulateur n'a tourné |

Deux choix qui méritent d'être défendus :

**L'état est scellé après l'exécution, pas avant.** Le placement n'existe
qu'une fois la transpilation faite, et c'est lui qui dit quels qubits sceller.
`ibm_marrakesh` publie 156 qubits et 2 420 portes ; scellés en entier, ce
serait des centaines de kilo-octets pour un circuit qui en touche trois.

**Le placement entre dans le hash sémantique.** Sur cette machine, les erreurs
de lecture des qubits 0, 1 et 2 valent `9.5e-3`, `4.3e-3` et `5.7e-3` — plus du
double d'écart entre le meilleur et le pire. Deux placements différents du même
circuit logique ne sont pas la même expérience physique, et un hash qui les
confondrait mentirait.

Et une règle qui prime sur tout le reste : **une calibration illisible ne fait
jamais échouer une exécution réussie.** Perdre l'état d'appareil est fâcheux ;
perdre un job matériel qui a coûté du temps QPU et ne se rejoue pas le serait
bien plus. Le manifeste le dit alors, au lieu de le taire.

Ce que la machine a révélé d'elle-même au passage :

- La calibration « instantanée » de `ibm_marrakesh` s'étale sur **35 heures**.
  Même phénomène que les 60 jours d'`ibm_fez`, cette fois sur une machine
  vivante.
- Le qubit 0 affiche **T1 = 254 µs mais T2 = 39,5 µs**. Sa cohérence de phase
  s'effondre six fois plus vite que son énergie.
- **73,8 % du budget d'erreur** de ce circuit est de l'erreur de **lecture**,
  contre 20 % pour les portes à deux qubits. Ce n'est pas là qu'on regarde
  d'habitude.

## Sur la qualité de ce code

Ce projet a été écrit en une journée, et **six vagues de relecture adverse y ont
trouvé dix-neuf défauts exploitables**. Parmi eux :

- une collision d'empreinte permettant de **substituer une archive entière** ;
- une signature qui **ne couvrait pas les données** qu'elle prétendait protéger ;
- un test statistique qui certifiait « compatibles » deux jeux de tirages
  **sans aucun rapport**.

Tous sont corrigés, et chacun a un test qui **rejoue l'attaque d'origine**.

Le chiffre à retenir n'est pas dix-neuf, c'est ceci : **trois de ces défauts ont
été introduits par le correctif d'un défaut précédent.** Le code écrit sous
pression pour réparer autre chose est le plus dangereux du lot.

Conclusion honnête : c'est utilisable pour du travail personnel sur simulateur.
Ce n'est pas encore une pièce à verser dans un dossier opposable — le taux de
découverte de défauts n'a pas fléchi en six vagues, ce qui veut dire qu'il en
reste.

## Démarrer

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install -e .
```

```bash
.venv/Scripts/python.exe -m pytest -q
```

```bash
.venv/Scripts/python.exe examples/demo.py
```

**Note de langue** : le code, les commentaires et les messages d'erreur sont en
français. C'est un choix assumé, pas un oubli.

---

# Documentation technique

Ce qui suit s'adresse à quelqu'un qui veut modifier le harnais.

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
contexte classique scellé, et **deux** hashes.

### Pourquoi deux hashes

| Hash | Répond à | Couvre |
|---|---|---|
| `semantic_hash` | « le rejeu donnera-t-il le même résultat quantique ? » | circuit, seed, mode, options SEMANTIC et NUMERIC, noyau SIMD |
| `content_hash` | « ce document est-il intact ? » | **tous** les champs, y compris ceux que le premier ignore à dessein |

Un hash unique forcerait à choisir entre deux propriétés incompatibles :
détecter toute modification, et ne pas invalider un rejeu pour une raison qui
n'en est pas une. Changer le code de post-analyse modifie le document mais ne
peut pas modifier l'exécution quantique — le `content_hash` bouge, le
`semantic_hash` non, et le rejeu reste `BIT_EXACT`.

`content_hash` est calculé par énumération des champs de la dataclass, jamais
depuis une liste écrite à la main : un champ ajouté plus tard est scellé
d'office. Une liste manuelle est précisément ce qui avait laissé `circuit_json`
hors du sceau.

> **Portée exacte des hashes.** Ils sont **publics et déterministes** : ils
> prouvent qu'un document est cohérent avec lui-même, rien de plus. Quiconque
> modifie le manifeste les recalcule en deux lignes. Contre un adversaire, il
> faut une **signature** — voir plus bas.

## Signature

Un hash prouve l'intégrité. Il ne dit rien de l'origine. La signature ajoute la
seule chose qui manquait : **qui** a scellé.

```bash
qbridge keygen --key-id simon --private-out priv.key --public-out pub.key
```

```bash
qbridge sign runs/essai --key-id simon --private-key priv.key
```

```bash
qbridge verify runs/essai --public-key pub.key --key-id simon
```

### Deux algorithmes, deux garanties à ne pas confondre

| Algorithme | Ce que ça prouve | Opposable à un tiers |
|---|---|---|
| `hmac-sha256` | intégrité, pour le porteur de la clé | **non** — qui peut vérifier peut forger |
| `ed25519` | origine : seule la clé privée signe | **oui** |

HMAC ne dépend que de la bibliothèque standard et marchera encore dans dix ans,
mais il est **symétrique** : il ne vaut rien comme preuve face à un tiers. Seul
ed25519 rend une archive opposable ; il demande `pip install qbridge[sign]`, la
seule dépendance optionnelle du projet.

### Trois détails qui ne sont pas des détails

**La signature est détachée** (`signature.json`), jamais rangée dans le
manifeste : `content_hash` couvre tous les champs, donc l'y mettre changerait le
hash qu'elle signe.

**On signe l'ARCHIVE, pas la seule recette.** `sign_record` couvre le manifeste
*et* les tirages ; `sign_manifest` ne couvre que la recette et le dit. Signer le
seul `content_hash` du manifeste laissait `result_hash` — qui vit dans
`record.json` — hors de toute signature : on remplaçait `samples.npz`, on
recalculait `result_hash` en deux lignes, et l'archive se vérifiait « valide et
opposable ». Les bitstrings, la seule donnée non regénérable, étaient exactement
ce que la signature n'atteignait pas.

**On ne signe pas le hash nu**, mais un lien canonique
`{schéma, algorithme, key_id, portée, content_hash}`. Sans ces liens : une
signature HMAC présentée comme une ed25519 (substitution d'algorithme), une
signature de la clé A revendiquée pour la clé B (confusion de clés), ou une
signature de recette présentée comme couvrant une archive.

**Une signature authentique d'un *autre* document est refusée.** La vérification
contrôle **six** choses séparément — document cohérent, algorithme attendu,
identité de clé attendue, portée attendue, signature qui vise *ce* document,
cryptographie valide — et dit laquelle a échoué. Un vérificateur naïf qui
validerait seulement la cryptographie accepterait une signature vraie collée sur
un faux document.

### Ce que qbridge ne fait pas

Il ne gère aucune clé. Où vit la clé privée, qui y accède, comment on révoque :
hors périmètre, entièrement à votre charge. `keygen` écrit la clé privée en
`0o600` — ce que Windows n'applique qu'imparfaitement — et sans chiffrement au
repos. Aucune clé n'apparaît jamais dans un manifeste, une signature ou un
journal.

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

Puis `qbridge replay`, `qbridge info`, `qbridge diff`, `qbridge sign` et
`qbridge keygen`. `verify`, `info`, `diff`, `sign` et `keygen` acceptent
`--json` ; `capture` et `replay` ne l'acceptent pas encore.
`verify`, `info`, `diff`, `sign` et `keygen` n'exécutent **aucun circuit** —
c'est vérifié par des tests qui cassent volontairement les deux backends.

Codes de sortie : `0` conforme · `1` compatible seulement statistiquement ·
`2` divergent · `3` erreur · `4` indéterminé. Un verdict que la table ne connaît
pas retombe sur `3`, jamais sur `0` : une CI croirait à une réussite. Une erreur
d'usage (option mal orthographiée) rend `3` et non `2` — sinon une faute de
frappe se lirait comme une divergence physique.

## Capture du versant classique

Un vrai calcul quantique est une boucle hybride : du code classique construit le
circuit, du code classique interprète les tirages. Sceller le milieu ne suffit
pas.

```python
ctx = capture_classical(callables={"reduce": ma_reduction}, input_data=probleme)
run = capture(circuit, seed=7, repetitions=500, classical=ctx)   # scellé au manifeste
...
ctx = RunRecord.load("runs/essai").manifest.classical()
verify_source_unchanged(ctx, {"reduce": ma_reduction}).has_drift
```

Chaque champ porte un **marqueur de preuve** — `captured`, `derived`,
`unavailable`, `not_applicable` — parce qu'un champ absent est autrement
ambigu entre « non capturé », « inapplicable » et « capture échouée ». Une
fonction définie dans un REPL ou un builtin est enregistrée `unavailable` avec
sa cause, jamais silencieusement ignorée.

C'est ce qui complète la garantie archivistique : tirages bruts + code de
réduction + environnement épinglé = tous les chiffres publiés regénérables,
sans aucune ressource quantique.

## Le backend matériel — l'épreuve du protocole

`Backend` est un protocole abstrait, et il a maintenant une implémentation qui le
**met à l'épreuve** plutôt que de le confirmer. `hardware-sim` viole les trois
hypothèses confortables du simulateur :

| Hypothèse du simulateur | Ce que fait `hardware-sim` |
|---|---|
| on peut lire le vecteur d'état | `simulate()` lève `NotImplementedError` — no-cloning, mesure destructive |
| le résultat est reproductible bit-à-bit | `is_bit_exact_replayable()` → `False`, le verdict est plafonné |
| le backend est sans état | exige un **instantané de calibration daté**, scellé au manifeste |

Le test qui compte :

```
verdict : STATISTICALLY_COMPATIBLE
detail  : echantillons identiques (plafonné : le backend hardware-sim
          ne garantit pas la reproductibilité bit-pour-bit)
```

Les octets **coïncident** — qsim est déterministe sous le capot — et le harnais
refuse quand même `BIT_EXACT`. Le plafond est un **contrat du backend**, jamais
une déduction de ce qu'on observe. Déduire le verdict de l'observation donnerait,
sur une vraie machine, une conclusion que la physique ne permet pas.

`hardware-sim` s'appelle ainsi et pas `hardware` : c'est un substitut, et un
manifeste ne doit jamais laisser croire qu'il vient d'une vraie machine. Le jour
où une l'est, **seule `sample()` change** — elle postera un job au lieu d'appeler
qsim. Contrat, manifeste et verdicts restent identiques.

### Données de calibration réelles

`cirq-google` embarque les calibrations médianes de trois vrais processeurs.
Aucun compte, aucune autorisation, aucun réseau — ce sont des données ouvertes.

```python
from qbridge.providers import from_google_calibration
snap, avertissements = from_google_calibration("willow_pink")   # 105 qubits
```

| | `rainbow` (2021) | `willow_pink` (2024) |
|---|---|---|
| qubits | 23 | 105 |
| T1 moyen | 14,3 µs | **71,9 µs** |
| erreur de lecture | 2,98 % | **0,60 %** |

Un GHZ à 5 qubits sous le bruit dérivé de Willow donne **96,2 %** de
`|00000⟩+|11111⟩` sur 2000 tirages, et les T1 de la chaîne valent
`[71.9, 71.9, 38.0, 74.6, 67.2]` — le qubit (0,8) a la moitié du T1 de ses
voisins. C'est de l'inhomogénéité d'appareil réelle, et c'est exactement ce
qu'une moyenne aurait effacé.

**L'adaptateur ne devine rien en silence.** Google ne publie pas les durées de
porte dans cette calibration ; elles sont fournies par l'appelant et leur
provenance est **scellée dans l'instantané** (`unit` porte
`"ABSENT de la calibration Google"`), pour que personne dans cinq ans ne les
prenne pour des mesures du fournisseur. Les métriques manquantes sont signalées
plutôt qu'ignorées — `rainbow` utilise des `sqrt_iswap` et non des CZ, et
l'adaptateur le dit.

Deux différences réelles avec IBM, documentées : Google publie **un seul
horodatage** pour tout l'instantané (donc `temporal_spread_seconds()` vaut 0,
ce qui est correct pour une calibration médiane), et `readout_error` est ici la
moyenne des deux directions d'erreur, que le modèle de bruit de qbridge ne sait
pas distinguer.

> **Ceci n'est pas un accès au matériel.** Exécuter sur les vraies machines
> Google demande un partenariat de recherche approuvé. Une archive produite avec
> ces données a tourné sur `hardware-sim`, et le manifeste le déclare.

### IBM : 69 appareils, et une révélation sur les dates

```python
from qbridge.providers import from_ibm_backend
snap, avertissements = from_ibm_backend("FakeFez", qubits=range(6))
```

Le `fake_provider` de qiskit-ibm-runtime embarque **69 instantanés** de vrais
appareils. Contrairement à Google, IBM **date chaque paramètre séparément** — ce
qui a permis de vérifier pour de vrai la fonctionnalité pour laquelle
`DatedValue` a été conçu, et le résultat justifie le choix :

```
ibm_fez, last_update_date : 2025-02-26
4 060 mesures datées
la plus ancienne : 2024-12-28
la plus récente  : 2025-02-26
ÉTALEMENT        : 60 jours
```

**Un « instantané » étiqueté du 26 février contient des mesures qui s'étalent
sur deux mois.** Qui le traite comme l'état de l'appareil à cette date se trompe
de soixante jours sur certains paramètres. `temporal_spread_seconds()` l'expose
au lieu de le masquer derrière la seule `last_update_date`.

Deux autres choses que les vraies données IBM ont apprises au projet :

**Les unités se trompent en silence.** IBM publie T1 et T2 en *secondes*
(`4.88e-05` pour 48,8 µs) et `gate_length` en *nanosecondes*. Une erreur de
facteur produirait un modèle de bruit absurde sans rien signaler ; un test borne
les valeurs converties.

**Une moyenne de durées n'est représentative de rien.** Sur `ibm_fez` :

| porte | durée |
|---|---|
| `x`, `sx`, `rx`, `id` | 24 ns |
| `cz`, `rzz` | 84 ns |
| `rz` | **0 ns** (rotation virtuelle) |
| `reset` | **1584 ns** |

La moyenne vaut 210 ns, écrasée par le `reset`. Elle servait à convertir T1 en
relaxation, donc une porte X de 24 ns se voyait appliquer **neuf fois trop** de
bruit. `gate_length_for` prend désormais la durée de la porte précise — même
chaîne de repli que `gate_error_for`, parce que le raisonnement est le même.

### L'instantané de calibration

Un instantané n'est **pas** l'état d'un appareil à un instant : c'est un sac de
mesures datées séparément. Dans le `props_fez.json` d'IBM, T1 est mesuré le
26 février à 06h56 et `readout_error` le 24 février — deux jours d'écart dans un
même « instantané ». Chaque paramètre porte donc sa propre date, et
`temporal_spread_seconds()` expose l'écart au lieu de le cacher.

On scelle les **données datées**, jamais le modèle de bruit qu'on en dérive : la
dérivation est du code et le code change, les mesures sont un fait historique.
Cirq fait le même choix (`GoogleNoiseProperties` est sérialisable, pas le modèle),
et Aer le démontre par l'absurde — `NoiseModel.from_dict()` y est déprécié depuis
0.15, ce qui casserait toute archive ayant scellé un modèle.

La calibration entre dans le **`semantic_hash`**, contrairement au contexte
classique : l'état de l'appareil détermine bel et bien le résultat. Deux
exécutions sur des états d'appareil différents ne sont pas la même expérience.


```python
snap = synthetic_snapshot(qubits)          # ou chargé depuis un fournisseur
run = capture(circuit, backend="hardware-sim", seed=7,
              repetitions=400, calibration=snap)
```

Une calibration passée à un backend qui ne s'en sert pas est **refusée**, jamais
ignorée : l'ignorer laisserait croire qu'elle a influencé le résultat.

## Ce qui rend ce harnais différent

`Backend` est un protocole abstrait, implémenté par `qsim`, le simulateur de
référence de Cirq, et `hardware-sim`. **Le manifeste ne change pas de forme**
d'un backend à l'autre ; seul le verdict atteignable se dégrade.

| Verdict | Signification | Atteignable par |
|---|---|---|
| `BIT_EXACT` | octets identiques | simulateur, options identiques |
| `NUMERICALLY_EQUIVALENT` | infidélité ≤ 1e-4 | simulateur, options numériques différentes |
| `STATISTICALLY_COMPATIBLE` | χ² p ≥ 0.001 | **plafond du matériel réel** |
| `DIVERGENT` | au-delà | dérive à expliquer |
| `INDETERMINATE` | trop peu de tirages pour décider | régime clairsemé |

`INDETERMINATE` est placé **après** `DIVERGENT` dans l'énumération, pour qu'aucun
test du type `verdict <= NUMERICALLY_EQUIVALENT` ne l'accepte : ne pas pouvoir
conclure n'est pas une réussite. Il se déclenche quand l'effectif attendu par
bitstring tombe sous 5 — à 20 qubits avec 200 tirages, presque chaque bitstring
est unique et **aucun** test statistique ne peut distinguer deux distributions.
C'est une limite de complexité d'échantillonnage, pas un défaut réparable ; le
dire est la seule réponse honnête.

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

- ~~`HardwareBackend`, snapshots de calibration, fournisseur réel~~ — **fait** (`hardware-sim`, `CalibrationSnapshot`, `providers.google`)
- ~~Rejeu archivistique~~ — **fait en v0.2** (`verify_archival`, `replay_record`)
- Export vers OpenQASM 3 / QIR
- ~~Capture du pré/post-traitement classique~~ — **faite et scellée au `Manifest`**
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
