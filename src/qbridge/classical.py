"""Capture du versant classique d'une execution hybride.

Une charge de travail quantique reelle est une BOUCLE : du code classique
construit le circuit, du code classique interprete les tirages. `Manifest` ne
scelle aujourd'hui que le milieu. Ce module scelle les deux bouts.

Le raisonnement est celui deja etabli par `record.py` : les bitstrings bruts
sont la seule donnee non regenerable de la chaine. Tout chiffre publie en est
DERIVE par du code classique. Sceller les bitstrings, PLUS le code qui les a
reduits, PLUS l'environnement qui a execute ce code, c'est se donner de quoi
regenerer chaque chiffre publie sans aucune ressource quantique, indefiniment.

Un champ absent est ambigu : « pas capture », « sans objet » et « capture
echouee » ne sont pas la meme chose, et seule la derniere est une alarme.
Chaque champ porte donc un marqueur de preuve explicite (`Evidence`). On
n'avale jamais une erreur en silence : une capture qui a echoue doit se voir
dans le JSON scelle, pas y disparaitre.

Portee honnete de la capture de source : elle scelle le TEXTE d'une fonction,
pas la totalite de son comportement. Les trous connus sont documentes au fil
des fonctions concernees plutot que masques.
"""

from __future__ import annotations

import datetime as _dt
import importlib.metadata
import inspect
import linecache
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from qbridge.digest import sha256_of, sha256_of_array, sha256_of_text
from qbridge.fingerprint import environment_fingerprint

CLASSICAL_SCHEMA_VERSION = "1.0"


class Evidence(str, Enum):
    """Comment un champ a ete obtenu.

    Sans ce marqueur, un champ vide est indistinguable de trois situations qui
    n'appellent pas la meme reaction.
    """

    CAPTURED = "captured"
    """Obtenu tel quel a la source (texte, liste, valeur)."""

    DERIVED = "derived"
    """Calcule a partir d'une donnee qui n'est PAS conservee — un hash
    d'entree, par exemple : la donnee elle-meme reste chez son auteur."""

    UNAVAILABLE = "unavailable"
    """La capture a ete TENTEE et a echoue. C'est une alarme, pas un blanc."""

    NOT_APPLICABLE = "not_applicable"
    """Rien a capturer : le champ n'a pas de sens pour cette execution."""


# Raisons d'indisponibilite. Chaines stables : elles sont scellees dans le JSON
# et relues par des humains dans cinq ans.
REASON_BUILTIN = "builtin_or_c_implemented"
"""`inspect.getsource` leve TypeError sur un callable implemente en C."""

REASON_NOT_A_FUNCTION = "not_a_python_function"
"""functools.partial, ufunc numpy, instance appelable : pas de bloc source."""

REASON_LAMBDA = "lambda_source_ambiguous"
"""Pour un lambda, `getsource` rend la LIGNE qui le contient, pas le lambda :
`f = lambda x: x + 1` rend l'affectation entiere, et `g(lambda x: x, lambda
y: y)` rend le meme texte pour deux lambdas differents. Le texte est conserve
pour l'humain mais n'est PAS hache : il ne prouverait rien."""

REASON_NO_SOURCE = "source_unavailable"
"""REPL, `exec`, code compile sans fichier : `getsource` leve OSError."""

WARNING_DECORATED = "decorated: le code des decorateurs n'est pas scelle"
"""Un callable portant `__wrapped__` a ete decore. MESURE : `getsourcelines`
appelle `inspect.unwrap`, donc le texte rendu est celui de la fonction
d'ORIGINE (ligne `@decorateur` comprise), pas celui du wrapper. Le corps du
decorateur, lui, vit dans un autre module et n'est PAS capture : une
modification du decorateur passera inapercue. Le sceau est donc partiel."""


class _Absent:
    """Sentinelle : distingue « aucune entree fournie » de « l'entree None »."""

    def __repr__(self) -> str:  # pragma: no cover - confort de debogage
        return "<absent>"


ABSENT = _Absent()

_PEP503 = re.compile(r"[-_.]+")


# --------------------------------------------------------------------------
# Source des callables
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CallableCapture:
    """Ce qui a pu etre scelle d'UN callable classique.

    `source_hash` n'est renseigne QUE si `evidence == captured`. Un texte de
    source peut donc etre present sans hash (cas du lambda) : il est alors une
    indication pour l'humain, jamais une preuve.
    """

    role: str
    module: str
    qualname: str
    evidence: str
    source: Optional[str] = None
    source_hash: Optional[str] = None
    source_file: Optional[str] = None
    first_line: Optional[int] = None
    reason: Optional[str] = None
    warnings: Tuple[str, ...] = ()

    @property
    def is_verifiable(self) -> bool:
        """Vrai si ce champ permet de detecter une derive plus tard."""
        return self.evidence == Evidence.CAPTURED.value and self.source_hash is not None

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["warnings"] = list(self.warnings)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CallableCapture":
        d = dict(data)
        d["warnings"] = tuple(d.get("warnings") or ())
        return cls(**d)


def _read_source(
    fn: Callable[..., Any],
) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[str]]:
    """Lit le bloc source d'un callable. Rend (texte, fichier, ligne, erreur).

    Le callable est DEBALLE (`inspect.unwrap`) avant toute chose. MESURE :
    `inspect.getsourcelines` deballe deja de son cote, mais pas
    `inspect.getsourcefile` — sur une fonction decoree par `functools.wraps`,
    l'un rend le texte de la fonction d'origine et l'autre le fichier du
    DECORATEUR. Enregistrer les deux tels quels produirait un couple
    (fichier, ligne) qui ne designe pas le texte scelle.

    `linecache.checkcache` est appele avant lecture : sans lui, `getsource`
    sert une copie du fichier mise en cache a un instant arbitraire du
    processus, et le texte scelle dependrait de l'ordre des appels. Avec lui,
    le texte scelle est toujours celui du fichier SUR LE DISQUE.

    Trou assume : « sur le disque » n'est pas « ce que l'interpreteur execute ».
    Si le fichier a ete modifie apres l'import, la fonction vivante execute
    l'ancien bytecode alors que la capture rend le nouveau texte. Le bytecode
    n'expose pas son texte d'origine ; aucune correction n'est possible ici.
    """
    try:
        cible = inspect.unwrap(fn)
    except ValueError as exc:  # chaine `__wrapped__` circulaire
        return None, None, None, f"{type(exc).__name__}: {exc}"

    fichier: Optional[str] = None
    try:
        fichier = inspect.getsourcefile(cible)
    except TypeError:
        fichier = None
    if fichier:
        linecache.checkcache(fichier)
    try:
        lignes, premiere = inspect.getsourcelines(cible)
    except (OSError, TypeError, SyntaxError) as exc:
        return None, fichier, None, f"{type(exc).__name__}: {exc}"
    return "".join(lignes), fichier, int(premiere), None


def capture_callable(role: str, fn: Callable[..., Any]) -> CallableCapture:
    """Scelle la source d'un callable, ou dit pourquoi c'est impossible.

    Les modes d'echec reels sont distingues parce qu'ils n'appellent pas la
    meme reaction : un builtin ne sera jamais capturable, un lambda le
    deviendrait s'il etait nomme, une fonction du REPL le deviendrait si elle
    etait placee dans un fichier.
    """
    if not callable(fn):
        raise TypeError(f"Le role {role!r} ne designe pas un callable : {fn!r}")

    module = str(getattr(fn, "__module__", None) or "<unknown>")
    qualname = str(
        getattr(fn, "__qualname__", None)
        or getattr(fn, "__name__", None)
        or f"{type(fn).__qualname__}.__call__"
    )

    def indisponible(raison: str, **extra: Any) -> CallableCapture:
        return CallableCapture(
            role=role,
            module=module,
            qualname=qualname,
            evidence=Evidence.UNAVAILABLE.value,
            reason=raison,
            **extra,
        )

    # 1. Implemente en C, ou pas un objet dont Python garde un bloc source.
    if not (inspect.isfunction(fn) or inspect.ismethod(fn) or inspect.isclass(fn)):
        return indisponible(
            REASON_BUILTIN
            if inspect.isbuiltin(fn) or inspect.ismethoddescriptor(fn)
            else REASON_NOT_A_FUNCTION
        )

    # 2. Lambda : le texte rendu n'identifie pas le callable. On le garde pour
    #    l'humain, sans hash, et on declare la capture indisponible.
    if getattr(fn, "__name__", "") == "<lambda>":
        texte, fichier, ligne, _ = _read_source(fn)
        return indisponible(
            REASON_LAMBDA, source=texte, source_file=fichier, first_line=ligne
        )

    # 3. Cas normal : soit le fichier existe, soit il n'existe pas (REPL, exec).
    texte, fichier, ligne, erreur = _read_source(fn)
    if texte is None:
        return indisponible(f"{REASON_NO_SOURCE}: {erreur}", source_file=fichier)

    avertissements: List[str] = []
    if hasattr(fn, "__wrapped__"):
        avertissements.append(WARNING_DECORATED)

    return CallableCapture(
        role=role,
        module=module,
        qualname=qualname,
        evidence=Evidence.CAPTURED.value,
        source=texte,
        source_hash=sha256_of_text(texte),
        source_file=fichier,
        first_line=ligne,
        warnings=tuple(avertissements),
    )


# --------------------------------------------------------------------------
# Donnees d'entree
# --------------------------------------------------------------------------


def normalize_input(value: Any) -> Any:
    """Rend une structure arbitraire visible par `canonical_json`.

    Aucun hachage n'est re-implemente ici : un tableau est remplace par le
    condense de `sha256_of_array` (qui couvre deja dtype, forme et octets
    bruts), et le reste part tel quel dans `sha256_of`.
    """
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": sha256_of_array(value),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, np.generic):
        return {"__npscalar__": str(value.dtype), "value": value.item()}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"__bytes__": sha256_of_array(np.frombuffer(bytes(value), np.uint8))}
    if isinstance(value, dict):
        return {str(k): normalize_input(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_input(v) for v in value]
    if isinstance(value, (set, frozenset)):
        # Un ensemble n'a pas d'ordre : on trie sur l'empreinte des elements
        # deja normalises, seul ordre total disponible sur des types melanges.
        return {"__set__": sorted(sha256_of(normalize_input(v)) for v in value)}
    return value


def hash_input_data(data: Any) -> str:
    """SHA-256 deterministe d'une entree classique arbitraire.

    Leve `ValueError` sur un flottant non fini (`canonical_json` refuse NaN et
    Infinity par construction) et `TypeError` sur un objet que JSON ne sait pas
    representer. Les deux remontent telles quelles : c'est l'appelant qui
    decide s'il les enregistre comme indisponibilite ou s'il echoue.
    """
    return sha256_of(normalize_input(data))


# --------------------------------------------------------------------------
# Verrou d'environnement
# --------------------------------------------------------------------------


def installed_distributions() -> List[str]:
    """Distributions installees, forme `nom==version`, triee et dedupliquee.

    Le nom est normalise selon PEP 503 : le meme paquet peut apparaitre
    plusieurs fois quand plusieurs entrees de `sys.path` le contiennent, et
    `Pillow` / `pillow` designent la meme distribution. On garde la premiere
    vue, celle que l'import resoudrait.
    """
    vues: Dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        try:
            nom = getattr(dist, "name", None) or dist.metadata["Name"]
        except Exception:  # metadonnees illisibles : distribution ignoree
            continue
        if not nom:
            continue
        cle = _PEP503.sub("-", str(nom)).lower()
        if cle in vues:
            continue
        try:
            version = getattr(dist, "version", None) or "unknown"
        except Exception:
            version = "unknown"
        vues[cle] = f"{cle}=={version}"
    return [vues[c] for c in sorted(vues)]


def environment_lock(*, include_quantum_fingerprint: bool = True) -> Dict[str, Any]:
    """Verrou d'environnement du versant classique.

    Ne REDEFINIT pas les bases de plateforme : `environment_fingerprint()` de
    `fingerprint.py` couvre deja version courte de Python, implementation,
    plateforme, machine, processeur, versions cirq/qsimcirq/numpy et noyau
    SIMD. On COMPOSE avec lui plutot que de le recopier, et on ajoute
    exactement ce qu'il ne couvre pas :

    - la liste COMPLETE des distributions installees, seule chose qui permette
      de reconstituer l'interpreteur qui a reduit les tirages ;
    - l'identite exacte de cet interpreteur (chemin, prefixe, banniere de
      version avec le compilateur), que l'empreinte resume a un numero.

    Les cles d'erreur sont toujours presentes, a `None` quand tout va bien :
    une cle absente serait ambigue, ce que ce module refuse par principe.
    """
    lock: Dict[str, Any] = {
        "interpreter": {
            "executable": sys.executable or "",
            "prefix": sys.prefix,
            "version_full": " ".join(sys.version.split()),
        },
        "quantum_fingerprint_source": "qbridge.fingerprint.environment_fingerprint",
        "quantum_fingerprint": None,
        "quantum_fingerprint_error": None,
        "distributions": [],
        "distribution_count": 0,
        "distributions_error": None,
    }

    try:
        distributions = installed_distributions()
    except Exception as exc:
        lock["distributions_error"] = f"{type(exc).__name__}: {exc}"
    else:
        lock["distributions"] = distributions
        lock["distribution_count"] = len(distributions)

    if include_quantum_fingerprint:
        try:
            lock["quantum_fingerprint"] = environment_fingerprint()
        except Exception as exc:
            lock["quantum_fingerprint_error"] = f"{type(exc).__name__}: {exc}"

    return lock


# --------------------------------------------------------------------------
# Le contexte classique
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassicalContext:
    """Les deux bouts classiques d'une execution hybride, scelles.

    Concu pour tenir dans UN champ JSON de `Manifest` : `to_dict()` ne rend que
    des types JSON et `from_dict()` reconstruit l'objet a l'identique.

    `context_hash` couvre tout sauf `created_at` — l'heure de la capture ne
    change pas ce qui a ete scelle. Il inclut en revanche `environment`, a la
    difference du `semantic_hash` de `Manifest` qui l'exclut : pour un rejeu
    quantique l'environnement EXPLIQUE une divergence, alors qu'ici il EST la
    chose a reproduire, puisque c'est lui qui recalculera les chiffres.
    """

    schema_version: str
    created_at: str
    callables: Dict[str, CallableCapture]
    input_hash: Optional[str]
    input_kind: Optional[str]
    input_error: Optional[str]
    environment: Dict[str, Any]
    evidence: Dict[str, str]
    context_hash: str = field(default="")

    def _compute_context_hash(self) -> str:
        """Hash de tout ce qui a ete scelle, hors horodatage."""
        return sha256_of(
            {
                "schema_version": self.schema_version,
                "callables": {r: c.to_dict() for r, c in self.callables.items()},
                "input_hash": self.input_hash,
                "input_kind": self.input_kind,
                "input_error": self.input_error,
                "environment": self.environment,
                "evidence": self.evidence,
            }
        )

    def verify_integrity(self) -> None:
        """Refuse un contexte dont le hash ne correspond plus au contenu."""
        attendu = self._compute_context_hash()
        if attendu != self.context_hash:
            raise ValueError(
                "Echec du controle d'integrite du contexte classique : il a ete "
                f"modifie. hash stocke={self.context_hash[:16]}... "
                f"hash recalcule={attendu[:16]}..."
            )

    def verifiable_roles(self) -> Tuple[str, ...]:
        """Roles dont la source scellee permettra de detecter une derive."""
        return tuple(sorted(r for r, c in self.callables.items() if c.is_verifiable))

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["callables"] = {r: c.to_dict() for r, c in self.callables.items()}
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClassicalContext":
        d = dict(data)
        d["callables"] = {
            r: CallableCapture.from_dict(c)
            for r, c in (d.get("callables") or {}).items()
        }
        return cls(**d)


def capture_classical(
    *,
    prepare: Optional[Callable[..., Any]] = None,
    reduce: Optional[Callable[..., Any]] = None,
    callables: Optional[Mapping[str, Callable[..., Any]]] = None,
    input_data: Any = ABSENT,
    include_quantum_fingerprint: bool = True,
) -> ClassicalContext:
    """Scelle le versant classique : code, entree, environnement.

    `prepare` et `reduce` sont les deux roles usuels de la boucle hybride —
    celui qui construit le circuit et celui qui transforme les tirages en
    chiffre publie. `callables` permet d'en ajouter d'autres sous des noms
    libres (post-traitement, calibration, filtrage).

    `input_data` n'est PAS conserve, seulement hache : la donnee reste chez son
    auteur, le contexte ne porte que de quoi prouver qu'on lui redonne la meme.
    D'ou le marqueur `derived` et non `captured`.

    Une entree impossible a hacher (NaN, objet non serialisable) ne fait pas
    echouer la capture : elle est enregistree `unavailable`, avec sa cause dans
    `input_error`. Perdre tout le contexte parce qu'un champ resiste serait
    pire que le signaler.
    """
    roles: Dict[str, Callable[..., Any]] = {}
    if prepare is not None:
        roles["prepare"] = prepare
    if reduce is not None:
        roles["reduce"] = reduce
    for nom, fn in (callables or {}).items():
        if str(nom) in roles:
            raise ValueError(f"Role {str(nom)!r} fourni deux fois.")
        roles[str(nom)] = fn

    prises = {r: capture_callable(r, fn) for r, fn in roles.items()}
    preuves: Dict[str, str] = {}
    if not prises:
        preuves["callables"] = Evidence.NOT_APPLICABLE.value
    elif all(p.evidence == Evidence.CAPTURED.value for p in prises.values()):
        preuves["callables"] = Evidence.CAPTURED.value
    else:
        # Agregat volontairement pessimiste : un seul role non scelle suffit a
        # rendre le lot incapable de tout regenerer.
        preuves["callables"] = Evidence.UNAVAILABLE.value

    input_hash: Optional[str] = None
    input_kind: Optional[str] = None
    input_error: Optional[str] = None
    if input_data is ABSENT:
        preuves["input_data"] = Evidence.NOT_APPLICABLE.value
    else:
        input_kind = type(input_data).__name__
        try:
            input_hash = hash_input_data(input_data)
        except (TypeError, ValueError) as exc:
            input_error = f"{type(exc).__name__}: {exc}"
            preuves["input_data"] = Evidence.UNAVAILABLE.value
        else:
            preuves["input_data"] = Evidence.DERIVED.value

    environnement = environment_lock(
        include_quantum_fingerprint=include_quantum_fingerprint
    )
    preuves["interpreter"] = Evidence.CAPTURED.value
    preuves["distributions"] = (
        Evidence.CAPTURED.value
        if environnement["distributions_error"] is None
        else Evidence.UNAVAILABLE.value
    )
    if not include_quantum_fingerprint:
        preuves["quantum_fingerprint"] = Evidence.NOT_APPLICABLE.value
    elif environnement["quantum_fingerprint"] is not None:
        preuves["quantum_fingerprint"] = Evidence.CAPTURED.value
    else:
        preuves["quantum_fingerprint"] = Evidence.UNAVAILABLE.value

    brut = ClassicalContext(
        schema_version=CLASSICAL_SCHEMA_VERSION,
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        callables=prises,
        input_hash=input_hash,
        input_kind=input_kind,
        input_error=input_error,
        environment=environnement,
        evidence=preuves,
    )
    return ClassicalContext(
        **{**brut.__dict__, "context_hash": brut._compute_context_hash()}
    )


# --------------------------------------------------------------------------
# Detection de derive
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceDriftReport:
    """Verdict de conformite du CODE, sans aucune ressource quantique.

    `unverifiable` n'est pas `unchanged` : un role dont la source n'a jamais pu
    etre scellee ne peut pas etre declare intact. Les confondre serait
    exactement le mensonge que ce module existe pour eviter.
    """

    unchanged: Tuple[str, ...]
    drifted: Tuple[str, ...]
    unverifiable: Dict[str, str]
    missing: Tuple[str, ...]
    unknown: Tuple[str, ...]
    detail: Dict[str, Dict[str, Any]]

    @property
    def has_drift(self) -> bool:
        return bool(self.drifted)

    @property
    def fully_verified(self) -> bool:
        """Vrai seulement si CHAQUE role scelle a ete confronte et retrouve."""
        return not (self.drifted or self.unverifiable or self.missing)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unchanged": list(self.unchanged),
            "drifted": list(self.drifted),
            "unverifiable": dict(self.unverifiable),
            "missing": list(self.missing),
            "unknown": list(self.unknown),
            "detail": {r: dict(d) for r, d in self.detail.items()},
        }


def verify_source_unchanged(
    context: ClassicalContext,
    callables: Mapping[str, Callable[..., Any]],
) -> SourceDriftReport:
    """Confronte les callables VIVANTS a ce qui avait ete scelle.

    C'est ce qui dit a quelqu'un, dans cinq ans, que le code de reduction qu'il
    s'apprete a executer n'est pas celui qui a produit le chiffre publie.
    Aucune ressource quantique n'est consommee : on re-hache du texte.
    """
    inchanges: List[str] = []
    derives: List[str] = []
    invalides: Dict[str, str] = {}
    manquants: List[str] = []
    detail: Dict[str, Dict[str, Any]] = {}

    for role in sorted(context.callables):
        scelle = context.callables[role]
        if role not in callables:
            manquants.append(role)
            detail[role] = {
                "sealed_hash": scelle.source_hash,
                "current_hash": None,
                "status": "missing",
            }
            continue

        actuel = capture_callable(role, callables[role])
        entree: Dict[str, Any] = {
            "sealed_hash": scelle.source_hash,
            "current_hash": actuel.source_hash,
            "sealed_qualname": scelle.qualname,
            "current_qualname": actuel.qualname,
            "warnings": list(scelle.warnings) + list(actuel.warnings),
        }

        if not scelle.is_verifiable:
            invalides[role] = f"scelle {scelle.evidence}: {scelle.reason}"
            entree["status"] = "unverifiable"
        elif not actuel.is_verifiable:
            invalides[role] = f"vivant {actuel.evidence}: {actuel.reason}"
            entree["status"] = "unverifiable"
        elif actuel.source_hash == scelle.source_hash:
            inchanges.append(role)
            entree["status"] = "unchanged"
        else:
            derives.append(role)
            entree["status"] = "drifted"
        detail[role] = entree

    inconnus = tuple(sorted(set(map(str, callables)) - set(context.callables)))
    for role in inconnus:
        detail[role] = {
            "sealed_hash": None,
            "current_hash": capture_callable(role, callables[role]).source_hash,
            "status": "unknown",
        }

    return SourceDriftReport(
        unchanged=tuple(inchanges),
        drifted=tuple(derives),
        unverifiable=invalides,
        missing=tuple(manquants),
        unknown=inconnus,
        detail=detail,
    )
