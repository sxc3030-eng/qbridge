"""Interface en ligne de commande de qbridge.

Cinq sous-commandes, deux familles :

- `capture` et `replay` executent un circuit. Elles consomment des ressources
  de calcul.
- `verify`, `info` et `diff` n'en consomment aucune : elles ne lisent que des
  octets deja poses sur le disque. C'est delibere — la garantie archivistique
  doit rester exercable le jour ou plus rien ne s'execute.

Nuance a garder honnete : « n'execute rien » ne veut pas dire « n'importe
rien ». `qbridge.record` importe `qbridge.capture`, qui importe le registre des
backends, qui importe qsimcirq. Les commandes archivistiques ne touchent aucun
simulateur, mais elles chargent encore le module. Les imports du domaine sont
donc faits DANS les fonctions et jamais au niveau du module : `qbridge --help`
n'a besoin ni de cirq ni de qsimcirq.

Aucune dependance hors bibliotheque standard. Le projet implemente son propre
chi2 plutot que de dependre de scipy ; ce n'est pas le moment d'y ajouter un
parseur d'arguments tiers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Codes de sortie
# --------------------------------------------------------------------------

EXIT_OK = 0
"""Succes : dossier intact, enregistrements identiques, rejeu conforme."""

EXIT_MISMATCH = 1
"""Ecart de niveau avertissement : falsification detectee (`verify`),
enregistrements differents (`diff`), rejeu STATISTICALLY_COMPATIBLE."""

EXIT_DIVERGENT = 2
"""Reserve a `replay` : verdict DIVERGENT, la derive est a expliquer."""

EXIT_ERROR = 3
"""Panne operationnelle : dossier absent, JSON illisible, backend ou option
inconnus. Distinct de 1 et 2 pour qu'un script sache separer « le harnais n'a
pas pu travailler » de « le harnais a travaille et le resultat est mauvais »."""

EXIT_INDETERMINATE = 4
"""Reserve a `replay` : verdict INDETERMINATE.

Code distinct de 1 et de 2 a dessein. « Aucun test ne peut conclure dans ce
regime » n'est ni un avertissement statistique (1, ou le test a conclu qu'on
est compatible) ni une divergence (2, ou le test a conclu qu'on ne l'est pas).
Le confondre avec l'un des deux ferait exactement l'erreur que le verdict a ete
ajoute pour empecher."""

_LABEL_WIDTH = 22


class CliError(Exception):
    """Erreur destinee a l'utilisateur : message lisible, jamais de traceback."""


# --------------------------------------------------------------------------
# Petits utilitaires de presentation
# --------------------------------------------------------------------------


def _line(label: str, value: Any) -> str:
    """Une ligne « etiquette : valeur » alignee."""
    return f"  {label:<{_LABEL_WIDTH}}: {value}"


def _short(value: Any) -> Any:
    """Abrege un hash hexadecimal pour l'affichage texte."""
    if isinstance(value, str) and len(value) == 64:
        return f"{value[:16]}..."
    return value


def _format_options(options: Dict[str, Any]) -> str:
    """Rend un dict d'options sur une ligne, ou `(aucune)` s'il est vide."""
    if not options:
        return "(aucune)"
    return ", ".join(f"{key}={options[key]!r}" for key in sorted(options))


def _error_message(exc: BaseException) -> str:
    """Message propre pour une exception du coeur.

    `str()` sur un KeyError entoure de guillemets le message deja formate par
    le coeur ; on preleve directement l'argument.
    """
    if isinstance(exc, KeyError) and exc.args and isinstance(exc.args[0], str):
        return exc.args[0]
    return str(exc)


def _emit(payload: Dict[str, Any], as_json: bool, text: List[str]) -> None:
    """Ecrit la sortie, en JSON canonique ou en texte."""
    if as_json:
        from qbridge.digest import canonical_json

        print(canonical_json(payload))
    else:
        print("\n".join(text))


def _fail(message: str, as_json: bool, code: int = EXIT_ERROR) -> int:
    """Signale une panne et renvoie le code de sortie.

    En mode `--json`, l'erreur part sur stdout sous forme d'objet JSON : un
    consommateur automatique lit un seul flux, succes comme echec.
    """
    if as_json:
        from qbridge.digest import canonical_json

        print(canonical_json({"error": message, "exit_code": code}))
    else:
        print(f"qbridge : {message}", file=sys.stderr)
    return code


# --------------------------------------------------------------------------
# Analyse des arguments composites
# --------------------------------------------------------------------------


def _coerce(raw: str) -> Any:
    """Type une valeur d'option : booleen, entier, flottant, sinon texte."""
    lowered = raw.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        value = float(raw)
    except ValueError:
        return raw
    # NaN et Infinity ne sont pas du JSON valide : `canonical_json` les refuse.
    # Les laisser entrer produirait un manifeste qu'on ne pourrait plus relire.
    if value != value or value in (float("inf"), float("-inf")):
        raise CliError(
            f"Valeur d'option non representable en JSON : {raw!r}. "
            "NaN et Infinity casseraient le hash du manifeste."
        )
    return value


def _parse_option(spec: str) -> Tuple[str, Any]:
    """Decoupe `cle=valeur` et type la valeur."""
    if "=" not in spec:
        raise CliError(f"Option mal formee : {spec!r}. Attendu : cle=valeur.")
    key, _, raw = spec.partition("=")
    key = key.strip()
    raw = raw.strip()
    if not key:
        raise CliError(f"Option sans nom : {spec!r}. Attendu : cle=valeur.")
    if not raw:
        raise CliError(f"Option sans valeur : {spec!r}. Attendu : cle=valeur.")
    return key, _coerce(raw)


def _collect_options(specs: Optional[List[str]]) -> Dict[str, Any]:
    """Assemble les `--option` repetes en un dict, en refusant les doublons."""
    options: Dict[str, Any] = {}
    for spec in specs or []:
        key, value = _parse_option(spec)
        if key in options:
            raise CliError(f"Option repetee : {key!r}.")
        options[key] = value
    return options


def _check_option_names(options: Dict[str, Any]) -> None:
    """Refuse toute option absente de la table des niveaux.

    Une option ignoree en silence est exactement ce qui rend un rejeu
    faussement rassurant : on echoue tot, avec la liste des noms connus.
    """
    from qbridge.tiers import known_options

    known = known_options()
    unknown = sorted(key for key in options if key not in known)
    if unknown:
        raise CliError(
            f"Option(s) inconnue(s) : {', '.join(unknown)}. "
            f"Options classees : {', '.join(sorted(known))}"
        )


def _check_backend(name: str) -> None:
    """Refuse un backend absent du registre."""
    from qbridge.backends import BACKENDS

    if name not in BACKENDS:
        raise CliError(
            f"Backend inconnu : {name!r}. Disponibles : {', '.join(sorted(BACKENDS))}"
        )


# --------------------------------------------------------------------------
# Chargement depuis le disque
# --------------------------------------------------------------------------


def _load_record(directory: str) -> Any:
    """Charge un `RunRecord`, en traduisant toute panne en `CliError`."""
    from qbridge.record import RunRecord

    path = Path(directory)
    if not path.exists():
        raise CliError(f"Dossier introuvable : {path}")
    if not path.is_dir():
        raise CliError(f"Ce n'est pas un dossier : {path}")
    for name in ("record.json", "manifest.json"):
        if not (path / name).is_file():
            raise CliError(f"Fichier manquant dans {path} : {name}")
    try:
        return RunRecord.load(path)
    except json.JSONDecodeError as exc:
        raise CliError(f"JSON illisible dans {path} : {exc}") from None
    except KeyError as exc:
        raise CliError(
            f"Champ absent de l'enregistrement {path} : {_error_message(exc)}"
        ) from None
    except (OSError, ValueError, TypeError) as exc:
        raise CliError(f"Enregistrement illisible dans {path} : {exc}") from None


def _load_circuit(circuit_path: str) -> Any:
    """Charge un circuit depuis un fichier JSON produit par `cirq.to_json`."""
    import cirq

    path = Path(circuit_path)
    if not path.is_file():
        raise CliError(f"Fichier de circuit introuvable : {path}")
    try:
        obj = cirq.read_json(str(path))
    except Exception as exc:  # cirq leve des types varies sur un JSON casse
        raise CliError(
            f"Circuit JSON illisible ({path}) : {type(exc).__name__} : {exc}"
        ) from None
    if not isinstance(obj, cirq.Circuit):
        raise CliError(
            f"Le fichier {path} contient un {type(obj).__name__}, pas un cirq.Circuit."
        )
    return obj


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------


def _cmd_capture(args: argparse.Namespace) -> int:
    """Execute un circuit et scelle le resultat dans un dossier."""
    from qbridge import capture
    from qbridge.record import RunRecord

    _check_backend(args.backend)
    options = _collect_options(args.option)
    _check_option_names(options)
    circuit = _load_circuit(args.circuit)

    try:
        run = capture(
            circuit,
            backend=args.backend,
            seed=args.seed,
            repetitions=args.repetitions,
            options=options,
        )
    except (KeyError, ValueError) as exc:
        raise CliError(_error_message(exc)) from None

    record = RunRecord.from_capture(run)
    try:
        out = record.save(args.out)
    except OSError as exc:
        raise CliError(f"Ecriture impossible dans {args.out} : {exc}") from None

    manifest = record.manifest
    print(
        "\n".join(
            [
                f"Capture scellee dans {out}",
                _line("hash semantique", manifest.semantic_hash),
                _line("mode", manifest.mode),
                _line(
                    "backend",
                    f"{manifest.backend_name} {manifest.backend_version}",
                ),
                _line("seed", manifest.seed),
                _line(
                    "repetitions",
                    manifest.repetitions
                    if manifest.repetitions is not None
                    else "(aucune : vecteur d'etat)",
                ),
                _line("hash du resultat", record.result_hash),
                "  options par niveau :",
                _line("    semantic", _format_options(manifest.semantic_options)),
                _line("    numeric", _format_options(manifest.numeric_options)),
                _line(
                    "    performance", _format_options(manifest.performance_options)
                ),
            ]
        )
    )
    return EXIT_OK


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------


def _cmd_verify(args: argparse.Namespace) -> int:
    """Verifie un dossier sans consommer la moindre ressource quantique."""
    from qbridge import verify_archival

    record = _load_record(args.directory)
    report = verify_archival(record)
    intact = report.manifest_intact and report.results_intact
    code = EXIT_OK if intact else EXIT_MISMATCH

    payload = {
        "directory": str(Path(args.directory)),
        "manifest_intact": report.manifest_intact,
        "results_intact": report.results_intact,
        "measurement_keys": list(report.measurement_keys),
        "total_shots": report.total_shots,
        "semantic_hash": record.manifest.semantic_hash,
        "result_hash": record.result_hash,
        "detail": report.detail,
        "executed_circuit": False,
        "exit_code": code,
    }
    text = [
        f"Verification archivistique de {Path(args.directory)}",
        _line("manifeste", "intact" if report.manifest_intact else "FALSIFIE"),
        _line("resultats", "intacts" if report.results_intact else "FALSIFIES"),
        _line(
            "controles",
            "hash semantique recalcule depuis le manifeste, "
            "hash des bitstrings recalcule depuis samples.npz",
        ),
        _line("cles de mesure", ", ".join(report.measurement_keys) or "(aucune)"),
        _line("tirages archives", report.total_shots),
        _line("detail", report.detail),
        "  Aucun circuit n'a ete execute.",
    ]
    _emit(payload, args.json_output, text)
    return code


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------


def _exit_code_for(verdict: Any) -> int:
    """Traduit un verdict en code de sortie POSIX.

    Le repli est EXIT_ERROR et non EXIT_OK, deliberement : un verdict que cette
    table ne connait pas est un verdict que l'appelant ne peut pas interpreter.
    Le faire tomber sur 0 signalerait une reussite qui n'a pas ete constatee —
    et un script d'integration continue le croirait.
    """
    from qbridge.verdict import Verdict

    return {
        Verdict.BIT_EXACT: EXIT_OK,
        Verdict.NUMERICALLY_EQUIVALENT: EXIT_OK,
        Verdict.STATISTICALLY_COMPATIBLE: EXIT_MISMATCH,
        Verdict.DIVERGENT: EXIT_DIVERGENT,
        Verdict.INDETERMINATE: EXIT_INDETERMINATE,
    }.get(verdict, EXIT_ERROR)


def _cmd_replay(args: argparse.Namespace) -> int:
    """Rejoue un enregistrement et compare au resultat archive."""
    from qbridge import replay_record

    record = _load_record(args.directory)
    options = _collect_options(args.option)
    _check_option_names(options)
    if args.backend is not None:
        _check_backend(args.backend)

    try:
        report = replay_record(
            record,
            backend=args.backend,
            override_performance=options or None,
        )
    except (KeyError, ValueError) as exc:
        raise CliError(_error_message(exc)) from None

    code = _exit_code_for(report.verdict)
    comparison = report.comparison
    lines = [
        f"Rejeu de {Path(args.directory)}",
        _line("verdict", report.verdict.name),
        _line("detail", report.detail),
        _line("backend d'origine", report.original_backend),
        _line("backend de rejeu", report.replay_backend),
        _line("noyau SIMD", "CHANGE" if report.kernel_changed else "inchange"),
    ]
    if comparison.infidelity is not None:
        lines.append(_line("infidelite", f"{comparison.infidelity:.3e}"))
    if comparison.max_abs_delta is not None:
        lines.append(_line("max|delta|", f"{comparison.max_abs_delta:.3e}"))
    if comparison.p_value is not None:
        lines.append(_line("chi2 p", f"{comparison.p_value:.4g}"))
    if report.environment_drift:
        lines.append("  derive d'environnement :")
        for key in sorted(report.environment_drift):
            drift = report.environment_drift[key]
            lines.append(
                _line(f"    {key}", f"{drift['capture']!r} -> {drift['replay']!r}")
            )
    else:
        lines.append(_line("environnement", "identique a la capture"))
    lines.append(_line("code de sortie", code))
    print("\n".join(lines))
    return code


# --------------------------------------------------------------------------
# info
# --------------------------------------------------------------------------


def _cmd_info(args: argparse.Namespace) -> int:
    """Resume un enregistrement. N'execute rien."""
    record = _load_record(args.directory)
    manifest = record.manifest

    keys = sorted(record.samples) if record.samples else []
    shots = {key: int(record.samples[key].shape[0]) for key in keys}

    payload = {
        "directory": str(Path(args.directory)),
        "record_schema_version": record.schema_version,
        "manifest_schema_version": manifest.schema_version,
        "created_at": manifest.created_at,
        "semantic_hash": manifest.semantic_hash,
        "circuit_hash": manifest.circuit_hash,
        "mode": manifest.mode,
        "backend": {
            "name": manifest.backend_name,
            "version": manifest.backend_version,
        },
        "seed": manifest.seed,
        "repetitions": manifest.repetitions,
        "noise": manifest.noise_json is not None,
        "options": {
            "semantic": manifest.semantic_options,
            "numeric": manifest.numeric_options,
            "performance": manifest.performance_options,
        },
        "kernel": manifest.kernel,
        "environment": manifest.environment,
        "measurement_keys": keys,
        "shots_per_key": shots,
        "total_shots": sum(shots.values()),
        "result_hash": record.result_hash,
        "state_vector_hash": record.state_vector_hash,
    }

    text = [
        f"Enregistrement {Path(args.directory)}",
        _line(
            "schema",
            f"record {record.schema_version} / manifest {manifest.schema_version}",
        ),
        _line("capture le", manifest.created_at),
        _line("mode", manifest.mode),
        _line("backend", f"{manifest.backend_name} {manifest.backend_version}"),
        _line("seed", manifest.seed),
        _line(
            "repetitions",
            manifest.repetitions
            if manifest.repetitions is not None
            else "(aucune : vecteur d'etat)",
        ),
        _line("bruit", "present" if manifest.noise_json is not None else "aucun"),
        _line("hash semantique", manifest.semantic_hash),
        _line("hash du circuit", manifest.circuit_hash),
        _line("hash du resultat", record.result_hash),
        _line(
            "hash du vecteur",
            record.state_vector_hash if record.state_vector_hash else "(non archive)",
        ),
        "  options par niveau :",
        _line("    semantic", _format_options(manifest.semantic_options)),
        _line("    numeric", _format_options(manifest.numeric_options)),
        _line("    performance", _format_options(manifest.performance_options)),
        "  noyau SIMD :",
    ]
    for key in sorted(manifest.kernel):
        text.append(_line(f"    {key}", manifest.kernel[key]))
    text.append("  environnement :")
    for key in sorted(manifest.environment):
        text.append(_line(f"    {key}", manifest.environment[key]))
    text.append("  mesures :")
    if keys:
        for key in keys:
            text.append(_line(f"    {key}", f"{shots[key]} tirages"))
        text.append(_line("    total", f"{sum(shots.values())} tirages"))
    else:
        text.append(_line("    (aucune)", "mode vecteur d'etat"))

    _emit(payload, args.json_output, text)
    return EXIT_OK


# --------------------------------------------------------------------------
# diff
# --------------------------------------------------------------------------


def _field_diff(a: Any, b: Any) -> Dict[str, Any]:
    """Une entree de comparaison : les deux valeurs et leur egalite."""
    return {"a": a, "b": b, "equal": a == b}


def _dict_drift(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Ecarts champ par champ entre deux dicts, cles absentes comprises."""
    return {
        key: {"a": a.get(key), "b": b.get(key)}
        for key in sorted(set(a) | set(b))
        if a.get(key) != b.get(key)
    }


def _cmd_diff(args: argparse.Namespace) -> int:
    """Compare deux enregistrements. Purement classique : rien n'est execute."""
    from qbridge.digest import sha256_of_text

    record_a = _load_record(args.directory_a)
    record_b = _load_record(args.directory_b)
    man_a, man_b = record_a.manifest, record_b.manifest

    # Le modele de bruit peut peser des dizaines de kilo-octets : on ne compare
    # que son empreinte, jamais son texte.
    noise_a = sha256_of_text(man_a.noise_json) if man_a.noise_json else None
    noise_b = sha256_of_text(man_b.noise_json) if man_b.noise_json else None

    fields = {
        "semantic_hash": _field_diff(man_a.semantic_hash, man_b.semantic_hash),
        "circuit_hash": _field_diff(man_a.circuit_hash, man_b.circuit_hash),
        "mode": _field_diff(man_a.mode, man_b.mode),
        "backend_name": _field_diff(man_a.backend_name, man_b.backend_name),
        "backend_version": _field_diff(man_a.backend_version, man_b.backend_version),
        "seed": _field_diff(man_a.seed, man_b.seed),
        "repetitions": _field_diff(man_a.repetitions, man_b.repetitions),
        "noise_hash": _field_diff(noise_a, noise_b),
        "result_hash": _field_diff(record_a.result_hash, record_b.result_hash),
        "state_vector_hash": _field_diff(
            record_a.state_vector_hash, record_b.state_vector_hash
        ),
    }

    # Le niveau d'une option depend du mode : si les deux enregistrements n'ont
    # pas le meme mode, une meme option peut apparaitre dans deux niveaux
    # differents. C'est le modele central du harnais, pas un defaut du diff.
    options = {
        "semantic": _dict_drift(man_a.semantic_options, man_b.semantic_options),
        "numeric": _dict_drift(man_a.numeric_options, man_b.numeric_options),
        "performance": _dict_drift(
            man_a.performance_options, man_b.performance_options
        ),
    }
    kernel = _dict_drift(man_a.kernel, man_b.kernel)
    environment = _dict_drift(man_a.environment, man_b.environment)

    # « Identiques » = meme recette scellee ET meme resultat scelle. Les options
    # de niveau PERFORMANCE, l'environnement et la date peuvent differer sans
    # rompre cette identite : c'est exactement ce que la table affirme.
    identical = (
        fields["semantic_hash"]["equal"]
        and fields["result_hash"]["equal"]
        and fields["state_vector_hash"]["equal"]
    )
    code = EXIT_OK if identical else EXIT_MISMATCH

    payload = {
        "a": str(Path(args.directory_a)),
        "b": str(Path(args.directory_b)),
        "identical": identical,
        "fields": fields,
        "options": options,
        "kernel_drift": kernel,
        "environment_drift": environment,
        "executed_circuit": False,
        "exit_code": code,
    }

    text = [
        "Comparaison sans execution",
        _line("A", Path(args.directory_a)),
        _line("B", Path(args.directory_b)),
        _line("verdict", "semantiquement IDENTIQUES" if identical else "DIFFERENTS"),
        "  champs scelles :",
    ]
    for name in fields:
        entry = fields[name]
        if entry["equal"]:
            text.append(_line(f"    {name}", "identique"))
        else:
            text.append(_line(f"    {name}", "DIFFERENT"))
            text.append(_line("      A", _short(entry["a"])))
            text.append(_line("      B", _short(entry["b"])))
    text.append("  options par niveau :")
    for tier in ("semantic", "numeric", "performance"):
        drift = options[tier]
        if not drift:
            text.append(_line(f"    {tier}", "identiques"))
            continue
        text.append(_line(f"    {tier}", f"{len(drift)} ecart(s)"))
        for key in drift:
            text.append(
                _line(f"      {key}", f"{drift[key]['a']!r} -> {drift[key]['b']!r}")
            )
    text.append("  noyau SIMD :")
    if kernel:
        for key in kernel:
            text.append(
                _line(f"    {key}", f"{kernel[key]['a']!r} -> {kernel[key]['b']!r}")
            )
    else:
        text.append(_line("    (aucun ecart)", "meme noyau"))
    text.append("  environnement :")
    if environment:
        for key in environment:
            text.append(
                _line(
                    f"    {key}",
                    f"{environment[key]['a']!r} -> {environment[key]['b']!r}",
                )
            )
    else:
        text.append(_line("    (aucun ecart)", "meme environnement"))

    _emit(payload, args.json_output, text)
    return code


# --------------------------------------------------------------------------
# Analyseur d'arguments
# --------------------------------------------------------------------------

_EPILOG = """\
Codes de sortie
  0  succes : dossier intact, enregistrements identiques, rejeu conforme
  1  ecart de niveau avertissement (falsification, difference, chi2 compatible)
  2  reserve a `replay` : verdict DIVERGENT
  3  panne : dossier absent, JSON illisible, backend ou option inconnus

Les commandes `verify`, `info` et `diff` n'executent aucun circuit.
"""

_REPLAY_EPILOG = """\
Codes de sortie de `replay`
  0  BIT_EXACT                 octets identiques a l'archive
  0  NUMERICALLY_EQUIVALENT    infidelite <= 1e-4
  1  STATISTICALLY_COMPATIBLE  chi2 p >= 0.001 - avertissement, pas un echec :
                               c'est le plafond atteignable par du materiel reel
  2  DIVERGENT                 au-dela : la derive est a expliquer
  3  panne (dossier absent, backend inconnu, option non surchargeable)

`--option` ne peut porter que des options de niveau PERFORMANCE POUR LE MODE de
l'enregistrement. Une option SEMANTIC ou NUMERIC est refusee : la surcharger
changerait le resultat, donc invaliderait la comparaison.
"""

_DIFF_EPILOG = """\
`diff` ne lit que des octets : aucun circuit n'est execute.

"Semantiquement identiques" signifie meme `semantic_hash` ET memes hashes de
resultat. Deux enregistrements peuvent donc differer par leurs options de niveau
PERFORMANCE, leur environnement ou leur date, et rester identiques : c'est
precisement ce que la table des niveaux affirme.

  0  identiques
  1  differents
  3  panne (dossier absent, JSON illisible)
"""


def _build_parser() -> argparse.ArgumentParser:
    """Construit l'analyseur. Ne fait aucun import du domaine quantique."""
    parser = argparse.ArgumentParser(
        prog="qbridge",
        description=(
            "Harnais de capture/replay pour executions de circuits quantiques. "
            "On scelle la recette, jamais l'etat."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMANDE")
    subparsers.required = True

    # ---- capture ----
    capture_p = subparsers.add_parser(
        "capture",
        help="executer un circuit et sceller le resultat dans un dossier",
        description=(
            "Charge un circuit serialise par cirq.to_json, l'execute, et ecrit "
            "manifest.json + record.json + samples.npz dans le dossier de sortie."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    capture_p.add_argument("circuit", help="fichier JSON produit par cirq.to_json")
    capture_p.add_argument(
        "--seed",
        type=int,
        required=True,
        metavar="N",
        help="seed explicite, obligatoire : sans lui le manifeste serait mensonger",
    )
    capture_p.add_argument(
        "--backend",
        default="qsim",
        metavar="NOM",
        help="qsim (defaut) ou cirq-reference",
    )
    capture_p.add_argument(
        "--repetitions",
        type=int,
        default=None,
        metavar="N",
        help="nombre de tirages ; omis, on calcule le vecteur d'etat complet",
    )
    capture_p.add_argument(
        "--option",
        action="append",
        metavar="CLE=VALEUR",
        help="option d'execution, repetable (ex. --option cpu_threads=8)",
    )
    capture_p.add_argument(
        "--out", required=True, metavar="DOSSIER", help="dossier de destination"
    )
    capture_p.set_defaults(handler=_cmd_capture, json_output=False)

    # ---- verify ----
    verify_p = subparsers.add_parser(
        "verify",
        help="verifier l'integrite d'un dossier, sans aucune ressource quantique",
        description=(
            "Recalcule le hash semantique du manifeste et le hash des bitstrings "
            "archives, puis les compare aux hashes scelles. N'execute aucun "
            "circuit : c'est la garantie archivistique, elle ne peut pas echouer "
            "faute de materiel."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    verify_p.add_argument("directory", help="dossier d'un enregistrement")
    verify_p.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="sortie JSON canonique sur stdout",
    )
    verify_p.set_defaults(handler=_cmd_verify)

    # ---- replay ----
    replay_p = subparsers.add_parser(
        "replay",
        help="rejouer un enregistrement et comparer au resultat archive",
        description=(
            "Re-execute le circuit scelle et compare au resultat ARCHIVE, pas a "
            "une re-execution : la reference vient du disque."
        ),
        epilog=_REPLAY_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    replay_p.add_argument("directory", help="dossier d'un enregistrement")
    replay_p.add_argument(
        "--backend",
        default=None,
        metavar="NOM",
        help="backend de rejeu ; par defaut celui du manifeste",
    )
    replay_p.add_argument(
        "--option",
        action="append",
        metavar="CLE=VALEUR",
        help="surcharge d'option de niveau PERFORMANCE, repetable",
    )
    replay_p.set_defaults(handler=_cmd_replay, json_output=False)

    # ---- info ----
    info_p = subparsers.add_parser(
        "info",
        help="resumer un enregistrement, sans rien executer",
        description=(
            "Affiche mode, backend, seed, repetitions, les trois niveaux "
            "d'options, le noyau SIMD, l'environnement et le decompte des "
            "tirages. Aucun circuit n'est execute."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    info_p.add_argument("directory", help="dossier d'un enregistrement")
    info_p.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="sortie JSON canonique sur stdout",
    )
    info_p.set_defaults(handler=_cmd_info)

    # ---- diff ----
    diff_p = subparsers.add_parser(
        "diff",
        help="comparer deux enregistrements, sans rien executer",
        description=(
            "Compare hashes semantiques, hashes de resultat, derive "
            "d'environnement champ par champ et ecarts d'options par niveau."
        ),
        epilog=_DIFF_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    diff_p.add_argument("directory_a", metavar="DOSSIER_A")
    diff_p.add_argument("directory_b", metavar="DOSSIER_B")
    diff_p.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="sortie JSON canonique sur stdout",
    )
    diff_p.set_defaults(handler=_cmd_diff)

    return parser


# --------------------------------------------------------------------------
# Point d'entree
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Point d'entree. Renvoie un code de sortie POSIX, ne leve jamais."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse ecrit lui-meme son message (--help, argument manquant).
        # On convertit en code de retour pour que `main` reste appelable
        # directement, sans sous-processus.
        return int(exc.code or 0)

    as_json = bool(getattr(args, "json_output", False))
    try:
        return args.handler(args)
    except CliError as exc:
        return _fail(str(exc), as_json)
    except KeyboardInterrupt:
        return _fail("interrompu", as_json)
    except Exception as exc:  # dernier filet : aucun traceback ne doit sortir
        return _fail(f"erreur inattendue - {type(exc).__name__} : {exc}", as_json)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
