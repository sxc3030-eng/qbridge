"""Tests de l'interface en ligne de commande.

`main()` est appele DIRECTEMENT, jamais via un sous-processus. Le contrat de la
fonction est de renvoyer un code de sortie POSIX sans jamais lever : c'est ce
contrat qu'on teste. Passer par un sous-processus testerait le lanceur de
paquets, pas le code.
"""

from __future__ import annotations

import json

import cirq
import numpy as np
import pytest

from qbridge.capture import capture, hash_samples
from qbridge.cli import (
    EXIT_DIVERGENT,
    EXIT_ERROR,
    EXIT_MISMATCH,
    EXIT_OK,
    _parse_option,
    main,
)
from qbridge.record import RunRecord

# --------------------------------------------------------------------------
# Materiel de test
# --------------------------------------------------------------------------


def _bell_measured() -> cirq.Circuit:
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1), cirq.measure(q0, q1, key="m"))


def _bell() -> cirq.Circuit:
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1))


def _write_circuit(tmp_path, name: str = "bell.json", circuit=None) -> str:
    """Ecrit un circuit sur disque comme le ferait l'utilisateur."""
    path = tmp_path / name
    cirq.to_json(_bell_measured() if circuit is None else circuit, str(path))
    return str(path)


def _capture(
    tmp_path,
    out_name: str = "run",
    *extra: str,
    seed: int = 7,
    repetitions: int | None = 200,
    circuit_path: str | None = None,
) -> str:
    """Scelle un enregistrement via la CLI et renvoie son dossier."""
    circuit_path = circuit_path or _write_circuit(tmp_path)
    out = tmp_path / out_name
    argv = ["capture", circuit_path, "--seed", str(seed), "--out", str(out)]
    if repetitions is not None:
        argv += ["--repetitions", str(repetitions)]
    argv += list(extra)
    assert main(argv) == EXIT_OK
    return str(out)


def _break_the_backends(monkeypatch) -> None:
    """Rend les deux backends inutilisables.

    Meme idiome que `test_record.py::test_verify_archival_n_execute_aucun_circuit` :
    si une commande archivistique touchait un simulateur, le test exploserait.
    """
    import qbridge.backends.cirq_ref as cr
    import qbridge.backends.qsim as qs

    def forbidden(*a, **k):
        raise AssertionError("la commande a execute un circuit — elle ne doit pas")

    monkeypatch.setattr(qs.QsimBackend, "sample", forbidden)
    monkeypatch.setattr(qs.QsimBackend, "simulate", forbidden)
    monkeypatch.setattr(cr.CirqReferenceBackend, "sample", forbidden)
    monkeypatch.setattr(cr.CirqReferenceBackend, "simulate", forbidden)


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------


def test_capture_seals_a_directory(tmp_path, capsys):
    out = _capture(tmp_path)
    for name in ("manifest.json", "record.json", "samples.npz"):
        assert (tmp_path / "run" / name).is_file(), f"{name} absent"
    printed = capsys.readouterr().out
    assert "hash semantique" in printed
    assert "terminal_sampling" in printed
    assert "options par niveau" in printed
    assert RunRecord.load(out).samples["m"].shape == (200, 2)


def test_capture_without_repetitions_uses_state_vector_mode(tmp_path, capsys):
    # Circuit SANS mesure : c'est la seule facon d'obtenir le mode
    # state_vector. Un circuit qui mesure est classe midcircuit meme sans
    # repetitions — voir le test suivant.
    chemin = _write_circuit(tmp_path, "pur.json", circuit=_bell())
    out = _capture(tmp_path, "sv", repetitions=None, circuit_path=chemin)
    assert not (tmp_path / "sv" / "samples.npz").exists()
    record = RunRecord.load(out)
    assert record.samples is None
    assert record.state_vector_hash is not None
    assert "state_vector" in capsys.readouterr().out


def test_capture_sans_repetitions_mais_avec_mesures_reste_midcircuit(tmp_path):
    """Garde-fou de surete, pas un detail.

    `simulate()` sur un circuit qui mesure ECHANTILLONNE ces mesures. Le classer
    state_vector rendrait `cpu_threads` librement surchargeable sur un resultat
    qui depend de tirages — or c'est justement le cas ou le nombre de threads
    change les bitstrings.
    """
    out = _capture(tmp_path, "mesure", repetitions=None)  # _bell_measured par defaut
    assert RunRecord.load(out).manifest.mode == "midcircuit_sampling"


def test_capture_types_an_integer_option(tmp_path):
    """`--option cpu_threads=8` doit produire un entier, pas la chaine "8"."""
    out = _capture(tmp_path, "run", "--option", "cpu_threads=8")
    manifest = RunRecord.load(out).manifest
    # En echantillonnage terminal, cpu_threads est de niveau PERFORMANCE.
    assert manifest.performance_options == {"cpu_threads": 8}
    assert isinstance(manifest.performance_options["cpu_threads"], int)


def test_capture_types_a_boolean_option(tmp_path):
    out = _capture(tmp_path, "run", "--option", "denormals_are_zeros=true")
    manifest = RunRecord.load(out).manifest
    assert manifest.numeric_options["denormals_are_zeros"] is True


def test_capture_accepts_several_options(tmp_path):
    out = _capture(
        tmp_path,
        "run",
        "--option",
        "cpu_threads=4",
        "--option",
        "max_fused_gate_size=2",
    )
    manifest = RunRecord.load(out).manifest
    assert manifest.performance_options == {"cpu_threads": 4}
    assert manifest.numeric_options == {"max_fused_gate_size": 2}


def test_capture_places_the_option_in_the_tier_of_the_mode(tmp_path):
    """Le niveau depend du mode : c'est le modele central du harnais.

    Le meme `cpu_threads` est PERFORMANCE en echantillonnage terminal et
    SEMANTIC en midcircuit. `repetitions=1` bascule qsim sur la boucle par
    repetition, donc sur le mode midcircuit.
    """
    terminal = RunRecord.load(
        _capture(tmp_path, "term", "--option", "cpu_threads=2", repetitions=200)
    ).manifest
    midcircuit = RunRecord.load(
        _capture(tmp_path, "mid", "--option", "cpu_threads=2", repetitions=1)
    ).manifest
    assert terminal.performance_options == {"cpu_threads": 2}
    assert midcircuit.semantic_options == {"cpu_threads": 2}


def test_capture_rejects_an_unknown_option(tmp_path, capsys):
    path = _write_circuit(tmp_path)
    code = main(
        [
            "capture",
            path,
            "--seed",
            "7",
            "--repetitions",
            "10",
            "--option",
            "pas_une_option=3",
            "--out",
            str(tmp_path / "run"),
        ]
    )
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert "pas_une_option" in err and "qbridge :" in err


def test_capture_rejects_a_malformed_option(tmp_path, capsys):
    path = _write_circuit(tmp_path)
    code = main(
        [
            "capture",
            path,
            "--seed",
            "7",
            "--option",
            "cpu_threads",
            "--out",
            str(tmp_path / "run"),
        ]
    )
    assert code == EXIT_ERROR
    assert "cle=valeur" in capsys.readouterr().err


def test_capture_rejects_an_unknown_backend(tmp_path, capsys):
    path = _write_circuit(tmp_path)
    code = main(
        [
            "capture",
            path,
            "--seed",
            "7",
            "--backend",
            "ordinateur-quantique-du-futur",
            "--out",
            str(tmp_path / "run"),
        ]
    )
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert "Backend inconnu" in err and "qsim" in err


def test_capture_rejects_a_missing_circuit_file(tmp_path, capsys):
    code = main(
        [
            "capture",
            str(tmp_path / "absent.json"),
            "--seed",
            "7",
            "--out",
            str(tmp_path / "run"),
        ]
    )
    assert code == EXIT_ERROR
    assert "introuvable" in capsys.readouterr().err


def test_capture_rejects_malformed_json(tmp_path, capsys):
    path = tmp_path / "casse.json"
    path.write_text("{ceci n'est pas du json", encoding="utf-8")
    code = main(
        ["capture", str(path), "--seed", "7", "--out", str(tmp_path / "run")]
    )
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert "illisible" in err
    assert "Traceback" not in err


def test_capture_rejects_json_that_is_not_a_circuit(tmp_path, capsys):
    path = _write_circuit(tmp_path, "qubit.json", circuit=cirq.LineQubit(0))
    code = main(
        ["capture", path, "--seed", "7", "--out", str(tmp_path / "run")]
    )
    assert code == EXIT_ERROR
    assert "pas un cirq.Circuit" in capsys.readouterr().err


def test_capture_rejects_zero_repetitions(tmp_path, capsys):
    path = _write_circuit(tmp_path)
    code = main(
        [
            "capture",
            path,
            "--seed",
            "7",
            "--repetitions",
            "0",
            "--out",
            str(tmp_path / "run"),
        ]
    )
    assert code == EXIT_ERROR
    assert "repetitions" in capsys.readouterr().err


def test_capture_rejects_options_on_the_reference_backend(tmp_path, capsys):
    """L'oracle Cirq n'accepte aucune option d'execution."""
    path = _write_circuit(tmp_path)
    code = main(
        [
            "capture",
            path,
            "--seed",
            "7",
            "--repetitions",
            "20",
            "--backend",
            "cirq-reference",
            "--option",
            "cpu_threads=2",
            "--out",
            str(tmp_path / "run"),
        ]
    )
    assert code == EXIT_ERROR
    assert "cirq-reference" in capsys.readouterr().err


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------


def test_verify_accepts_a_healthy_directory(tmp_path, capsys):
    out = _capture(tmp_path)
    capsys.readouterr()
    assert main(["verify", out]) == EXIT_OK
    printed = capsys.readouterr().out
    assert "intact" in printed
    assert "Aucun circuit n'a ete execute" in printed
    assert "200" in printed


def test_verify_json_output(tmp_path, capsys):
    out = _capture(tmp_path)
    capsys.readouterr()
    assert main(["verify", out, "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest_intact"] is True
    assert payload["results_intact"] is True
    assert payload["measurement_keys"] == ["m"]
    assert payload["total_shots"] == 200
    assert payload["executed_circuit"] is False
    assert payload["exit_code"] == EXIT_OK


def test_verify_detects_a_tampered_manifest(tmp_path, capsys):
    out = _capture(tmp_path)
    capsys.readouterr()
    path = tmp_path / "run" / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["seed"] = 999
    path.write_text(json.dumps(data), encoding="utf-8")

    assert main(["verify", out]) == EXIT_MISMATCH
    assert "FALSIFIE" in capsys.readouterr().out


def test_verify_detects_tampered_bitstrings(tmp_path, capsys):
    out = _capture(tmp_path)
    capsys.readouterr()
    path = tmp_path / "run" / "record.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["result_hash"] = "0" * 64
    path.write_text(json.dumps(data), encoding="utf-8")

    assert main(["verify", out]) == EXIT_MISMATCH
    assert "FALSIFIE" in capsys.readouterr().out


def test_verify_reports_a_missing_directory(tmp_path, capsys):
    assert main(["verify", str(tmp_path / "nulle-part")]) == EXIT_ERROR
    assert "introuvable" in capsys.readouterr().err


def test_verify_reports_a_directory_without_a_record(tmp_path, capsys):
    (tmp_path / "vide").mkdir()
    assert main(["verify", str(tmp_path / "vide")]) == EXIT_ERROR
    assert "record.json" in capsys.readouterr().err


def test_verify_reports_malformed_json(tmp_path, capsys):
    out = _capture(tmp_path)
    capsys.readouterr()
    (tmp_path / "run" / "record.json").write_text("{pas du json", encoding="utf-8")
    assert main(["verify", out]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert "illisible" in err
    assert "Traceback" not in err


def test_verify_json_error_stays_machine_readable(tmp_path, capsys):
    """En mode --json, meme l'echec doit sortir en JSON sur stdout."""
    code = main(["verify", str(tmp_path / "nulle-part"), "--json"])
    assert code == EXIT_ERROR
    payload = json.loads(capsys.readouterr().out)
    assert payload["exit_code"] == EXIT_ERROR
    assert "introuvable" in payload["error"]


def test_verify_executes_no_circuit(tmp_path, monkeypatch, capsys):
    """La garantie archivistique doit tenir sans le moindre simulateur."""
    out = _capture(tmp_path)  # capture AVANT de casser les backends
    capsys.readouterr()
    _break_the_backends(monkeypatch)
    assert main(["verify", out]) == EXIT_OK
    assert "Aucun circuit n'a ete execute" in capsys.readouterr().out


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------


def test_replay_bit_exact_exits_zero(tmp_path, capsys):
    out = _capture(tmp_path)
    capsys.readouterr()
    assert main(["replay", out]) == EXIT_OK
    assert "BIT_EXACT" in capsys.readouterr().out


def test_replay_statistically_compatible_exits_one(tmp_path, capsys):
    """Verdict d'avertissement : le plafond atteignable par du materiel reel.

    On archive des tirages issus d'un autre seed : meme distribution, octets
    differents. Le rejeu (seed du manifeste) ne peut donc pas etre bit-exact,
    mais le chi2 doit rester compatible.
    """
    circuit = _bell_measured()
    original = RunRecord.from_capture(capture(circuit, seed=7, repetitions=400))
    other = capture(circuit, seed=99, repetitions=400)
    assert original.samples["m"].tobytes() != other.samples["m"].tobytes()

    mixed = RunRecord(
        schema_version=original.schema_version,
        manifest=original.manifest,
        result_hash=hash_samples(other.samples),  # coherent avec les tirages
        samples=other.samples,
        state_vector_hash=original.state_vector_hash,
    )
    mixed.save(tmp_path / "melange")
    capsys.readouterr()

    assert main(["replay", str(tmp_path / "melange")]) == EXIT_MISMATCH
    printed = capsys.readouterr().out
    assert "STATISTICALLY_COMPATIBLE" in printed
    assert "chi2 p" in printed


def test_replay_divergent_exits_two(tmp_path, capsys):
    circuit = _bell_measured()
    original = RunRecord.from_capture(capture(circuit, seed=7, repetitions=200))
    rng = np.random.default_rng(0)
    fake = {
        "m": rng.integers(
            0, 2, size=original.samples["m"].shape, dtype=original.samples["m"].dtype
        )
    }
    divergent = RunRecord(
        schema_version=original.schema_version,
        manifest=original.manifest,
        result_hash=hash_samples(fake),
        samples=fake,
        state_vector_hash=original.state_vector_hash,
    )
    divergent.save(tmp_path / "divergent")
    capsys.readouterr()

    assert main(["replay", str(tmp_path / "divergent")]) == EXIT_DIVERGENT
    assert "DIVERGENT" in capsys.readouterr().out


def test_replay_accepts_a_performance_override(tmp_path, capsys):
    """En echantillonnage terminal, cpu_threads est mesure neutre."""
    out = _capture(tmp_path, "run", "--option", "cpu_threads=1")
    capsys.readouterr()
    assert main(["replay", out, "--option", "cpu_threads=2"]) == EXIT_OK
    assert "BIT_EXACT" in capsys.readouterr().out


def test_replay_refuses_a_numeric_override(tmp_path, capsys):
    """Surcharger une option NUMERIC invaliderait la comparaison."""
    out = _capture(tmp_path)
    capsys.readouterr()
    code = main(["replay", out, "--option", "max_fused_gate_size=4"])
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert "PERFORMANCE" in err and "max_fused_gate_size" in err


def test_replay_rejects_an_unknown_backend(tmp_path, capsys):
    out = _capture(tmp_path)
    capsys.readouterr()
    assert main(["replay", out, "--backend", "inexistant"]) == EXIT_ERROR
    assert "Backend inconnu" in capsys.readouterr().err


def test_replay_rejects_an_unknown_option(tmp_path, capsys):
    out = _capture(tmp_path)
    capsys.readouterr()
    assert main(["replay", out, "--option", "pas_une_option=1"]) == EXIT_ERROR
    assert "pas_une_option" in capsys.readouterr().err


def test_replay_reports_a_missing_directory(tmp_path, capsys):
    assert main(["replay", str(tmp_path / "nulle-part")]) == EXIT_ERROR
    assert "introuvable" in capsys.readouterr().err


def test_replay_refuses_a_tampered_record(tmp_path, capsys):
    """Une archive falsifiee est une panne, pas un verdict."""
    out = _capture(tmp_path)
    capsys.readouterr()
    path = tmp_path / "run" / "record.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["result_hash"] = "0" * 64
    path.write_text(json.dumps(data), encoding="utf-8")

    assert main(["replay", out]) == EXIT_ERROR
    assert "bitstrings" in capsys.readouterr().err


def test_replay_help_documents_the_exit_codes(capsys):
    assert main(["replay", "--help"]) == EXIT_OK
    printed = capsys.readouterr().out
    assert "0  BIT_EXACT" in printed
    assert "0  NUMERICALLY_EQUIVALENT" in printed
    assert "1  STATISTICALLY_COMPATIBLE" in printed
    assert "2  DIVERGENT" in printed


# --------------------------------------------------------------------------
# info
# --------------------------------------------------------------------------


def test_info_summarizes_a_record(tmp_path, capsys):
    out = _capture(tmp_path, "run", "--option", "cpu_threads=8")
    capsys.readouterr()
    assert main(["info", out]) == EXIT_OK
    printed = capsys.readouterr().out
    assert "terminal_sampling" in printed
    assert "qsim" in printed
    assert "seed" in printed
    assert "cpu_threads=8" in printed
    assert "qsim_kernel_module" in printed
    assert "cirq_version" in printed
    assert "200 tirages" in printed


def test_info_json_output(tmp_path, capsys):
    out = _capture(tmp_path, "run", "--option", "cpu_threads=8")
    capsys.readouterr()
    assert main(["info", out, "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "terminal_sampling"
    assert payload["backend"]["name"] == "qsim"
    assert payload["backend"]["version"]
    assert payload["seed"] == 7
    assert payload["repetitions"] == 200
    assert payload["options"]["performance"] == {"cpu_threads": 8}
    assert payload["options"]["semantic"] == {}
    assert payload["options"]["numeric"] == {}
    assert "qsim_kernel_module" in payload["kernel"]
    assert "cirq_version" in payload["environment"]
    assert payload["measurement_keys"] == ["m"]
    assert payload["shots_per_key"] == {"m": 200}
    assert payload["total_shots"] == 200
    assert len(payload["semantic_hash"]) == 64


def test_info_on_a_state_vector_record(tmp_path, capsys):
    chemin = _write_circuit(tmp_path, "pur.json", circuit=_bell())
    out = _capture(tmp_path, "sv", repetitions=None, circuit_path=chemin)
    capsys.readouterr()
    assert main(["info", out, "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "state_vector"
    assert payload["repetitions"] is None
    assert payload["measurement_keys"] == []
    assert payload["total_shots"] == 0
    assert len(payload["state_vector_hash"]) == 64


def test_info_reports_a_missing_directory(tmp_path, capsys):
    assert main(["info", str(tmp_path / "nulle-part")]) == EXIT_ERROR
    assert "introuvable" in capsys.readouterr().err


def test_info_executes_no_circuit(tmp_path, monkeypatch, capsys):
    out = _capture(tmp_path)
    capsys.readouterr()
    _break_the_backends(monkeypatch)
    assert main(["info", out]) == EXIT_OK


# --------------------------------------------------------------------------
# diff
# --------------------------------------------------------------------------


def test_diff_identical_records_exit_zero(tmp_path, capsys):
    path = _write_circuit(tmp_path)
    a = _capture(tmp_path, "a", circuit_path=path)
    b = _capture(tmp_path, "b", circuit_path=path)
    capsys.readouterr()
    assert main(["diff", a, b]) == EXIT_OK
    assert "IDENTIQUES" in capsys.readouterr().out


def test_diff_different_seeds_exit_one(tmp_path, capsys):
    path = _write_circuit(tmp_path)
    a = _capture(tmp_path, "a", circuit_path=path, seed=7)
    b = _capture(tmp_path, "b", circuit_path=path, seed=11)
    capsys.readouterr()
    assert main(["diff", a, b]) == EXIT_MISMATCH
    printed = capsys.readouterr().out
    assert "DIFFERENTS" in printed
    assert "seed" in printed


def test_diff_json_output(tmp_path, capsys):
    path = _write_circuit(tmp_path)
    a = _capture(tmp_path, "a", circuit_path=path, seed=7)
    b = _capture(tmp_path, "b", circuit_path=path, seed=11)
    capsys.readouterr()
    assert main(["diff", a, b, "--json"]) == EXIT_MISMATCH
    payload = json.loads(capsys.readouterr().out)

    assert payload["identical"] is False
    assert payload["executed_circuit"] is False
    assert payload["fields"]["seed"]["equal"] is False
    assert payload["fields"]["seed"]["a"] == 7
    assert payload["fields"]["seed"]["b"] == 11
    assert payload["fields"]["circuit_hash"]["equal"] is True
    assert payload["fields"]["semantic_hash"]["equal"] is False
    assert payload["kernel_drift"] == {}


def test_diff_performance_options_do_not_break_identity(tmp_path, capsys):
    """Le coeur du modele : PERFORMANCE ne change pas le hash semantique.

    Deux captures qui ne different que par `cpu_threads` en echantillonnage
    terminal restent semantiquement identiques — c'est ce que la table
    OPTION_TIERS affirme, et le diff doit le refleter.
    """
    path = _write_circuit(tmp_path)
    a = _capture(tmp_path, "a", "--option", "cpu_threads=1", circuit_path=path)
    b = _capture(tmp_path, "b", "--option", "cpu_threads=4", circuit_path=path)
    capsys.readouterr()

    assert main(["diff", a, b, "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["identical"] is True
    assert payload["fields"]["semantic_hash"]["equal"] is True
    assert payload["options"]["semantic"] == {}
    assert payload["options"]["numeric"] == {}
    assert payload["options"]["performance"] == {"cpu_threads": {"a": 1, "b": 4}}


def test_diff_different_options_by_tier(tmp_path, capsys):
    path = _write_circuit(tmp_path)
    a = _capture(
        tmp_path, "a", "--option", "max_fused_gate_size=2", circuit_path=path
    )
    b = _capture(
        tmp_path, "b", "--option", "max_fused_gate_size=4", circuit_path=path
    )
    capsys.readouterr()

    assert main(["diff", a, b, "--json"]) == EXIT_MISMATCH
    payload = json.loads(capsys.readouterr().out)
    # NUMERIC entre dans le hash semantique : les deux doivent differer.
    assert payload["fields"]["semantic_hash"]["equal"] is False
    assert payload["options"]["numeric"] == {
        "max_fused_gate_size": {"a": 2, "b": 4}
    }


def test_diff_reports_a_missing_directory(tmp_path, capsys):
    a = _capture(tmp_path, "a")
    capsys.readouterr()
    assert main(["diff", a, str(tmp_path / "nulle-part")]) == EXIT_ERROR
    assert "introuvable" in capsys.readouterr().err


def test_diff_reports_malformed_json(tmp_path, capsys):
    path = _write_circuit(tmp_path)
    a = _capture(tmp_path, "a", circuit_path=path)
    b = _capture(tmp_path, "b", circuit_path=path)
    (tmp_path / "b" / "manifest.json").write_text("{casse", encoding="utf-8")
    capsys.readouterr()
    assert main(["diff", a, b]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert "illisible" in err
    assert "Traceback" not in err


def test_diff_executes_no_circuit(tmp_path, monkeypatch, capsys):
    """La comparaison est purement classique."""
    path = _write_circuit(tmp_path)
    a = _capture(tmp_path, "a", circuit_path=path)
    b = _capture(tmp_path, "b", circuit_path=path, seed=11)
    capsys.readouterr()
    _break_the_backends(monkeypatch)

    assert main(["diff", a, b]) == EXIT_MISMATCH
    assert main(["diff", a, a]) == EXIT_OK


# --------------------------------------------------------------------------
# Analyse des arguments et garde-fous generaux
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("cpu_threads=8", ("cpu_threads", 8)),
        ("verbosity=0", ("verbosity", 0)),
        ("use_gpu=true", ("use_gpu", True)),
        ("denormals_are_zeros=False", ("denormals_are_zeros", False)),
        ("gpu_mode=1.5", ("gpu_mode", 1.5)),
        ("gpu_mode=texte", ("gpu_mode", "texte")),
    ],
)
def test_option_values_are_typed(spec, expected):
    assert _parse_option(spec) == expected


def test_option_value_that_json_cannot_represent_is_refused():
    """NaN casserait `canonical_json`, donc le hash du manifeste."""
    from qbridge.cli import CliError

    with pytest.raises(CliError, match="NaN"):
        _parse_option("gpu_mode=nan")


def test_main_returns_a_code_without_a_subcommand():
    """argparse sort en SystemExit ; `main` doit renvoyer un code, pas lever."""
    assert main([]) == EXIT_ERROR  # et surtout pas 2, reserve a DIVERGENT


def test_main_returns_a_code_on_an_unknown_subcommand():
    assert main(["pas-une-commande"]) != EXIT_OK


def test_root_help_documents_the_exit_codes(capsys):
    assert main(["--help"]) == EXIT_OK
    printed = capsys.readouterr().out
    assert "Codes de sortie" in printed
    assert "n'executent aucun circuit" in printed


# --------------------------------------------------------------------------
# Codes de sortie : couverture totale des verdicts
# --------------------------------------------------------------------------


def test_chaque_verdict_a_un_code_de_sortie():
    """Non-regression : EXIT_INDETERMINATE etait defini mais branche nulle part,
    et `_exit_code_for` faisait un acces direct au dictionnaire — un verdict
    INDETERMINATE levait un KeyError brut jusque dans le terminal."""
    from qbridge.cli import _exit_code_for
    from qbridge.verdict import Verdict

    # `isinstance(code, int)` seul passait meme avec une table VIDE, puisque
    # `.get` rend EXIT_ERROR par defaut : BIT_EXACT aurait rendu 3 et le test
    # serait reste vert. On verifie donc la correspondance REELLE.
    from qbridge.cli import EXIT_DIVERGENT, EXIT_INDETERMINATE, EXIT_MISMATCH

    attendu = {
        Verdict.BIT_EXACT: EXIT_OK,
        Verdict.NUMERICALLY_EQUIVALENT: EXIT_OK,
        Verdict.STATISTICALLY_COMPATIBLE: EXIT_MISMATCH,
        Verdict.DIVERGENT: EXIT_DIVERGENT,
        Verdict.INDETERMINATE: EXIT_INDETERMINATE,
    }
    for verdict in Verdict:
        assert verdict in attendu, f"{verdict.name} n'a pas de code attendu"
        assert _exit_code_for(verdict) == attendu[verdict]


def test_indetermine_ne_vaut_pas_succes():
    from qbridge.cli import EXIT_INDETERMINATE, _exit_code_for
    from qbridge.verdict import Verdict

    assert _exit_code_for(Verdict.INDETERMINATE) == EXIT_INDETERMINATE
    assert EXIT_INDETERMINATE != EXIT_OK


def test_un_verdict_inconnu_ne_tombe_jamais_sur_succes():
    """Si un verdict est ajoute plus tard sans mettre la table a jour, le repli
    doit signaler une erreur, jamais une reussite : une CI le croirait."""
    from qbridge.cli import _exit_code_for

    assert _exit_code_for("verdict_qui_n_existe_pas") != EXIT_OK
