"""Non-regression sur les defauts de robustesse de la CLI.

Le contrat de `main()` : rendre un code POSIX, ne JAMAIS lever, et ne jamais
confondre « le harnais n'a pas pu travailler » avec « le harnais a travaille et
le resultat est mauvais ».
"""

from __future__ import annotations

import io
import json
import sys

import cirq
import numpy as np
import pytest

from qbridge.cli import EXIT_ERROR, EXIT_MISMATCH, EXIT_OK, main


def _circuit(tmp_path):
    q0, q1 = cirq.LineQubit.range(2)
    chemin = tmp_path / "bell.json"
    cirq.to_json(
        cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1), cirq.measure(q0, q1, key="m")),
        str(chemin),
    )
    return str(chemin)


@pytest.fixture
def dossier(tmp_path):
    out = tmp_path / "run"
    assert (
        main(
            [
                "capture",
                _circuit(tmp_path),
                "--seed",
                "7",
                "--repetitions",
                "50",
                "--out",
                str(out),
            ]
        )
        == EXIT_OK
    )
    return str(out)


class _FluxCasse(io.TextIOBase):
    """Un flux dont toute ecriture echoue, comme un tube ferme."""

    encoding = "utf-8"

    def write(self, s):  # noqa: D102
        raise BrokenPipeError(32, "Broken pipe")

    def flush(self):  # noqa: D102
        raise BrokenPipeError(32, "Broken pipe")

    def close(self):  # noqa: D102
        # Ne pas relever au ramasse-miettes : le test porte sur `main`, pas sur
        # le bruit de fermeture du faux flux.
        pass


# ---------- F1 : le point d'entree doit exister ----------


def test_le_point_d_entree_declare_existe_vraiment():
    """AVANT correction : pyproject declarait `qbridge.cli:cli_entry_point`, un
    nom qui n'existait nulle part. La commande installee ne pouvait pas
    demarrer — `qbridge --help` echouait avant meme d'atteindre argparse.
    Personne ne l'avait vu parce que tout le monde utilisait `python -m`.
    """
    import importlib
    import pathlib
    import tomllib

    racine = pathlib.Path(__file__).resolve().parent.parent
    config = tomllib.loads((racine / "pyproject.toml").read_text(encoding="utf-8"))
    spec = config["project"]["scripts"]["qbridge"]
    module, _, attribut = spec.partition(":")

    cible = importlib.import_module(module)
    assert hasattr(cible, attribut), (
        f"pyproject declare {spec!r} mais {attribut!r} n'existe pas dans {module}"
    )
    assert callable(getattr(cible, attribut))


# ---------- F2 : --help ne doit dependre ni de cirq ni de qsimcirq ----------


def test_help_fonctionne_sans_cirq_ni_qsimcirq():
    """AVANT correction : `cli.py` evitait soigneusement les imports du domaine
    au niveau du module, mais `qbridge/__init__.py` etait eager et les tirait
    quand meme. La precaution etait annulee par le paquet, et le docstring de
    `cli.py` affirmait le contraire de ce qui se passait.
    """
    import importlib.abc
    import subprocess

    programme = (
        "import sys, importlib.abc\n"
        "BLOQUES = {'cirq', 'qsimcirq'}\n"
        "class B(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, nom, chemin=None, cible=None):\n"
        "        if nom.split('.')[0] in BLOQUES:\n"
        "            raise ImportError('absent (simule)')\n"
        "        return None\n"
        "sys.meta_path.insert(0, B())\n"
        "from qbridge.cli import main\n"
        "raise SystemExit(main(['--help']))\n"
    )
    resultat = subprocess.run(
        [sys.executable, "-c", programme],
        capture_output=True,
        text=True,
    )
    assert resultat.returncode == 0, (
        f"--help a echoue sans cirq : {resultat.stderr[-400:]}"
    )


# ---------- F3, F4 : main() ne doit jamais lever, ni degrader un succes ----


def test_un_tube_casse_ne_fait_pas_lever_main(dossier, monkeypatch):
    """AVANT correction : `_fail` et `_emit` faisaient `print` DEPUIS les blocs
    `except` de `main`. Si l'ecriture echouait, la nouvelle exception sortait de
    `main`, ou plus aucun handler ne l'attendait. `qbridge info DIR --json |
    head -1` suffisait."""
    monkeypatch.setattr(sys, "stdout", _FluxCasse())
    monkeypatch.setattr(sys, "stderr", _FluxCasse())
    for argv in (
        ["verify", dossier],
        ["verify", dossier, "--json"],
        ["info", dossier],
        ["diff", dossier, dossier],
    ):
        code = main(argv)  # ne doit pas lever
        assert isinstance(code, int)


def test_un_tube_casse_ne_transforme_PAS_un_succes_en_panne(dossier, monkeypatch):
    """Le pire du meme mecanisme : la verification d'une archive PARFAITEMENT
    INTACTE renvoyait 3. Une chaine d'integration lisait « le harnais n'a pas
    pu travailler » pour une archive qui venait de se verifier."""
    assert main(["verify", dossier]) == EXIT_OK

    monkeypatch.setattr(sys, "stdout", _FluxCasse())
    monkeypatch.setattr(sys, "stderr", _FluxCasse())
    assert main(["verify", dossier]) == EXIT_OK, (
        "l'echec d'affichage a ete confondu avec un echec de verification"
    )
    assert main(["info", dossier]) == EXIT_OK


def test_une_vraie_panne_reste_une_panne_meme_sur_tube_casse(tmp_path, monkeypatch):
    """Controle : sans lui, le test precedent serait satisfait par « rendre
    toujours 0 »."""
    monkeypatch.setattr(sys, "stdout", _FluxCasse())
    monkeypatch.setattr(sys, "stderr", _FluxCasse())
    assert main(["verify", str(tmp_path / "nulle-part")]) == EXIT_ERROR


# ---------- F5 : le filtre d'exceptions doit tenir sa promesse ----------


@pytest.mark.parametrize("corps", ["null", "[]", '"coucou"', "123"])
def test_un_record_json_non_dictionnaire_donne_un_message_propre(
    dossier, corps, capsys
):
    """AVANT correction : `record.json` valant `null` produisait
    `erreur inattendue - AttributeError`, un message interne Python la ou le
    docstring promettait de traduire TOUTE panne."""
    from pathlib import Path

    (Path(dossier) / "record.json").write_text(corps, encoding="utf-8")
    code = main(["verify", dossier])
    assert code == EXIT_ERROR
    assert "erreur inattendue" not in capsys.readouterr().err


def test_un_samples_npz_tronque_donne_un_message_propre(dossier, capsys):
    from pathlib import Path

    chemin = Path(dossier) / "samples.npz"
    octets = chemin.read_bytes()
    chemin.write_bytes(octets[: len(octets) // 2])
    assert main(["verify", dossier]) == EXIT_ERROR
    assert "erreur inattendue" not in capsys.readouterr().err


# ---------- F6 : une falsification ne doit pas etre retrogradee ----------


def test_une_falsification_reste_signalee_malgre_une_signature_illisible(dossier):
    """AVANT correction : un attaquant capable de reecrire samples.npz pouvait
    aussi ecrire `null` dans signature.json. `verify` levait alors sur la
    signature AVANT de rendre le verdict d'integrite deja calcule, retrogradant
    une falsification DETECTEE de 1 vers 3.

    Une chaine qui retente sur 3 (« probleme d'infra ») et alerte sur 1
    (« archive falsifiee ») lisait le mauvais signal, et l'attaquant choisissait
    lequel.
    """
    from pathlib import Path

    from qbridge.record import RunRecord

    record = RunRecord.load(dossier)
    faux = {"m": np.zeros(record.samples["m"].shape, dtype=record.samples["m"].dtype)}
    np.savez_compressed(Path(dossier) / "samples.npz", **faux)

    seule = main(["verify", dossier])
    assert seule == EXIT_MISMATCH

    (Path(dossier) / "signature.json").write_text("null", encoding="utf-8")
    avec_signature_pourrie = main(["verify", dossier])
    assert avec_signature_pourrie == EXIT_MISMATCH, (
        "la falsification a ete retrogradee en panne d'infrastructure"
    )


# ---------- F7 : les VALEURS d'options doivent etre typees ----------


@pytest.mark.parametrize("valeur", ["off", "on", "no", "true", "1.5", "-1"])
def test_une_valeur_d_option_absurde_est_refusee(tmp_path, valeur, capsys):
    """AVANT correction : `--option cpu_threads=off` produisait `False`, scelle
    tel quel dans le manifeste et passe a qsim comme zero thread. Le nom etait
    valide, la valeur ne l'etait pas, et rien ne la regardait."""
    code = main(
        [
            "capture",
            _circuit(tmp_path),
            "--seed",
            "7",
            "--repetitions",
            "20",
            "--option",
            f"cpu_threads={valeur}",
            "--out",
            str(tmp_path / f"r-{valeur}"),
        ]
    )
    assert code == EXIT_ERROR
    assert "erreur inattendue" not in capsys.readouterr().err


def test_une_valeur_d_option_correcte_passe_toujours(tmp_path):
    """Controle : la validation ne doit pas tout refuser."""
    from qbridge.record import RunRecord

    out = tmp_path / "bon"
    assert (
        main(
            [
                "capture",
                _circuit(tmp_path),
                "--seed",
                "7",
                "--repetitions",
                "20",
                "--option",
                "cpu_threads=4",
                "--out",
                str(out),
            ]
        )
        == EXIT_OK
    )
    assert RunRecord.load(str(out)).manifest.performance_options == {"cpu_threads": 4}


def test_une_option_booleenne_accepte_bien_un_booleen(tmp_path):
    from qbridge.record import RunRecord

    out = tmp_path / "bool"
    assert (
        main(
            [
                "capture",
                _circuit(tmp_path),
                "--seed",
                "7",
                "--repetitions",
                "20",
                "--option",
                "denormals_are_zeros=true",
                "--out",
                str(out),
            ]
        )
        == EXIT_OK
    )
    assert RunRecord.load(str(out)).manifest.numeric_options == {
        "denormals_are_zeros": True
    }


def test_le_type_attendu_vient_de_qsim_et_non_d_une_liste_locale():
    """Une table ecrite a la main se desynchroniserait de qsim en silence."""
    import dataclasses

    import qsimcirq

    from qbridge.tiers import option_types

    types = option_types()
    for champ in dataclasses.fields(qsimcirq.QSimOptions):
        assert champ.name in types, f"{champ.name} n'a pas de type derive"
