"""Tests CLI des commandes `keygen` et `sign`, et de `verify` signe.

`main()` est appele directement : le contrat est de rendre un code POSIX sans
jamais lever, y compris sur des cles absentes ou mal formees.
"""

from __future__ import annotations

import json
import os

import cirq
import pytest

from qbridge.cli import EXIT_ERROR, EXIT_MISMATCH, EXIT_OK, main


def _write_circuit(tmp_path):
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1), cirq.measure(q0, q1, key="m"))
    chemin = tmp_path / "bell.json"
    cirq.to_json(c, str(chemin))
    return str(chemin)


@pytest.fixture
def dossier(tmp_path):
    """Un enregistrement scelle, pret a signer."""
    out = tmp_path / "run"
    assert (
        main(
            [
                "capture",
                _write_circuit(tmp_path),
                "--seed",
                "7",
                "--repetitions",
                "100",
                "--out",
                str(out),
            ]
        )
        == EXIT_OK
    )
    return str(out)


@pytest.fixture
def paire(tmp_path):
    """Une paire ed25519 sur disque."""
    priv = tmp_path / "priv.key"
    pub = tmp_path / "pub.key"
    assert (
        main(
            [
                "keygen",
                "--key-id",
                "simon",
                "--private-out",
                str(priv),
                "--public-out",
                str(pub),
            ]
        )
        == EXIT_OK
    )
    return str(priv), str(pub)


@pytest.fixture
def cle_hmac(tmp_path):
    chemin = tmp_path / "hmac.key"
    chemin.write_text(os.urandom(32).hex(), encoding="utf-8")
    return str(chemin)


# ---------- keygen ----------


def test_keygen_ecrit_les_deux_fichiers(tmp_path, paire):
    priv, pub = paire
    for chemin in (priv, pub):
        contenu = open(chemin, encoding="utf-8").read().strip()
        assert bytes.fromhex(contenu)  # hexadecimal valide


def test_les_deux_cles_different(paire):
    priv, pub = paire
    assert open(priv, encoding="utf-8").read() != open(pub, encoding="utf-8").read()


def test_keygen_refuse_d_ecraser_une_cle(tmp_path, paire, capsys):
    """Ecraser une cle privee est irreversible : on refuse, on n'ecrase pas."""
    priv, _pub = paire
    code = main(
        [
            "keygen",
            "--key-id",
            "simon",
            "--private-out",
            priv,
            "--public-out",
            str(tmp_path / "autre.key"),
        ]
    )
    assert code == EXIT_ERROR
    assert "existe deja" in capsys.readouterr().err


def test_keygen_avertit_sur_la_cle_privee(tmp_path, capsys):
    main(
        [
            "keygen",
            "--key-id",
            "x",
            "--private-out",
            str(tmp_path / "p.key"),
            "--public-out",
            str(tmp_path / "q.key"),
        ]
    )
    sortie = capsys.readouterr().out
    assert "n'est PAS chiffree" in sortie and "jamais le versionner" in sortie


# ---------- sign ----------


def test_signer_puis_verifier_avec_la_cle_publique(dossier, paire, capsys):
    priv, pub = paire
    assert main(["sign", dossier, "--key-id", "simon", "--private-key", priv]) == EXIT_OK
    capsys.readouterr()
    assert (
        main(["verify", dossier, "--public-key", pub, "--key-id", "simon"]) == EXIT_OK
    )
    assert "VALIDE" in capsys.readouterr().out


def test_la_signature_est_detachee(dossier, paire):
    """Elle ne doit jamais entrer dans le manifeste : elle signe son hash."""
    priv, _pub = paire
    main(["sign", dossier, "--key-id", "simon", "--private-key", priv])
    from pathlib import Path

    assert (Path(dossier) / "signature.json").is_file()
    manifeste = json.loads((Path(dossier) / "manifest.json").read_text(encoding="utf-8"))
    assert "signature" not in manifeste


def test_sign_hmac_annonce_sa_portee_limitee(dossier, cle_hmac, capsys):
    assert (
        main(["sign", dossier, "--key-id", "ci", "--hmac-key", cle_hmac]) == EXIT_OK
    )
    assert "integrite seulement" in capsys.readouterr().out


def test_sign_refuse_deux_cles(dossier, paire, cle_hmac, capsys):
    priv, _pub = paire
    code = main(
        [
            "sign",
            dossier,
            "--key-id",
            "x",
            "--private-key",
            priv,
            "--hmac-key",
            cle_hmac,
        ]
    )
    assert code == EXIT_ERROR
    assert "exactement une cle" in capsys.readouterr().err


def test_sign_refuse_sans_cle(dossier, capsys):
    assert main(["sign", dossier, "--key-id", "x"]) == EXIT_ERROR
    assert "exactement une cle" in capsys.readouterr().err


def test_sign_signale_une_cle_introuvable(dossier, tmp_path, capsys):
    code = main(
        ["sign", dossier, "--key-id", "x", "--hmac-key", str(tmp_path / "nulle-part")]
    )
    assert code == EXIT_ERROR
    assert "introuvable" in capsys.readouterr().err


def test_sign_signale_une_cle_non_hexadecimale(dossier, tmp_path, capsys):
    mauvaise = tmp_path / "mauvaise.key"
    mauvaise.write_text("ceci n'est pas de l'hexadecimal", encoding="utf-8")
    code = main(["sign", dossier, "--key-id", "x", "--hmac-key", str(mauvaise)])
    assert code == EXIT_ERROR
    assert "hexadecimal" in capsys.readouterr().err


# ---------- verify signe ----------


def test_verify_sans_signature_le_dit(dossier, capsys):
    assert main(["verify", dossier]) == EXIT_OK
    assert "signature             : absente" in capsys.readouterr().out


def test_verify_sans_cle_ne_pretend_PAS_avoir_verifie(dossier, paire, capsys):
    """Le piege a eviter : une signature presente mais non verifiee ne doit
    jamais ressembler a une signature valide."""
    priv, _pub = paire
    main(["sign", dossier, "--key-id", "simon", "--private-key", priv])
    capsys.readouterr()
    assert main(["verify", dossier]) == EXIT_OK
    sortie = capsys.readouterr().out
    assert "NON VERIFIEE" in sortie
    assert "VALIDE" not in sortie


def test_verify_json_marque_la_signature_non_verifiee(dossier, paire, capsys):
    priv, _pub = paire
    main(["sign", dossier, "--key-id", "simon", "--private-key", priv])
    capsys.readouterr()
    main(["verify", dossier, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["signature"]["present"] is True
    assert payload["signature"]["verified"] is False
    assert "valid" not in payload["signature"]


def test_verify_avec_une_mauvaise_cle_publique(dossier, paire, tmp_path, capsys):
    priv, _pub = paire
    main(["sign", dossier, "--key-id", "simon", "--private-key", priv])
    main(
        [
            "keygen",
            "--key-id",
            "etranger",
            "--private-out",
            str(tmp_path / "e_priv.key"),
            "--public-out",
            str(tmp_path / "e_pub.key"),
        ]
    )
    capsys.readouterr()
    code = main(
        [
            "verify",
            dossier,
            "--public-key",
            str(tmp_path / "e_pub.key"),
            "--key-id",
            "etranger",
        ]
    )
    assert code == EXIT_MISMATCH
    assert "INVALIDE" in capsys.readouterr().out


def test_verify_detecte_une_falsification_apres_signature(dossier, paire, capsys):
    """Le scenario complet : sceller, signer, falsifier, se faire prendre."""
    from pathlib import Path

    priv, pub = paire
    main(["sign", dossier, "--key-id", "simon", "--private-key", priv])

    chemin = Path(dossier) / "manifest.json"
    data = json.loads(chemin.read_text(encoding="utf-8"))
    data["backend_version"] = "0.0.0-mensonge"
    chemin.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    capsys.readouterr()
    code = main(["verify", dossier, "--public-key", pub, "--key-id", "simon"])
    assert code == EXIT_MISMATCH
    sortie = capsys.readouterr().out
    assert "FALSIFIE" in sortie and "INVALIDE" in sortie


def test_la_portee_decrit_l_algorithme_pas_l_issue(dossier, paire, tmp_path, capsys):
    """Non-regression : une signature ed25519 INVALIDE affichait « integrite
    seulement », ce qui laissait croire que l'algorithme etait symetrique. La
    portee decrit ce que l'algorithme permet ; l'echec est dit ailleurs."""
    from pathlib import Path

    priv, pub = paire
    main(["sign", dossier, "--key-id", "simon", "--private-key", priv])
    chemin = Path(dossier) / "manifest.json"
    data = json.loads(chemin.read_text(encoding="utf-8"))
    data["backend_version"] = "mensonge"
    chemin.write_text(json.dumps(data), encoding="utf-8")

    capsys.readouterr()
    main(["verify", dossier, "--public-key", pub, "--key-id", "simon"])
    sortie = capsys.readouterr().out
    assert "INVALIDE" in sortie
    assert "opposable a un tiers" in sortie, (
        "ed25519 reste opposable par nature, meme quand la signature echoue"
    )


def test_verify_refuse_une_cle_sans_signature_a_verifier(dossier, paire, capsys):
    priv, pub = paire
    code = main(["verify", dossier, "--public-key", pub, "--key-id", "simon"])
    assert code == EXIT_ERROR
    assert "n'existe pas" in capsys.readouterr().err


def test_verify_exige_un_key_id_avec_une_cle(dossier, paire, capsys):
    priv, pub = paire
    main(["sign", dossier, "--key-id", "simon", "--private-key", priv])
    capsys.readouterr()
    assert main(["verify", dossier, "--public-key", pub]) == EXIT_ERROR
    assert "key-id" in capsys.readouterr().err


def test_verify_refuse_deux_types_de_cle(dossier, paire, cle_hmac, capsys):
    priv, pub = paire
    main(["sign", dossier, "--key-id", "simon", "--private-key", priv])
    capsys.readouterr()
    code = main(
        [
            "verify",
            dossier,
            "--public-key",
            pub,
            "--hmac-key",
            cle_hmac,
            "--key-id",
            "simon",
        ]
    )
    assert code == EXIT_ERROR
    assert "OU" in capsys.readouterr().err


def test_le_cycle_hmac_complet(dossier, cle_hmac, capsys):
    assert main(["sign", dossier, "--key-id", "ci", "--hmac-key", cle_hmac]) == EXIT_OK
    capsys.readouterr()
    assert (
        main(["verify", dossier, "--hmac-key", cle_hmac, "--key-id", "ci"]) == EXIT_OK
    )
    sortie = capsys.readouterr().out
    assert "VALIDE" in sortie
    assert "integrite seulement" in sortie, "HMAC n'est pas opposable, il faut le dire"
