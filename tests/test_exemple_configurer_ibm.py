"""Le script d'enregistrement ne doit jamais reclamer une cle pour rien.

Defaut vecu : le canal etait demande, puis la cle, puis seulement `qiskit`
rejetait le canal. L'utilisateur avait saisi un secret pour rien. Un champ doit
refuser tot ce qu'il peut refuser tot.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_chemin = pathlib.Path(__file__).parent.parent / "examples" / "configurer_ibm.py"
_spec = importlib.util.spec_from_file_location("configurer_ibm", _chemin)
configurer_ibm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(configurer_ibm)


def test_entree_vide_donne_le_canal_par_defaut():
    assert configurer_ibm.canal_valide("") == "ibm_quantum_platform"
    assert configurer_ibm.canal_valide("   ") == "ibm_quantum_platform"


def test_les_deux_canaux_reels_passent():
    assert configurer_ibm.canal_valide("ibm_quantum_platform") == "ibm_quantum_platform"
    assert configurer_ibm.canal_valide("ibm_cloud") == "ibm_cloud"
    assert configurer_ibm.canal_valide("  ibm_cloud  ") == "ibm_cloud"


@pytest.mark.parametrize(
    "saisie",
    [
        "IBMid-693001YU6L",        # LE cas vecu : un identifiant de compte
        "crn:v1:bluemix:public:",  # un CRN d'instance
        "sxc3030@gmail.com",       # une adresse courriel
        "ibm_quantum",             # l'ancien nom, retire apres migration
        "IBM_CLOUD",               # la casse compte
    ],
)
def test_ce_qui_n_est_pas_un_canal_est_refuse(saisie):
    """Chacune de ces valeurs est quelque chose qu'on a sous la main en
    creant un compte IBM, et qu'on peut coller dans le champ par confusion."""
    assert configurer_ibm.canal_valide(saisie) is None


def test_le_script_ne_contient_aucun_secret():
    """Ce fichier est versionne : il doit rester lisible par tous."""
    texte = _chemin.read_text(encoding="utf-8")
    assert "getpass" in texte, "la cle doit etre saisie sans echo"
    for interdit in ("token=\"sk", "apikey=", "IBMid-693001"):
        assert interdit not in texte
