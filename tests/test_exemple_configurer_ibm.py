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
        "IBMid-0000000000",        # LE cas vecu : un identifiant de compte
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


# ---------- l'instance de service (CRN) ----------


def test_aucune_instance_est_un_cas_normal():
    """L'instance est FACULTATIVE : vide n'est pas une erreur."""
    assert configurer_ibm.instance_valide("") == (None, None)
    assert configurer_ibm.instance_valide("   ") == (None, None)


def test_un_crn_complet_est_accepte():
    crn = ("crn:v1:bluemix:public:quantum-computing:us-east:"
           "a/1a2b3c4d5e:6f7g8h9i-jklm::")
    valeur, probleme = configurer_ibm.instance_valide(crn)
    assert probleme is None
    assert valeur == crn


def test_un_crn_tronque_est_refuse():
    """Cas frequent : le CRN est coupe par la largeur de la fenetre. L'accepter
    ferait echouer plus tard avec un message sans rapport."""
    valeur, probleme = configurer_ibm.instance_valide(
        "crn:v1:bluemix:public:quantum-computing"
    )
    assert valeur is None
    assert "tronquee" in probleme


def test_un_nom_de_service_est_accepte():
    """L'API accepte « the CRN or service name »."""
    assert configurer_ibm.instance_valide("mon-service") == ("mon-service", None)


def test_un_ibmid_n_est_pas_une_instance():
    """LE cas vecu, transpose : un IBMid identifie une personne."""
    valeur, probleme = configurer_ibm.instance_valide("IBMid-0000000000")
    assert valeur is None
    assert "PERSONNE" in probleme


def test_un_courriel_n_est_pas_une_instance():
    valeur, probleme = configurer_ibm.instance_valide("sxc3030@gmail.com")
    assert valeur is None
    assert "courriel" in probleme


def test_le_crn_n_est_jamais_traite_comme_un_secret():
    """Un CRN nomme une ressource, il n'authentifie rien. Il doit passer par
    `input` visible, jamais par `getpass` — le confondre avec la cle est
    exactement la confusion que ce script doit dissiper."""
    texte = _chemin.read_text(encoding="utf-8")
    avant_cle = texte.split("getpass.getpass")[0]
    assert "Instance / CRN" in avant_cle, "l'instance doit etre demandee AVANT la cle"
    assert "getpass" not in texte.split("Instance / CRN")[0].split("canal_valide")[-1]
