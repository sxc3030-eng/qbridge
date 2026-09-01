"""Enregistre VOTRE jeton IBM Quantum, en local, une seule fois.

A LIRE AVANT DE LANCER.

Ce script ne transmet votre jeton a personne. Il appelle `save_account` de
qiskit-ibm-runtime, qui l'ecrit dans votre trousseau local
(`~/.qiskit/qiskit-ibm.json`). qbridge ne le lit jamais, ne le stocke jamais et
ne l'affiche jamais.

TROIS PRECAUTIONS, ET ELLES COMPTENT.

1. Le jeton est demande par `getpass` : il ne s'affiche pas a l'ecran et
   n'entre pas dans l'historique de votre terminal. Ne le passez PAS en
   argument de ligne de commande — l'historique du shell le garderait.
2. Ne le collez dans aucune conversation, aucun ticket, aucun depot. Un jeton
   qui apparait quelque part est un jeton a revoquer.
3. Ce fichier ne contient aucun secret et peut etre versionne sans risque.
   Le fichier qu'il ECRIT, lui, ne doit jamais l'etre.

OU OBTENIR LA CLE — ET CE N'EST PLUS UN « JETON QUANTIQUE ».

La documentation de `save_account` dans la version installee est explicite :

    token : IBM Cloud API key.
    url   : defaults to https://cloud.ibm.com

Depuis la migration vers IBM Cloud, ce n'est plus le jeton affiche par
l'ancienne plateforme quantum.ibm.com : c'est une CLE API IBM CLOUD, creee dans
la gestion des acces d'IBM Cloud. Chercher un « API token » dans les pages
quantiques mene a la documentation des API, pas a votre compte.

Le chemin :

    cloud.ibm.com  ->  Manage  ->  Access (IAM)  ->  API keys
                   ->  Create  ->  copier la cle IMMEDIATEMENT

La cle n'est affichee QU'UNE FOIS. Si vous fermez la fenetre sans la copier, il
faut en creer une autre — ce n'est pas grave, mais autant le savoir.

Il faut aussi qu'une INSTANCE du service Qiskit Runtime existe sur le compte
(plan Open, gratuit). Elle est generalement creee a l'inscription via la
plateforme quantique. `instance` reste optionnel ici : sans lui, le service
cherche toutes les instances accessibles au compte.
"""

from __future__ import annotations

import getpass
import sys

CANAL_PAR_DEFAUT = "ibm_quantum_platform"


def main() -> int:
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        print(
            "qiskit-ibm-runtime est absent.\n"
            "  pip install qiskit-ibm-runtime",
            file=sys.stderr,
        )
        return 1

    print("Enregistrement d'un compte IBM Quantum")
    print("--------------------------------------")
    print("Ce qui est attendu est une CLE API IBM CLOUD, et non un jeton de")
    print("l'ancienne plateforme quantique. Elle se cree ici :")
    print()
    print("    cloud.ibm.com -> Manage -> Access (IAM) -> API keys -> Create")
    print()
    print("Elle n'est affichee QU'UNE FOIS : copiez-la avant de fermer.")
    print("La saisie ci-dessous est INVISIBLE. La cle est ecrite dans votre")
    print("trousseau local, nulle part ailleurs.\n")

    canal = input(f"Canal [{CANAL_PAR_DEFAUT}] : ").strip() or CANAL_PAR_DEFAUT
    jeton = getpass.getpass("Cle API IBM Cloud (invisible) : ").strip()

    if not jeton:
        print("Aucune cle saisie, rien n'a ete ecrit.", file=sys.stderr)
        return 1
    if len(jeton) < 20:
        # Garde-fou : un jeton IBM est long. Une saisie trop courte est
        # probablement une erreur de copie, et l'ecrire ferait echouer plus
        # tard avec un message obscur.
        print(
            f"La cle saisie fait {len(jeton)} caracteres, ce qui est "
            "anormalement court. Rien n'a ete ecrit — verifiez la copie.",
            file=sys.stderr,
        )
        return 1

    try:
        QiskitRuntimeService.save_account(
            channel=canal,
            token=jeton,
            overwrite=True,
            set_as_default=True,
        )
    except Exception as exc:
        # On n'affiche jamais le jeton, meme dans un message d'erreur.
        print(f"Echec de l'enregistrement : {type(exc).__name__} : {exc}",
              file=sys.stderr)
        print(
            "\nSi le canal est refuse, ouvrez votre tableau de bord IBM "
            "Quantum : il affiche l'extrait de code avec le nom de canal "
            "attendu par votre compte.",
            file=sys.stderr,
        )
        return 1
    finally:
        # Le jeton ne traine pas en memoire plus longtemps que necessaire.
        # Ce n'est pas une garantie forte en Python, mais c'est gratuit.
        jeton = ""

    print("\nCompte enregistre.")
    print("Verification de l'acces...\n")

    try:
        service = QiskitRuntimeService()
        appareils = service.backends(operational=True, simulator=False)
    except Exception as exc:
        print(f"Compte ecrit, mais l'acces echoue : {exc}", file=sys.stderr)
        return 1

    if not appareils:
        print("Aucun appareil operationnel visible pour ce compte.")
        return 0

    print(f"{len(appareils)} appareil(s) accessible(s) :")
    for appareil in appareils:
        try:
            file_attente = appareil.status().pending_jobs
        except Exception:
            file_attente = "?"
        print(f"  {appareil.name:24} {appareil.num_qubits:>4} qubits   "
              f"file d'attente : {file_attente}")

    print(
        "\nRien d'autre a faire : qbridge lira ce compte via "
        "`backend_reel()`, sans jamais toucher au jeton."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
