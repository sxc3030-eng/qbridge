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

OU OBTENIR LE JETON. Sur la plateforme IBM Quantum, apres creation du compte :
le jeton figure dans le tableau de bord, section API token. Les URL et le nom
du canal ont change lors de la migration vers la nouvelle plateforme IBM Cloud —
si `channel` ci-dessous est refuse, la page d'accueil de votre compte affiche le
bon extrait de code, copiez le nom de canal qu'elle indique.
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
    print("Le jeton ne sera PAS affiche pendant la saisie.")
    print("Il sera ecrit dans votre trousseau local, nulle part ailleurs.\n")

    canal = input(f"Canal [{CANAL_PAR_DEFAUT}] : ").strip() or CANAL_PAR_DEFAUT
    jeton = getpass.getpass("Jeton API (invisible) : ").strip()

    if not jeton:
        print("Aucun jeton saisi, rien n'a ete ecrit.", file=sys.stderr)
        return 1
    if len(jeton) < 20:
        # Garde-fou : un jeton IBM est long. Une saisie trop courte est
        # probablement une erreur de copie, et l'ecrire ferait echouer plus
        # tard avec un message obscur.
        print(
            f"Le jeton saisi fait {len(jeton)} caracteres, ce qui est "
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
