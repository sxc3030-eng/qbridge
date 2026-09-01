"""Enregistre VOTRE jeton IBM Quantum, en local, une seule fois.

A LIRE AVANT DE LANCER.

Ce script ne transmet votre jeton a personne. Il appelle `save_account` de
qiskit-ibm-runtime, qui l'ecrit dans votre trousseau local
(`~/.qiskit/qiskit-ibm.json`). qbridge ne le lit jamais, ne le stocke jamais et
ne l'affiche jamais.

TROIS PRECAUTIONS, ET ELLES COMPTENT.

1. Le jeton est demande par `getpass` : il ne s'affiche pas a l'ecran et
   n'entre pas dans l'historique de votre terminal. Ne le passez PAS en
   argument de ligne de commande - l'historique du shell le garderait.
2. Ne le collez dans aucune conversation, aucun ticket, aucun depot. Un jeton
   qui apparait quelque part est un jeton a revoquer.
3. Ce fichier ne contient aucun secret et peut etre versionne sans risque.
   Le fichier qu'il ECRIT, lui, ne doit jamais l'etre.

OU OBTENIR LA CLE - ET CE N'EST PLUS UN « JETON QUANTIQUE ».

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
faut en creer une autre - ce n'est pas grave, mais autant le savoir.

Il faut aussi qu'une INSTANCE du service Qiskit Runtime existe sur le compte
(plan Open, gratuit). Elle est generalement creee a l'inscription via la
plateforme quantique. `instance` reste optionnel ici : sans lui, le service
cherche toutes les instances accessibles au compte.
"""

from __future__ import annotations

import getpass
import sys

CANAL_PAR_DEFAUT = "ibm_quantum_platform"

LONGUEUR_CLE_ATTENDUE = 44
"""Longueur annoncee par le tableau de bord IBM lui-meme :
« use the 44-character API_KEY you created ».

Sert d'AVERTISSEMENT, jamais de refus : un format peut changer, et
rejeter une cle valide serait pire que d'en laisser passer une douteuse.
"""


CANAUX_VALIDES = ("ibm_quantum_platform", "ibm_cloud")
"""Les deux seuls noms que `save_account` accepte.

Ce ne sont PAS des identifiants : un canal designe la plateforme, pas le
compte. Un `IBMid-...`, un CRN ou une adresse courriel n'en sont pas.
"""


def instance_valide(saisie: str):
    """Rend (valeur, probleme) pour une instance de service.

    `save_account(instance=...)` accepte « the CRN or service name ». C'est
    OPTIONNEL : sans instance, le service cherche toutes celles du compte.
    Un CRN n'est pas un secret — il nomme une ressource, il n'authentifie
    rien. Il ne remplace donc jamais la cle.

    valeur None avec probleme None signifie « aucune instance, et c'est bien ».
    """
    valeur = saisie.strip()
    if not valeur:
        return None, None

    if valeur.lower().startswith("ibmid-"):
        return None, (
            "un IBMid identifie votre PERSONNE, pas une instance de service"
        )
    if "@" in valeur:
        return None, "une adresse courriel n'est pas une instance"

    if valeur.startswith("crn:"):
        # Un CRN complet compte 10 segments, donc 9 deux-points. En dessous de
        # 8, la copie est tronquee — cas frequent quand le CRN est coupe par
        # la largeur d'une fenetre.
        if valeur.count(":") < 8:
            return None, "CRN incomplet — la copie a probablement ete tronquee"
        return valeur, None

    # Tout le reste est traite comme un nom de service, ce que l'API accepte.
    return valeur, None


def canal_valide(saisie: str):
    """Rend le canal retenu, ou None si la saisie n'en est pas un.

    Valider AVANT de reclamer la cle : sinon une faute de frappe sur un champ
    anodin fait saisir un secret pour rien, et le message d'erreur arrive
    apres coup. Un champ doit refuser ce qu'il peut refuser tot.
    """
    canal = saisie.strip() or CANAL_PAR_DEFAUT
    return canal if canal in CANAUX_VALIDES else None


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

    print("Le CANAL designe la plateforme, pas votre compte : ce n'est ni un")
    print("IBMid, ni un CRN, ni une adresse courriel. Dans le doute, ENTREE.")
    print()

    canal = None
    for essai in range(3):
        saisie = input(f"Canal - ENTREE pour [{CANAL_PAR_DEFAUT}] : ")
        canal = canal_valide(saisie)
        if canal is not None:
            break
        print(
            f"  {saisie.strip()!r} n'est pas un canal. Attendus : "
            f"{', '.join(CANAUX_VALIDES)} - ou ENTREE pour le defaut.",
            file=sys.stderr,
        )
    if canal is None:
        print(file=sys.stderr)
        print("Canal invalide, rien n'a ete demande ni ecrit.",
              file=sys.stderr)
        return 1

    print()
    print("L'INSTANCE est facultative : c'est le CRN de votre service")
    print("Qiskit Runtime, ou son nom. Sans elle, toutes les instances du")
    print("compte sont cherchees. Ce n'est PAS un secret et ce n'est pas")
    print("la cle. Dans le doute, ENTREE.")
    print()

    instance = None
    for _ in range(3):
        saisie = input("Instance / CRN - ENTREE pour aucune : ")
        instance, probleme = instance_valide(saisie)
        if probleme is None:
            break
        print(f"  refuse : {probleme}.", file=sys.stderr)
    else:
        print("Instance invalide, rien n'a ete demande ni ecrit.",
              file=sys.stderr)
        return 1

    # La cle n'est reclamee qu'une fois le canal ET l'instance valides :
    # ce qui est gratuit se verifie avant ce qui est secret.
    jeton = getpass.getpass("Cle API IBM Cloud (invisible) : ").strip()

    if not jeton:
        print("Aucune cle saisie, rien n'a ete ecrit.", file=sys.stderr)
        return 1
    if len(jeton) < 20:
        # Une saisie tres courte est une erreur de copie, pas une cle.
        # L'ecrire ferait echouer plus tard sur un message obscur.
        print(
            f"La cle saisie fait {len(jeton)} caracteres, ce qui est "
            "anormalement court. Rien n'a ete ecrit - verifiez la copie.",
            file=sys.stderr,
        )
        return 1
    if len(jeton) != LONGUEUR_CLE_ATTENDUE:
        # AVERTISSEMENT seulement : le format peut changer, et refuser une
        # cle valide serait pire que la laisser passer. Mais une cle
        # tronquee a la copie est frequente et vaut d'etre signalee ici
        # plutot que dix lignes plus bas sous forme d'erreur reseau.
        print(
            f"  avertissement : {len(jeton)} caracteres, alors qu'IBM "
            f"annonce {LONGUEUR_CLE_ATTENDUE}. On continue, mais si la "
            "suite echoue, suspectez une copie incomplete.",
            file=sys.stderr,
        )

    try:
        supplement = {"instance": instance} if instance else {}
        if not instance:
            # `plans_preference` est IGNORE si une instance est donnee. Sans
            # instance, il fait prioriser le plan gratuit — meme intention que
            # `verifier_le_plan` : une depense ne doit pas arriver en silence.
            supplement["plans_preference"] = ["open"]
        QiskitRuntimeService.save_account(
            channel=canal,
            token=jeton,
            overwrite=True,
            set_as_default=True,
            **supplement,
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
