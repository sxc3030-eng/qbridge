"""Backend IBM : soumet un vrai job a un vrai QPU.

CE QUE CE MODULE PROUVE. Toute l'architecture repose sur une affirmation :
« le jour ou une vraie machine est branchee, seule `sample()` change ; le
contrat, le manifeste et les verdicts restent identiques ». Ce fichier est la
verification de cette affirmation, et il tient en une classe.

POURQUOI IL PREND UN OBJET BACKEND ET NON UN NOM. Un backend qui exige des
identifiants ne peut pas etre construit depuis une simple chaine : c'est
pourquoi il n'est PAS dans le registre `BACKENDS` et se passe en objet a
`capture(backend=...)`. Consequence heureuse : un backend FACTICE d'IBM expose
exactement la meme interface qu'un vrai pour `transpile` et `SamplerV2`, donc
tout ce fichier se teste hors ligne. Le seul pas non testable sans jeton est
l'obtention du backend reel aupres du service.

CE MODULE NE VOIT JAMAIS VOTRE JETON. Il recoit un objet backend deja
construit. C'est a vous d'executer, une seule fois et vous-meme :

    QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="...")

Le jeton vit alors dans votre trousseau local. Ni qbridge, ni ce fichier, ni
aucun manifeste ne le lit, ne le stocke ni ne l'affiche.

CE MODULE A TOURNE CONTRE UN VRAI QPU. Le 2026-09-01, sur `ibm_marrakesh`
(156 qubits) : GHZ a trois qubits, 1024 tirages, 12 secondes, 97.3 % sur
`000`+`111`. Le chemin reseau complet — soumission, file d'attente,
recuperation, depaquetage — est donc exerce, et plus seulement le chemin
factice. Voir `examples/premier_job_reel.py`.

CE QUI RESTE INCERTAIN. Une seule execution, sur une seule machine, avec un
circuit minuscule. Rien n'est verifie sur : les circuits profonds, les mesures
en cours de circuit sur materiel, les files d'attente longues, les erreurs
reseau en cours de job, ni les autres appareils. Un succes n'est pas une
couverture.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import cirq
import numpy as np


class IbmRuntimeBackend:
    """Execute un circuit cirq sur un backend Qiskit, reel ou factice.

    Le contrat est celui d'une machine, jamais celui d'un simulateur :
    pas de vecteur d'etat, pas de reproductibilite bit-a-bit.
    """

    name = "ibm-runtime"
    USES_QSIM_KERNEL = False
    """Aucune ligne de qsim ne s'execute ici. Sceller son empreinte
    decrirait un simulateur qui n'a pas tourne."""
    BIT_EXACT_REPLAYABLE = False
    """Une vraie machine ne rejoue pas ses tirages. Ici ce n'est meme plus un
    contrat prudent : c'est la physique."""

    def __init__(
        self,
        qiskit_backend: Any,
        *,
        optimization_level: int = 1,
    ) -> None:
        self._backend = qiskit_backend
        self._optimization_level = optimization_level
        nom = getattr(qiskit_backend, "name", None) or "inconnu"
        if callable(nom):  # certaines versions exposent name()
            nom = nom()
        self.device_name = str(nom)
        self.version = f"{self.device_name}"
        self.derniere_transpilation: Optional[Dict[str, Any]] = None
        """Provenance de la derniere transpilation : le placement logique vers
        physique et le compte de portes. Sur du vrai materiel, c'est une donnee
        de provenance de premier ordre — deux transpilations differentes ne sont
        pas la meme experience."""

    def is_bit_exact_replayable(self) -> bool:
        return type(self).BIT_EXACT_REPLAYABLE

    def simulate(
        self,
        circuit: cirq.Circuit,
        *,
        seed: int,
        options: Dict[str, Any],
        noise: Optional[cirq.NoiseModel] = None,
    ) -> np.ndarray:
        raise NotImplementedError(
            "Une machine quantique ne rend pas de vecteur d'etat : le "
            "no-cloning interdit d'en copier un et la mesure est destructive. "
            "Utiliser `repetitions=` pour echantillonner."
        )

    # ---------- conversion ----------

    @staticmethod
    def _vers_qiskit(circuit: cirq.Circuit) -> Any:
        """cirq -> QASM 2 -> Qiskit.

        Passer par QASM plutot que par un convertisseur direct est deliberе :
        c'est un format que les deux ecosystemes lisent, et il reste
        inspectable si la conversion surprend.
        """
        from qiskit import qasm2

        return qasm2.loads(
            cirq.qasm(circuit),
            custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
        )

    @staticmethod
    def _operations_physiques(circuit_transpile: Any) -> Dict[str, int]:
        """Les operations REELLEMENT executees, par qubit physique.

        POURQUOI PAS `count_ops()`. Il dit combien de `cz`, jamais lesquels.
        Sur `ibm_marrakesh`, les paires cz scellees vont de 1.65e-3 a 3.63e-3 :
        un facteur 2.2. Predire avec une moyenne quand la transpilation a
        utilise deux fois la paire bruyante sous-estime l'erreur a deux qubits
        de 27 %, sans que rien ne le signale.

        Les cles suivent EXACTEMENT le format de `CalibrationSnapshot.gates`
        (`"cz:q(0),q(1)"`), pour que la prediction soit un lookup direct et non
        une traduction — une traduction serait un endroit de plus ou se tromper.
        """
        operations: Dict[str, int] = {}
        for instruction in circuit_transpile.data:
            indices = tuple(
                circuit_transpile.find_bit(bit).index for bit in instruction.qubits
            )
            cle = instruction.operation.name + ":" + ",".join(
                f"q({i})" for i in indices
            )
            operations[cle] = operations.get(cle, 0) + 1
        return dict(sorted(operations.items()))

    @staticmethod
    def _depaqueter(champ: Any) -> np.ndarray:
        """Bits empaquetes de Qiskit -> tableau (repetitions, n_qubits) uint8.

        Qiskit rend des octets empaquetes ; qbridge attend un bit par colonne.
        Le depaquetage a ete confronte aux comptages de Qiskit lui-meme sur un
        GHZ : correspondance exacte.
        """
        deplie = np.unpackbits(champ.array, axis=1, bitorder="big")
        return deplie[:, -champ.num_bits :].astype(np.uint8)

    # ---------- etat de l'appareil ----------

    def device_provenance(self):
        """Ce que l'appareil a reellement execute. None avant toute execution."""
        return self.derniere_transpilation

    def device_calibration(self):
        """Instantane DATE de l'appareil, restreint aux qubits reellement usites.

        Rend `(instantane, avertissements)`, ou None si l'appareil ne publie
        rien d'exploitable — un simulateur factice sans proprietes, par exemple.

        POURQUOI RESTREINT. `ibm_marrakesh` publie 156 qubits et 2 420 portes.
        Tout sceller produirait des centaines de kilo-octets pour un circuit
        qui touche trois qubits. Le placement issu de la transpilation dit
        exactement lesquels comptent, donc il pilote la restriction.

        POURQUOI APRES L'EXECUTION. Le placement n'existe qu'une fois la
        transpilation faite. Demander la calibration avant reviendrait a tout
        sceller, ou a sceller les mauvais qubits.
        """
        from qbridge.providers.ibm import from_ibm_backend

        trace = self.derniere_transpilation
        qubits = None
        if trace is not None and trace.get("initial_layout"):
            qubits = [int(i) for i in trace["initial_layout"]]

        try:
            return from_ibm_backend(self._backend, qubits=qubits)
        except Exception as exc:
            # Ne JAMAIS faire echouer une execution reussie parce que la
            # calibration est illisible. On perd l'etat d'appareil, et le
            # manifeste doit le dire plutot que de le taire.
            return None, [
                f"etat d'appareil NON scelle : {type(exc).__name__} : {exc}"
            ]

    # ---------- execution ----------

    def sample(
        self,
        circuit: cirq.Circuit,
        *,
        repetitions: int,
        seed: int,
        options: Dict[str, Any],
        noise: Optional[cirq.NoiseModel] = None,
    ) -> Dict[str, np.ndarray]:
        """Soumet le circuit et rend les mesures.

        `seed` sert au transpileur, PAS au materiel : une machine quantique
        n'accepte pas de graine. Le fixer rend la transpilation reproductible,
        ce qui est deja beaucoup — deux transpilations differentes du meme
        circuit logique ne sont pas la meme experience physique.
        """
        if noise is not None:
            raise ValueError(
                "Un backend materiel porte son propre bruit : passer un modele "
                "separe le contredirait sans que rien ne le signale."
            )
        if options:
            raise ValueError(
                f"Le backend {self.name!r} n'accepte pas les options qsim "
                f"(recu : {sorted(options)}). Elles decrivent un simulateur, "
                "pas une machine."
            )

        from qiskit import transpile
        from qiskit_ibm_runtime import SamplerV2

        logique = self._vers_qiskit(circuit)
        physique = transpile(
            logique,
            backend=self._backend,
            optimization_level=self._optimization_level,
            seed_transpiler=seed,
        )

        placement = None
        if getattr(physique, "layout", None) is not None:
            try:
                placement = list(
                    physique.layout.initial_index_layout(filter_ancillas=True)
                )
            except Exception:  # pragma: no cover - variantes d'API
                placement = None
        self.derniere_transpilation = {
            "device": self.device_name,
            "optimization_level": self._optimization_level,
            "seed_transpiler": seed,
            "initial_layout": placement,
            "gate_counts": {k: int(v) for k, v in physique.count_ops().items()},
            "operations": self._operations_physiques(physique),
            "depth": int(physique.depth()),
        }

        resultat = SamplerV2(mode=self._backend).run(
            [physique], shots=repetitions
        ).result()
        donnees = resultat[0].data

        mesures: Dict[str, np.ndarray] = {}
        for nom in dir(donnees):
            if nom.startswith("_"):
                continue
            champ = getattr(donnees, nom)
            if hasattr(champ, "array") and hasattr(champ, "num_bits"):
                # cirq nomme le registre `m_m` pour une cle de mesure `m` :
                # on retire le prefixe pour retrouver la cle d'origine.
                cle = nom[2:] if nom.startswith("m_") else nom
                mesures[cle] = self._depaqueter(champ)
        if not mesures:
            raise ValueError(
                "Aucun registre de mesure dans le resultat : le circuit "
                "contient-il bien une mesure ?"
            )
        return mesures


PLANS_GRATUITS = frozenset({"open", "lite"})
"""Plans IBM sans facturation a l'usage.

Liste d'AUTORISATION et non d'interdiction : un plan inconnu est traite comme
payant. Se tromper dans ce sens fait perdre une soumission ; se tromper dans
l'autre fait perdre de l'argent, et le temps QPU se facture a la minute.
"""


def _plans_du_compte(service: Any) -> list:
    """Plans des instances accessibles. Liste vide si l'information manque."""
    try:
        return [
            str(instance.get("plan", "")).lower()
            for instance in service.instances()
        ]
    except Exception:
        # On ne DEDUIT rien d'un echec : une liste vide veut dire « je ne sais
        # pas », et l'appelant traite l'ignorance comme un risque.
        return []


def verifier_le_plan(service: Any, autoriser_plan_payant: bool = False) -> list:
    """Refuse de continuer si aucun plan gratuit n'est visible.

    Le temps QPU se facture a la minute chez IBM. `backend_reel()` sans
    argument prend la machine la moins chargee SANS regarder le plan : sur un
    compte mixte, cela pourrait viser une machine payante. Ce controle existe
    pour qu'une depense ne puisse pas arriver en silence — meme discipline que
    partout ailleurs dans ce projet.

    Rend la liste des plans vus, pour que l'appelant puisse l'afficher.
    """
    plans = _plans_du_compte(service)
    if autoriser_plan_payant:
        return plans

    if not plans:
        raise RuntimeError(
            "Impossible de determiner le plan de votre compte IBM. Le temps "
            "QPU se facture a la minute : on ne soumet pas sans savoir. "
            "Verifiez votre instance sur cloud.ibm.com, ou passez "
            "`autoriser_plan_payant=True` si vous acceptez la facturation en "
            "connaissance de cause."
        )

    gratuits = [p for p in plans if p in PLANS_GRATUITS]
    if not gratuits:
        raise RuntimeError(
            f"Aucun plan gratuit sur ce compte (plans vus : {sorted(set(plans))}). "
            f"Plans consideres comme gratuits : {sorted(PLANS_GRATUITS)}. "
            "Le temps QPU se facture a la minute. Pour soumettre malgre tout, "
            "passer explicitement `autoriser_plan_payant=True`."
        )
    return plans


def backend_reel(
    nom_appareil: Optional[str] = None,
    *,
    autoriser_plan_payant: bool = False,
    **kwargs: Any,
):
    """Ouvre un backend IBM REEL. Demande un compte deja configure.

    Ne recoit ni ne lit aucun jeton : `QiskitRuntimeService()` lit celui que
    VOUS avez enregistre une fois pour toutes avec `save_account`. Si aucun
    compte n'est configure, l'erreur le dit clairement plutot que de reclamer
    des identifiants.

    Sans `nom_appareil`, IBM choisit le moins charge — mais SANS regarder le
    plan, d'ou le controle de `verifier_le_plan` avant toute soumission.
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "qiskit-ibm-runtime est absent. `pip install qbridge[ibm]`."
        ) from exc

    try:
        service = QiskitRuntimeService()
    except Exception as exc:
        raise RuntimeError(
            "Aucun compte IBM Quantum configure. A executer UNE FOIS, "
            "vous-meme, avec votre propre jeton :\n\n"
            "    from qiskit_ibm_runtime import QiskitRuntimeService\n"
            "    QiskitRuntimeService.save_account(\n"
            '        channel="ibm_quantum_platform", token="VOTRE_JETON")\n\n'
            "qbridge ne lit, ne stocke et n'affiche jamais ce jeton.\n"
            f"Detail : {exc}"
        ) from exc

    verifier_le_plan(service, autoriser_plan_payant)

    appareil = (
        service.backend(nom_appareil)
        if nom_appareil
        else service.least_busy(operational=True, simulator=False)
    )
    return IbmRuntimeBackend(appareil, **kwargs)
