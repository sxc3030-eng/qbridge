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

CE QUI EST HONNETEMENT INCERTAIN. Le chemin reseau — soumission, file
d'attente, recuperation — n'a jamais tourne contre un vrai QPU au moment ou ces
lignes sont ecrites. La conversion, la transpilation et le depaquetage des
resultats sont valides contre un backend factice, et le depaquetage a ete
confronte aux comptages de Qiskit lui-meme. Le reste attend un compte.
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
    def _depaqueter(champ: Any) -> np.ndarray:
        """Bits empaquetes de Qiskit -> tableau (repetitions, n_qubits) uint8.

        Qiskit rend des octets empaquetes ; qbridge attend un bit par colonne.
        Le depaquetage a ete confronte aux comptages de Qiskit lui-meme sur un
        GHZ : correspondance exacte.
        """
        deplie = np.unpackbits(champ.array, axis=1, bitorder="big")
        return deplie[:, -champ.num_bits :].astype(np.uint8)

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


def backend_reel(nom_appareil: Optional[str] = None, **kwargs: Any):
    """Ouvre un backend IBM REEL. Demande un compte deja configure.

    Ne recoit ni ne lit aucun jeton : `QiskitRuntimeService()` lit celui que
    VOUS avez enregistre une fois pour toutes avec `save_account`. Si aucun
    compte n'est configure, l'erreur le dit clairement plutot que de reclamer
    des identifiants.

    Sans `nom_appareil`, IBM choisit le moins charge.
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

    appareil = (
        service.backend(nom_appareil)
        if nom_appareil
        else service.least_busy(operational=True, simulator=False)
    )
    return IbmRuntimeBackend(appareil, **kwargs)
