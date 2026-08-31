"""Backend materiel : le contrat d'une vraie machine, sans vraie machine.

CE QUE CE BACKEND EST — ET N'EST PAS. Il ne simule pas fidelement un appareil
reel et ne pretend pas le faire. Il implemente son CONTRAT, c'est-a-dire les
trois choses qu'un simulateur laisse croire et qu'une machine reelle interdit :

1. `simulate()` est IMPOSSIBLE. On ne lit pas le vecteur d'etat d'une machine :
   le no-cloning interdit d'en prendre une copie et la mesure est destructive.
   Ici cela leve `NotImplementedError`, pas un tableau approximatif.
2. `is_bit_exact_replayable()` vaut False. Le bruit physique n'est pas rejouable.
   `replay` plafonne donc le verdict a STATISTICALLY_COMPATIBLE, meme quand les
   octets coincident. C'est un CONTRAT, pas une mesure — sous le capot qsim est
   deterministe a seed fixe, et c'est precisement pour ca qu'il ne faut pas
   deduire le verdict de ce qu'on observe.
3. L'execution depend d'un etat d'appareil DATE, qui doit etre scelle dans le
   manifeste. Sans lui, rejouer ne veut rien dire : on comparerait a un appareil
   dont on ignore l'etat.

C'est cette troisieme contrainte qui met le protocole a l'epreuve. Les deux
premieres se declarent ; celle-la oblige le manifeste a porter quelque chose
qu'il ne portait pas.

POURQUOI CE N'EST PAS UNE TRICHERIE. Le jour ou une vraie machine est branchee,
seule la methode `sample()` change — elle postera un job au lieu d'appeler qsim.
Le contrat, le manifeste et les verdicts restent identiques. Si le protocole
tient avec ce backend-ci, il tiendra avec l'autre ; s'il ne tenait pas, on le
saurait maintenant plutot qu'au moment ou le materiel arrive.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import cirq
import numpy as np

from qbridge.calibration import CalibrationSnapshot


class SimulatedHardwareBackend:
    """Backend adosse a un instantane de calibration fige.

    Nomme `hardware-sim` et non `hardware` : le nom doit dire que c'est un
    substitut. Un manifeste qui declare `hardware-sim` ne pretendra jamais
    venir d'une vraie machine.
    """

    name = "hardware-sim"
    BIT_EXACT_REPLAYABLE = False
    """Attribut de CLASSE : lisible sans construire d'instance.

    Le plafond de verdict doit pouvoir interroger le backend de CAPTURE,
    qu'on ne peut pas toujours instancier (le backend materiel exige une
    calibration). Construire un temoin dans un try/except revenait a
    desactiver le plafond en silence des que la construction echouait."""

    def __init__(self, calibration: Optional[CalibrationSnapshot] = None) -> None:
        self._calibration = calibration
        self.version = (
            f"{calibration.device_id}@{calibration.device_version}"
            if calibration is not None
            else "sans-calibration"
        )

    # ---------- le contrat ----------

    def is_bit_exact_replayable(self) -> bool:
        # Lit l'attribut de classe : instance et classe ne peuvent pas
        # diverger, il n'y a qu'une source de verite.
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
            "Un backend materiel ne peut pas rendre de vecteur d'etat : le "
            "no-cloning interdit d'en copier un et la mesure est destructive. "
            "Utiliser `repetitions=` pour echantillonner — c'est la seule chose "
            "qu'une machine reelle sait produire."
        )

    def sample(
        self,
        circuit: cirq.Circuit,
        *,
        repetitions: int,
        seed: int,
        options: Dict[str, Any],
        noise: Optional[cirq.NoiseModel] = None,
    ) -> Dict[str, np.ndarray]:
        """Echantillonne sous le bruit derive de la calibration scellee."""
        if self._calibration is None:
            raise ValueError(
                "Ce backend exige un instantane de calibration : sans lui, on "
                "echantillonnerait un appareil dont l'etat n'est pas connu, et "
                "le rejeu ne prouverait rien. Passer `calibration=` a capture()."
            )
        self._calibration.verify_self()

        # Le bruit est DERIVE de la calibration, jamais fourni de l'exterieur :
        # accepter un `noise` ici permettrait de contredire silencieusement
        # l'etat d'appareil scelle.
        if noise is not None:
            raise ValueError(
                "Ce backend derive son bruit de la calibration scellee. Passer "
                "un modele de bruit separe le contredirait sans que rien ne le "
                "signale."
            )

        import qsimcirq

        from qbridge.tiers import known_options

        for cle in options:
            if cle not in known_options():
                raise KeyError(f"Option qsim inconnue : {cle!r}.")

        sim = qsimcirq.QSimSimulator(
            qsimcirq.QSimOptions(**options),
            seed=seed,
            noise=self._calibration.noise_model(),
        )
        return dict(sim.run(circuit, repetitions=repetitions).measurements)

    # ---------- provenance ----------

    def calibration(self) -> Optional[CalibrationSnapshot]:
        return self._calibration
