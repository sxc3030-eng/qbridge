"""Instantane de calibration d'un appareil quantique.

CE QU'UN INSTANTANE EST VRAIMENT. Pas l'etat d'un appareil a un instant : un
SAC DE MESURES DATEES SEPAREMENT. Dans le `props_fez.json` d'IBM, T1 est mesure
le 26 fevrier a 06h56 et `readout_error` le 24 fevrier — deux jours d'ecart
dans un meme « instantane ». Chaque parametre porte donc sa propre date, et
`temporal_spread_seconds()` rend l'ecart entre la plus vieille et la plus
recente. Cacher cet ecart derriere une date unique serait mentir sur ce qu'on
archive.

CE QU'ON SCELLE ET POURQUOI. On seale les DONNEES datees, pas le modele de
bruit qu'on en derive. La derivation est du code, et le code change ; les
mesures, elles, sont un fait historique. Cirq fait le meme choix :
`GoogleNoiseProperties` est serialisable, `NoiseModelFromGoogleNoiseProperties`
ne l'est pas. Aer va plus loin dans la demonstration — `NoiseModel.from_dict()`
y est deprecie depuis 0.15, ce qui casserait toute archive qui aurait scelle un
modele plutot que ses donnees sources.

LA DERIVATION EST VOLONTAIREMENT SIMPLE. Relaxation d'amplitude depuis T1,
depolarisation depuis l'erreur de la porte PRECISE (`gate_error_for`, jamais la
moyenne tant qu'une donnee specifique existe), inversion de bit avant mesure
depuis l'erreur de lecture. Aucune diaphonie, aucun terme ZZ, aucune erreur
correlee. Ce n'est PAS un modele de bruit fidele a un appareil reel,
et ce module ne pretend pas l'etre : son role est d'eprouver le contrat du
harnais, pas de reproduire une physique. Un modele fidele viendra du fournisseur
le jour ou une vraie machine est branchee — le manifeste, lui, ne changera pas
de forme.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cirq

from qbridge.digest import sha256_of

CALIBRATION_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class DatedValue:
    """Une mesure et la date a laquelle elle a ete prise.

    La date est par PARAMETRE, jamais par instantane : c'est la seule facon
    honnete de representer ce que les fournisseurs publient reellement.
    """

    value: float
    date: str
    unit: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "date": self.date, "unit": self.unit}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatedValue":
        return cls(
            value=float(data["value"]),
            date=str(data["date"]),
            unit=str(data.get("unit", "")),
        )


@dataclass(frozen=True)
class CalibrationSnapshot:
    """Caracterisation datee d'un appareil, scellable dans un manifeste."""

    schema_version: str
    device_id: str
    device_version: str
    qubits: Dict[str, Dict[str, DatedValue]]
    """Par qubit (cle = repr du qubit) : t1_us, t2_us, readout_error,
    prob_meas0_prep1, prob_meas1_prep0."""
    gates: Dict[str, Dict[str, DatedValue]]
    """Par porte (cle = "nom:qubits") : gate_error, gate_length_ns."""
    basis_gates: List[str]
    coupling_map: List[List[int]]
    snapshot_hash: str = field(default="")

    # ---------- construction ----------

    @classmethod
    def build(
        cls,
        *,
        device_id: str,
        device_version: str,
        qubits: Dict[str, Dict[str, DatedValue]],
        gates: Dict[str, Dict[str, DatedValue]],
        basis_gates: List[str],
        coupling_map: List[List[int]],
    ) -> "CalibrationSnapshot":
        brut = cls(
            schema_version=CALIBRATION_SCHEMA_VERSION,
            device_id=device_id,
            device_version=device_version,
            qubits=qubits,
            gates=gates,
            basis_gates=list(basis_gates),
            coupling_map=[list(paire) for paire in coupling_map],
        )
        return cls(**{**brut.__dict__, "snapshot_hash": brut._compute_hash()})

    def _compute_hash(self) -> str:
        donnees = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "snapshot_hash"
        }
        donnees["qubits"] = {
            q: {k: v.to_dict() for k, v in params.items()}
            for q, params in self.qubits.items()
        }
        donnees["gates"] = {
            g: {k: v.to_dict() for k, v in params.items()}
            for g, params in self.gates.items()
        }
        return sha256_of(donnees)

    def verify_self(self) -> None:
        attendu = self._compute_hash()
        if attendu != self.snapshot_hash:
            raise ValueError(
                "Echec du controle d'integrite de la calibration : "
                f"hash stocke={self.snapshot_hash[:16]}... "
                f"recalcule={attendu[:16]}..."
            )

    # ---------- la verite sur les dates ----------

    def all_dates(self) -> List[str]:
        """Toutes les dates de mesure presentes, triees."""
        dates = [v.date for params in self.qubits.values() for v in params.values()]
        dates += [v.date for params in self.gates.values() for v in params.values()]
        return sorted(dates)

    def temporal_spread_seconds(self) -> float:
        """Ecart entre la mesure la plus ancienne et la plus recente.

        Non nul dans la vraie vie. Un appelant qui traite un instantane comme
        un etat instantane doit pouvoir constater de combien il se trompe.
        """
        dates = self.all_dates()
        if len(dates) < 2:
            return 0.0
        debut = _dt.datetime.fromisoformat(dates[0])
        fin = _dt.datetime.fromisoformat(dates[-1])
        return (fin - debut).total_seconds()

    # ---------- acces ----------

    def qubit_param(self, qubit: cirq.Qid, name: str) -> Optional[float]:
        params = self.qubits.get(str(qubit))
        if not params or name not in params:
            return None
        return params[name].value

    def gate_param(self, gate_key: str, name: str) -> Optional[float]:
        params = self.gates.get(gate_key)
        if not params or name not in params:
            return None
        return params[name].value

    @staticmethod
    def _qubits_de_cle(cle: str) -> frozenset:
        """Ensemble des qubits designes par une cle `nom:q0,q1`.

        On compare des ENSEMBLES et non des chaines : les fournisseurs ne
        publient qu'une entree par paire de qubits, sans garantir l'ordre dans
        lequel un circuit presentera cette paire. C'est une convention de
        RECHERCHE, pas une affirmation que la porte est symetrique.
        """
        _, _, qubits = cle.partition(":")
        return frozenset(q.strip() for q in qubits.split(",") if q.strip())

    def _param_for(
        self, operation: "cirq.Operation", nom_param: str, defaut_global: float
    ) -> float:
        """Valeur d'un parametre pour la porte PRECISE, avec repli documente.

        Chaine de repli, du plus specifique au plus vague :

        1. cle exacte `nom:qubits` ;
        2. meme nom de porte sur le MEME ENSEMBLE de qubits, quel que soit
           l'ordre — `CZ(q1,q0)` doit trouver l'entree `cz:q(0),q(1)` ;
        3. n'importe quelle porte sur ce meme ensemble de qubits ;
        4. moyenne des portes de MEME ARITE ;
        5. valeur globale fournie par l'appelant.

        Les etapes 2 et 4 corrigent des defauts mesures.

        Sans l'etape 2, `CZ(q1,q0)` tombait jusqu'a la moyenne globale : 0.0177
        au lieu de 0.050 sur un instantane ou la porte a deux qubits vaut 0.050.
        Le bruit de la porte a deux qubits — source d'erreur dominante sur du
        vrai materiel — etait sous-estime d'un facteur 3.

        Sans l'etape 4, une porte a deux qubits absente de l'instantane
        heritait d'une moyenne que les portes a un qubit tirent vers le bas.
        Un repli doit degrader vers plus de bruit, jamais vers moins.

        Erreur et duree partagent cette chaine : le meme raisonnement vaut pour
        les deux, et les separer avait laisse la duree sur une moyenne globale
        pendant que l'erreur etait deja corrigee.
        """
        noms = [str(q) for q in operation.qubits]
        ensemble = frozenset(noms)
        arite = len(operation.qubits)
        nom = str(operation.gate).lower().split("(")[0].strip()

        exacte = self.gate_param(f"{nom}:{','.join(noms)}", nom_param)
        if exacte is not None:
            return exacte

        for cle, params in self.gates.items():
            if nom_param not in params:
                continue
            if cle.split(":", 1)[0] == nom and self._qubits_de_cle(cle) == ensemble:
                return params[nom_param].value

        for cle, params in self.gates.items():
            if nom_param in params and self._qubits_de_cle(cle) == ensemble:
                return params[nom_param].value

        memes = [
            params[nom_param].value
            for cle, params in self.gates.items()
            if nom_param in params and len(self._qubits_de_cle(cle)) == arite
        ]
        if memes:
            return sum(memes) / len(memes)

        return defaut_global

    def gate_error_for(self, operation: "cirq.Operation") -> float:
        """Erreur de la porte PRECISE. Voir `_param_for` pour la chaine de
        repli."""
        return self._param_for(operation, "gate_error", self.mean_gate_error())

    def gate_length_for(self, operation: "cirq.Operation") -> float:
        """Duree de la porte PRECISE, en nanosecondes.

        Mesure sur `ibm_fez` qui a motive cette methode :

            x, sx, rx, id :   24 ns
            cz, rzz       :   84 ns
            rz            :    0 ns   (rotation virtuelle : correct)
            reset         : 1584 ns   (66x une porte normale)

            mean_gate_length_ns() = 210 ns

        Cette moyenne sert a convertir T1 en probabilite de relaxation. Une
        porte X de 24 ns se voyait donc appliquer la relaxation de 210 ns, soit
        NEUF FOIS trop, parce qu'un `reset` tres long ecrasait la moyenne. Le
        sens de l'erreur (trop de bruit) etait le moins dangereux, mais l'ordre
        de grandeur ne l'etait pas.

        Une duree nulle est une donnee legitime, pas une absence : sur du vrai
        materiel IBM, `rz` est une rotation virtuelle implementee comme un
        changement de phase de reference, et ne dure effectivement rien.
        """
        return self._param_for(
            operation, "gate_length_ns", self.mean_gate_length_ns()
        )

    def mean_gate_error(self) -> float:
        """Erreur de porte moyenne. Dernier repli de `gate_error_for`, jamais
        le chemin principal."""
        valeurs = [
            p["gate_error"].value for p in self.gates.values() if "gate_error" in p
        ]
        return sum(valeurs) / len(valeurs) if valeurs else 0.0

    def mean_gate_length_ns(self) -> float:
        valeurs = [
            p["gate_length_ns"].value
            for p in self.gates.values()
            if "gate_length_ns" in p
        ]
        return sum(valeurs) / len(valeurs) if valeurs else 0.0

    # ---------- serialisation ----------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "device_id": self.device_id,
            "device_version": self.device_version,
            "qubits": {
                q: {k: v.to_dict() for k, v in params.items()}
                for q, params in self.qubits.items()
            },
            "gates": {
                g: {k: v.to_dict() for k, v in params.items()}
                for g, params in self.gates.items()
            },
            "basis_gates": list(self.basis_gates),
            "coupling_map": [list(p) for p in self.coupling_map],
            "snapshot_hash": self.snapshot_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationSnapshot":
        version = data.get("schema_version")
        if version != CALIBRATION_SCHEMA_VERSION:
            raise ValueError(
                f"Version de schema de calibration incompatible : {version!r} "
                f"(attendu {CALIBRATION_SCHEMA_VERSION!r})."
            )
        return cls(
            schema_version=version,
            device_id=data["device_id"],
            device_version=data["device_version"],
            qubits={
                q: {k: DatedValue.from_dict(v) for k, v in params.items()}
                for q, params in data["qubits"].items()
            },
            gates={
                g: {k: DatedValue.from_dict(v) for k, v in params.items()}
                for g, params in data["gates"].items()
            },
            basis_gates=list(data["basis_gates"]),
            coupling_map=[list(p) for p in data["coupling_map"]],
            snapshot_hash=data["snapshot_hash"],
        )

    def to_json(self) -> str:
        from qbridge.digest import canonical_json

        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "CalibrationSnapshot":
        return cls.from_dict(json.loads(text))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationSnapshot":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ---------- derivation du modele de bruit ----------

    def noise_model(self) -> cirq.NoiseModel:
        """Derive un modele de bruit depuis les donnees datees.

        La derivation est du CODE, elle n'est jamais scellee : c'est
        l'instantane qui l'est. Le jour ou cette fonction change, une archive
        ancienne reste interpretable — on relit ses donnees et on redérive.
        """
        return _CalibrationNoiseModel(self)


class _CalibrationNoiseModel(cirq.NoiseModel):
    """Modele de bruit par qubit derive d'un instantane.

    Deliberement simple, et il faut le dire : relaxation d'amplitude depuis T1
    et la duree de porte, depolarisation depuis l'erreur de porte, inversion de
    bit avant mesure depuis l'erreur de lecture. Aucune diaphonie, aucun terme
    ZZ, aucune erreur correlee. Le but est d'eprouver le contrat du harnais,
    pas de reproduire un appareil.
    """

    def __init__(self, snapshot: CalibrationSnapshot) -> None:
        self._snap = snapshot

    def _readout_error(self, qubit: cirq.Qid) -> float:
        valeur = self._snap.qubit_param(qubit, "readout_error")
        return 0.0 if valeur is None else max(0.0, min(0.5, valeur))

    def _damping(self, qubit: cirq.Qid, duree_ns: float) -> float:
        """Probabilite de relaxation pendant la duree REELLE de la porte.

        La duree vient de `gate_length_for`, pas d'une moyenne globale : sur
        `ibm_fez`, un `reset` de 1584 ns tirait la moyenne a 210 ns et faisait
        appliquer neuf fois trop de relaxation a une porte X de 24 ns.
        """
        import math

        t1_us = self._snap.qubit_param(qubit, "t1_us")
        if not t1_us or t1_us <= 0 or duree_ns <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - math.exp(-(duree_ns / 1000.0) / t1_us)))

    def noisy_moment(self, moment: cirq.Moment, system_qubits):
        if self.is_virtual_moment(moment):
            return moment

        mesures = {
            q
            for op in moment.operations
            if cirq.is_measurement(op)
            for q in op.qubits
        }
        if mesures:
            # L'erreur de lecture s'applique AVANT la mesure : c'est l'etat lu
            # qui est errone, pas le bit une fois classique.
            #
            # Et UNIQUEMENT aux qubits reellement mesures. L'appliquer a tous
            # les qubits du moment injectait une inversion de bit reelle dans
            # l'etat de qubits qui n'etaient pas lus, inversion qui persistait
            # ensuite dans tout le circuit — une propriete de l'appareil de
            # mesure transformee en erreur physique sur des qubits voisins.
            avant = [
                cirq.bit_flip(self._readout_error(q)).on(q)
                for q in sorted(mesures)
                if self._readout_error(q) > 0
            ]
            resultat = [cirq.Moment(avant)] if avant else []
            resultat.append(moment)
            return resultat

        # Deux moments SEPARES : un cirq.Moment n'accepte pas deux operations
        # sur le meme qubit, et depolarisation puis relaxation s'appliquent
        # bien l'une apres l'autre, pas simultanement.
        depolarisation = []
        relaxation = []
        # Erreur ET duree sont prises PAR OPERATION, jamais en moyenne : ce sont
        # les donnees que l'instantane a scellees avec leur propre date.
        erreur_par_qubit = {}
        duree_par_qubit = {}
        for operation in moment.operations:
            erreur = self._snap.gate_error_for(operation)
            duree = self._snap.gate_length_for(operation)
            for qubit in operation.qubits:
                erreur_par_qubit[qubit] = erreur
                duree_par_qubit[qubit] = duree

        for qubit in sorted(moment.qubits):
            p = erreur_par_qubit.get(qubit, 0.0)
            if p > 0:
                depolarisation.append(cirq.depolarize(min(p, 0.75)).on(qubit))
            gamma = self._damping(qubit, duree_par_qubit.get(qubit, 0.0))
            if gamma > 0:
                relaxation.append(cirq.amplitude_damp(gamma).on(qubit))

        resultat = [moment]
        if depolarisation:
            resultat.append(cirq.Moment(depolarisation))
        if relaxation:
            resultat.append(cirq.Moment(relaxation))
        return resultat


# --------------------------------------------------------------------------
# Fabrique d'instantane synthetique, pour les tests et la demonstration
# --------------------------------------------------------------------------


def synthetic_snapshot(
    qubits: Tuple[cirq.Qid, ...],
    *,
    device_id: str = "qbridge-demo",
    device_version: str = "1.0",
    base_date: str = "2026-08-28T06:56:41+00:00",
    spread_hours: float = 48.0,
) -> CalibrationSnapshot:
    """Fabrique un instantane plausible, avec des dates VOLONTAIREMENT etalees.

    L'etalement est le point : il reproduit ce que publient les vrais
    fournisseurs, ou les parametres d'un meme « instantane » sont mesures a
    plusieurs jours d'intervalle.
    """
    debut = _dt.datetime.fromisoformat(base_date)
    tardif = (debut + _dt.timedelta(hours=spread_hours)).isoformat()

    donnees_qubits: Dict[str, Dict[str, DatedValue]] = {}
    for index, qubit in enumerate(qubits):
        donnees_qubits[str(qubit)] = {
            # T1 et T2 mesures tot, lecture mesuree deux jours plus tard :
            # exactement le decalage observe chez IBM.
            "t1_us": DatedValue(48.8 + index * 0.7, debut.isoformat(), "us"),
            "t2_us": DatedValue(42.4 + index * 0.5, debut.isoformat(), "us"),
            "readout_error": DatedValue(0.0115 + index * 0.0004, tardif, ""),
            "prob_meas0_prep1": DatedValue(0.0132 + index * 0.0003, tardif, ""),
            "prob_meas1_prep0": DatedValue(0.0098 + index * 0.0002, tardif, ""),
        }

    donnees_portes: Dict[str, Dict[str, DatedValue]] = {}
    for index, qubit in enumerate(qubits):
        donnees_portes[f"x:{qubit}"] = {
            "gate_error": DatedValue(7.64e-4 + index * 1e-5, debut.isoformat(), ""),
            "gate_length_ns": DatedValue(24.0, debut.isoformat(), "ns"),
        }
    for index in range(len(qubits) - 1):
        cle = f"cz:{qubits[index]},{qubits[index + 1]}"
        donnees_portes[cle] = {
            "gate_error": DatedValue(6.2e-3 + index * 1e-4, tardif, ""),
            "gate_length_ns": DatedValue(68.0, tardif, "ns"),
        }

    return CalibrationSnapshot.build(
        device_id=device_id,
        device_version=device_version,
        qubits=donnees_qubits,
        gates=donnees_portes,
        basis_gates=["cz", "id", "rz", "sx", "x"],
        coupling_map=[[i, i + 1] for i in range(len(qubits) - 1)],
    )
