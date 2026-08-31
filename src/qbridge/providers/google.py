"""Adaptateur : calibration reelle Google -> `CalibrationSnapshot`.

CE QUE CET ADAPTATEUR APPORTE. Jusqu'ici qbridge n'avait vu que des instantanes
synthetiques, fabriques pour eprouver le protocole. Ceux-ci viennent de vraies
machines : `rainbow` (23 qubits), `weber` (53) et `willow_pink` (105), publies
par Google et embarques dans `cirq-google`. Aucun compte, aucune autorisation :
ce sont des donnees ouvertes.

DEUX DIFFERENCES REELLES AVEC IBM, QU'IL FAUT DIRE.

1. Google publie UN SEUL horodatage pour tout l'instantane, la ou IBM date
   chaque parametre separement. `temporal_spread_seconds()` vaudra donc 0 sur
   ces instantanes — ce n'est pas un bug, c'est la nature d'une calibration
   MEDIANE : elle agrege deja, et Google ne publie pas la date de chaque mesure
   individuelle. Le champ par-parametre reste utile pour IBM, et honnete ici :
   il dit simplement que toutes les valeurs portent la meme date.

2. Google ne publie PAS les durees de porte dans cette calibration. Or il en
   faut une pour deriver la relaxation depuis T1. Elles sont donc fournies par
   l'APPELANT, jamais devinees en silence, et leur provenance est ecrite dans
   l'instantane scelle (champ `unit`) pour que personne ne les prenne plus tard
   pour des mesures de Google.

CE QUE L'ADAPTATEUR NE FAIT PAS. Il ne se connecte a rien. `load_median_device_
calibration` lit des fichiers embarques dans le paquet. Acceder au VRAI
materiel Google demande un partenariat de recherche approuve et un projet Google
Cloud ; ce n'est pas ce que fait ce module, et rien ici ne doit laisser croire
qu'une archive produite avec ces donnees a tourne sur une machine.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Tuple

from qbridge.calibration import CalibrationSnapshot, DatedValue

PROCESSEURS = ("rainbow", "weber", "willow_pink")
"""Processeurs dont la calibration est embarquee dans cirq-google."""

_UNITE_SUPPOSEE = "ns (fourni par l'appelant, ABSENT de la calibration Google)"
"""Marque la provenance d'une valeur que Google ne publie pas.

Elle est scellee avec l'instantane : dans cinq ans, personne ne pourra prendre
cette duree pour une mesure du fournisseur.
"""


def _iso(timestamp_ms: int) -> str:
    return _dt.datetime.fromtimestamp(
        timestamp_ms / 1000, _dt.timezone.utc
    ).isoformat()


def _premiere(valeurs: Any) -> float:
    """Les metriques cirq-google rendent une liste d'une seule valeur."""
    if isinstance(valeurs, (list, tuple)):
        return float(valeurs[0])
    return float(valeurs)


def from_google_calibration(
    processor_id: str,
    *,
    single_qubit_gate_ns: float = 25.0,
    two_qubit_gate_ns: float = 32.0,
) -> Tuple[CalibrationSnapshot, List[str]]:
    """Construit un instantane qbridge depuis une calibration Google reelle.

    Rend `(instantane, avertissements)`. Les avertissements ne sont jamais
    silencieux : ils listent ce qui a ete suppose plutot que mesure, et ce que
    la calibration ne contenait pas.

    `single_qubit_gate_ns` et `two_qubit_gate_ns` ont des valeurs par defaut
    plausibles pour cette classe de materiel, mais ce sont des HYPOTHESES DE
    L'APPELANT : Google ne publie pas les durees de porte ici. Elles servent a
    convertir T1 en probabilite de relaxation. Leur provenance est inscrite
    dans l'instantane.
    """
    if processor_id not in PROCESSEURS:
        raise ValueError(
            f"Processeur inconnu : {processor_id!r}. "
            f"Disponibles : {', '.join(PROCESSEURS)}"
        )

    from cirq_google.engine import (
        create_device_from_processor_id,
        load_median_device_calibration,
    )

    calibration = load_median_device_calibration(processor_id)
    device = create_device_from_processor_id(processor_id)
    date = _iso(calibration.timestamp)
    avertissements: List[str] = [
        "les durees de porte ne figurent pas dans la calibration Google : "
        f"{single_qubit_gate_ns} ns et {two_qubit_gate_ns} ns sont des "
        "hypotheses de l'appelant",
        "Google publie un horodatage unique pour tout l'instantane : "
        "l'etalement temporel par parametre vaut 0 par construction",
    ]

    def metrique(nom: str) -> Dict[Tuple, Any]:
        try:
            return calibration[nom]
        except (KeyError, TypeError):
            avertissements.append(f"metrique absente de cet instantane : {nom}")
            return {}

    t1 = metrique("single_qubit_idle_t1_micros")
    p00 = metrique("single_qubit_p00_error")
    p11 = metrique("single_qubit_p11_error")
    rb = metrique("single_qubit_rb_pauli_error_per_gate")
    cz = metrique("two_qubit_parallel_cz_gate_xeb_pauli_error_per_cycle")

    qubits: Dict[str, Dict[str, DatedValue]] = {}
    for cle, valeurs in t1.items():
        (qubit,) = cle
        qubits.setdefault(str(qubit), {})["t1_us"] = DatedValue(
            _premiere(valeurs), date, "us"
        )
    # p00_error : mesurer 1 alors qu'on a prepare 0, et reciproquement pour p11.
    for source, nom in ((p00, "prob_meas1_prep0"), (p11, "prob_meas0_prep1")):
        for cle, valeurs in source.items():
            (qubit,) = cle
            qubits.setdefault(str(qubit), {})[nom] = DatedValue(
                _premiere(valeurs), date, ""
            )
    # `readout_error` est la moyenne des deux directions. C'est une SIMPLIFICATION :
    # les deux ne sont pas symetriques sur du vrai materiel, et le modele de
    # bruit de qbridge n'applique qu'une inversion de bit unique.
    for nom_qubit, params in qubits.items():
        deux = [
            params[k].value
            for k in ("prob_meas0_prep1", "prob_meas1_prep0")
            if k in params
        ]
        if deux:
            params["readout_error"] = DatedValue(sum(deux) / len(deux), date, "")
    if p00 and p11:
        avertissements.append(
            "readout_error est la moyenne de prob_meas0_prep1 et "
            "prob_meas1_prep0 : le modele de bruit de qbridge n'applique qu'une "
            "inversion de bit symetrique, alors que le vrai materiel ne l'est pas"
        )

    gates: Dict[str, Dict[str, DatedValue]] = {}
    for cle, valeurs in rb.items():
        (qubit,) = cle
        gates[f"x:{qubit}"] = {
            "gate_error": DatedValue(_premiere(valeurs), date, ""),
            "gate_length_ns": DatedValue(
                single_qubit_gate_ns, date, _UNITE_SUPPOSEE
            ),
        }
    for cle, valeurs in cz.items():
        a, b = cle
        gates[f"cz:{a},{b}"] = {
            "gate_error": DatedValue(_premiere(valeurs), date, ""),
            "gate_length_ns": DatedValue(two_qubit_gate_ns, date, _UNITE_SUPPOSEE),
        }

    indices = {str(q): i for i, q in enumerate(sorted(device.metadata.qubit_set))}
    coupling = [
        [indices[str(a)], indices[str(b)]]
        for a, b in sorted(device.metadata.nx_graph.edges())
        if str(a) in indices and str(b) in indices
    ]

    instantane = CalibrationSnapshot.build(
        device_id=f"google:{processor_id}",
        device_version=_iso(calibration.timestamp),
        qubits=qubits,
        gates=gates,
        basis_gates=["cz", "x", "y", "z", "phased_x_z"],
        coupling_map=coupling,
    )
    return instantane, avertissements
