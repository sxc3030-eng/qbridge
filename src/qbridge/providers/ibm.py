"""Adaptateur : calibration reelle IBM -> `CalibrationSnapshot`.

CE QUE CES DONNEES ONT DEMONTRE. Le `fake_provider` de qiskit-ibm-runtime
embarque 69 instantanes de vrais appareils IBM, chacun etant un `props_*.json`
fige et versionne. Ils datent CHAQUE parametre separement, la ou Google publie
un horodatage unique.

Mesure sur `ibm_fez` (156 qubits, Heron r2), etiquete
`last_update_date: 2025-02-26` :

    4 060 mesures datees
    la plus ancienne : 2024-12-28
    la plus recente  : 2025-02-26
    ETALEMENT        : 60 jours

Un « instantane » porte donc des mesures qui s'etalent sur DEUX MOIS. Quiconque
le traite comme l'etat de l'appareil au 26 fevrier se trompe de soixante jours
sur certains parametres. C'est exactement pourquoi `DatedValue` date chaque
valeur et pourquoi `temporal_spread_seconds()` existe : masquer cet ecart
derriere la seule `last_update_date` serait mentir sur ce qu'on archive.

CONVERSIONS D'UNITES, PARCE QU'ELLES SE TROMPENT EN SILENCE. IBM publie T1 et
T2 en SECONDES (4.88e-05 pour 48.8 us) et `gate_length` en NANOSECONDES. Une
erreur d'unite ici produirait un modele de bruit absurde sans rien signaler ;
un test borne les valeurs converties.

CE QUE L'ADAPTATEUR NE FAIT PAS. Il ne se connecte a rien et ne demande aucun
jeton. Executer sur les VRAIS QPU d'IBM demande un compte IBM Quantum Platform
(gratuit) et passe par leur file d'attente ; ce module lit uniquement les
instantanes figes livres avec le paquet. Une archive produite ainsi a tourne
sur `hardware-sim`, et le manifeste le declare.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from qbridge.calibration import CalibrationSnapshot, DatedValue


def backends_disponibles() -> List[str]:
    """Noms des instantanes embarques dans qiskit-ibm-runtime."""
    from qiskit_ibm_runtime import fake_provider

    return sorted(
        nom
        for nom in dir(fake_provider)
        if nom.startswith("Fake") and not nom.endswith("V1")
    )


def _iso(valeur: Any) -> str:
    """Date ISO. Les dates IBM sont deja des datetime avec fuseau."""
    return valeur.isoformat() if hasattr(valeur, "isoformat") else str(valeur)


def from_ibm_backend(
    backend_name: Any = "FakeFez",
    *,
    qubits: Optional[Iterable[int]] = None,
) -> Tuple[CalibrationSnapshot, List[str]]:
    """Construit un instantane qbridge depuis un backend IBM, factice ou REEL.

    `backend_name` accepte un nom d'instantane embarque (`"FakeFez"`) ou un
    objet backend deja construit — y compris une VRAIE machine obtenue du
    service. Verifie sur `ibm_marrakesh` : un backend en direct expose le meme
    type `BackendProperties`, avec les memes dates par parametre, que les
    instantanes figes. C'est pourquoi il n'existe qu'un seul extracteur : deux
    finiraient par diverger.

    Rend `(instantane, avertissements)`. Les avertissements listent ce qui a
    ete converti, suppose ou omis — jamais rien en silence.

    `qubits` restreint l'instantane a un sous-ensemble d'indices. Sans cela,
    `ibm_fez` produit 156 qubits et ~1640 portes, soit un manifeste de plusieurs
    centaines de kilo-octets pour un circuit qui n'en utilise que cinq. Le
    sous-ensemble est un choix de l'appelant, et il est visible dans
    l'instantane : les qubits absents n'y figurent pas.

    Les indices IBM sont transposes en `cirq.LineQubit`, donc le qubit IBM 3
    devient la cle `q(3)`.
    """
    if isinstance(backend_name, str):
        from qiskit_ibm_runtime import fake_provider

        classe = getattr(fake_provider, backend_name, None)
        if classe is None:
            raise ValueError(
                f"Backend IBM inconnu : {backend_name!r}. "
                f"Exemples : {', '.join(backends_disponibles()[:6])}..."
            )
        backend = classe()
        etiquette = backend_name
    else:
        # Un backend deja construit : machine reelle ou factice, meme API.
        backend = backend_name
        etiquette = str(getattr(backend, "name", backend) or "inconnu")

    props = backend.properties()
    if props is None:
        raise ValueError(
            f"{etiquette} ne publie pas de proprietes calibrees : il n'y a "
            "rien a sceller."
        )

    avertissements: List[str] = []
    retenus = set(qubits) if qubits is not None else None
    if retenus is not None:
        avertissements.append(
            f"instantane restreint aux qubits {sorted(retenus)} : les autres "
            "qubits de l'appareil ne sont PAS scelles"
        )

    # ---- qubits : T1 et T2 sont en secondes chez IBM, en microsecondes ici
    donnees_qubits: Dict[str, Dict[str, DatedValue]] = {}
    index = 0
    while True:
        try:
            proprietes = props.qubit_property(index)
        except Exception:
            break
        if retenus is None or index in retenus:
            cle = f"q({index})"
            params: Dict[str, DatedValue] = {}
            for nom_ibm, nom_qbridge, facteur, unite in (
                ("T1", "t1_us", 1e6, "us"),
                ("T2", "t2_us", 1e6, "us"),
                ("readout_error", "readout_error", 1.0, ""),
                ("prob_meas0_prep1", "prob_meas0_prep1", 1.0, ""),
                ("prob_meas1_prep0", "prob_meas1_prep0", 1.0, ""),
            ):
                if nom_ibm in proprietes:
                    valeur, date = proprietes[nom_ibm]
                    params[nom_qbridge] = DatedValue(
                        float(valeur) * facteur, _iso(date), unite
                    )
            if params:
                donnees_qubits[cle] = params
        index += 1

    if not donnees_qubits:
        raise ValueError(
            f"Aucune donnee de qubit retenue pour {backend_name} "
            f"(qubits demandes : {sorted(retenus) if retenus else 'tous'})."
        )
    avertissements.append("T1 et T2 convertis de secondes vers microsecondes")

    # ---- portes
    donnees_portes: Dict[str, Dict[str, DatedValue]] = {}
    ignorees = 0
    for porte in props.gates:
        indices = list(porte.qubits)
        if retenus is not None and not set(indices) <= retenus:
            ignorees += 1
            continue
        cle = f"{porte.gate}:" + ",".join(f"q({i})" for i in indices)
        params = {}
        for parametre in porte.parameters:
            if parametre.name == "gate_error":
                params["gate_error"] = DatedValue(
                    float(parametre.value), _iso(parametre.date), ""
                )
            elif parametre.name == "gate_length":
                params["gate_length_ns"] = DatedValue(
                    float(parametre.value), _iso(parametre.date), parametre.unit or "ns"
                )
        if params:
            donnees_portes[cle] = params
    if ignorees:
        avertissements.append(
            f"{ignorees} portes ecartees car elles touchent des qubits hors du "
            "sous-ensemble retenu"
        )

    # ---- topologie, ramenee aux indices retenus
    try:
        config = backend.configuration()
    except Exception:
        # Un backend reel n'expose pas toujours `configuration()`. Perdre la
        # topologie est regrettable ; perdre T1, T2 et les erreurs de porte a
        # cause d'elle le serait bien plus.
        config = None
        avertissements.append(
            "topologie et portes de base indisponibles sur ce backend : "
            "l'instantane scelle les mesures, pas la carte de couplage"
        )
    coupling = []
    if config is not None and getattr(config, "coupling_map", None):
        for a, b in config.coupling_map:
            if retenus is None or {a, b} <= retenus:
                coupling.append([a, b])

    instantane = CalibrationSnapshot.build(
        device_id=f"ibm:{props.backend_name}",
        device_version=str(props.backend_version),
        qubits=donnees_qubits,
        gates=donnees_portes,
        basis_gates=list(getattr(config, "basis_gates", []) or [])
        if config is not None
        else [],
        coupling_map=coupling,
    )

    etalement = instantane.temporal_spread_seconds()
    if etalement > 0:
        avertissements.append(
            f"les mesures de cet instantane s'etalent sur "
            f"{etalement / 86400:.1f} jours (last_update_date = "
            f"{_iso(props.last_update_date)}) : ce n'est PAS l'etat de "
            "l'appareil a un instant"
        )
    return instantane, avertissements
