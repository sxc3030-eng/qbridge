"""Adaptateurs vers les calibrations publiees par les fournisseurs.

Aucun de ces modules ne se connecte a quoi que ce soit : ils lisent des donnees
embarquees dans les paquets clients. Acceder au vrai materiel demande des
autorisations qui sortent du perimetre de qbridge.

Les deux fournisseurs ne publient PAS la meme chose, et c'est instructif :

- Google donne un horodatage unique pour tout l'instantane (calibration
  mediane), mais pas les durees de porte ;
- IBM date chaque parametre separement, et l'etalement mesure sur `ibm_fez`
  atteint 60 JOURS sous une seule `last_update_date`.

C'est cette seconde observation qui justifie que `DatedValue` porte une date
par valeur plutot qu'une date par instantane.
"""

from qbridge.providers.google import PROCESSEURS, from_google_calibration
from qbridge.providers.ibm import backends_disponibles, from_ibm_backend

__all__ = [
    "from_google_calibration",
    "PROCESSEURS",
    "from_ibm_backend",
    "backends_disponibles",
]
