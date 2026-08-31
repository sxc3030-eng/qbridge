"""Adaptateurs vers les calibrations publiees par les fournisseurs.

Aucun de ces modules ne se connecte a quoi que ce soit : ils lisent des donnees
embarquees dans les paquets clients. Acceder au vrai materiel demande des
autorisations qui sortent du perimetre de qbridge.
"""

from qbridge.providers.google import PROCESSEURS, from_google_calibration

__all__ = ["from_google_calibration", "PROCESSEURS"]
