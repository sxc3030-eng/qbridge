"""Empreinte de l'environnement d'execution.

L'empreinte generale n'entre PAS dans le hash semantique : elle sert a
expliquer une divergence, pas a la provoquer. Le noyau SIMD est traite a part
parce qu'il PEUT changer le resultat : la wheel qsimcirq embarque quatre
noyaux (AVX512F / AVX2 / SSE4.1 / basic) et en choisit un par CPUID a l'import.
"""

from __future__ import annotations

import os
import platform
import sys
from typing import Any, Dict


def kernel_fingerprint() -> Dict[str, Any]:
    """Identifie le noyau de calcul reellement charge par qsimcirq."""
    import qsimcirq

    module = getattr(qsimcirq.qsim, "__name__", "unknown")
    try:
        jeu = int(qsimcirq.qsim_decide.detect_instructions())
    except Exception:
        jeu = -1
    try:
        gpu = int(qsimcirq.qsim_decide.detect_gpu())
    except Exception:
        gpu = -1
    return {
        "qsim_kernel_module": module,
        "qsim_instruction_set": jeu,  # 0=AVX512F 1=AVX2 2=SSE4.1 3=basic -1=inconnu
        "qsim_gpu_mode": gpu,  # 10 = pas de GPU
    }


def environment_fingerprint() -> Dict[str, Any]:
    """Capture machine, bibliotheques et noyau de calcul."""
    import cirq
    import numpy
    import qsimcirq

    fp: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count() or 0,
        "cirq_version": cirq.__version__,
        "qsimcirq_version": qsimcirq.__version__,
        "numpy_version": numpy.__version__,
    }
    fp.update(kernel_fingerprint())
    return fp
