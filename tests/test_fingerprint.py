from qbridge.digest import canonical_json
from qbridge.fingerprint import environment_fingerprint, kernel_fingerprint


def test_contient_les_versions():
    fp = environment_fingerprint()
    assert fp["cirq_version"] and fp["qsimcirq_version"] and fp["numpy_version"]


def test_contient_la_plateforme():
    fp = environment_fingerprint()
    for cle in ("python_version", "platform", "machine", "processor", "cpu_count"):
        assert cle in fp


def test_contient_le_noyau_simd_reellement_charge():
    fp = environment_fingerprint()
    assert fp["qsim_kernel_module"].startswith("qsimcirq.qsim")
    assert isinstance(fp["qsim_instruction_set"], int)


def test_le_noyau_est_isolable():
    k = kernel_fingerprint()
    assert set(k) == {"qsim_kernel_module", "qsim_instruction_set", "qsim_gpu_mode"}


def test_serialisable_en_json_canonique():
    canonical_json(environment_fingerprint())


def test_stable_dans_un_meme_processus():
    assert environment_fingerprint() == environment_fingerprint()


def test_aucune_valeur_none():
    assert all(v is not None for v in environment_fingerprint().values())
