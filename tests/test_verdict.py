import numpy as np

from qbridge.verdict import (
    Verdict,
    chi2_homogeneity_pvalue,
    compare_samples,
    compare_state_vectors,
)


def test_vecteurs_identiques_donnent_bit_exact():
    a = np.array([0.7071068, 0, 0, 0.7071068], dtype=np.complex64)
    assert compare_state_vectors(a, a.copy()).verdict is Verdict.BIT_EXACT


def test_ecart_d_arrondi_donne_numeriquement_equivalent():
    # Piege : ajouter 3e-9 ne change RIEN — l'ULP de complex64 vaut ~6e-8 pres
    # de 0.707, donc un tel delta n'est pas representable et les octets restent
    # identiques. On prend l'ecart representable le plus petit possible.
    a = np.array([0.7071068, 0, 0, 0.7071068], dtype=np.complex64)
    b = a.copy()
    b[0] = np.complex64(np.nextafter(np.float32(a[0].real), np.float32(1.0)))
    assert a.tobytes() != b.tobytes(), "les deux vecteurs doivent vraiment differer"

    r = compare_state_vectors(a, b)
    assert r.verdict is Verdict.NUMERICALLY_EQUIVALENT and r.infidelity < 1e-6
    assert r.max_abs_delta is not None and r.max_abs_delta > 0.0


def test_max_abs_delta_expose_un_ecart_que_l_infidelite_ne_voit_pas():
    # L'infidelite est quadratique pres de 1 : un ecart de 1e-8 par amplitude
    # produit une infidelite de ~1e-16, invisible en float64. Sans max_abs_delta
    # le rapport annoncerait "0.000e+00" pour deux vecteurs qui different.
    rng = np.random.default_rng(0)
    n = 4096
    a = rng.normal(size=n).astype(np.float32)
    a = (a / np.linalg.norm(a)).astype(np.complex64)
    b = (a + np.complex64(1e-8)).astype(np.complex64)

    r = compare_state_vectors(a, b)
    assert r.verdict is Verdict.NUMERICALLY_EQUIVALENT
    assert r.infidelity < 1e-9, "l'infidelite doit bien saturer ici"
    assert r.max_abs_delta > 0.0, "max|delta| doit revealer l'ecart reel"


def test_bit_exact_rapporte_un_ecart_nul():
    a = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.complex64)
    r = compare_state_vectors(a, a.copy())
    assert r.verdict is Verdict.BIT_EXACT and r.max_abs_delta == 0.0


def test_vecteurs_differents_donnent_divergent():
    a = np.array([1, 0, 0, 0], dtype=np.complex64)
    b = np.array([0, 0, 0, 1], dtype=np.complex64)
    assert compare_state_vectors(a, b).verdict is Verdict.DIVERGENT


def test_formes_incompatibles_donnent_divergent():
    a = np.array([1, 0], dtype=np.complex64)
    b = np.array([1, 0, 0, 0], dtype=np.complex64)
    assert compare_state_vectors(a, b).verdict is Verdict.DIVERGENT


def test_echantillons_identiques_donnent_bit_exact():
    s = {"m": np.array([[0, 0], [1, 1]], dtype=np.uint8)}
    assert compare_samples(s, {"m": s["m"].copy()}).verdict is Verdict.BIT_EXACT


def test_memes_distributions_donnent_statistiquement_compatible():
    rng = np.random.default_rng(0)
    a = {"m": rng.integers(0, 2, size=(4000, 1), dtype=np.uint8)}
    b = {"m": rng.integers(0, 2, size=(4000, 1), dtype=np.uint8)}
    assert compare_samples(a, b).verdict is Verdict.STATISTICALLY_COMPATIBLE


def test_distributions_differentes_donnent_divergent():
    a = {"m": np.zeros((4000, 1), dtype=np.uint8)}
    b = {"m": np.random.default_rng(1).integers(0, 2, size=(4000, 1), dtype=np.uint8)}
    assert compare_samples(a, b).verdict is Verdict.DIVERGENT


def test_cles_de_mesure_differentes_donnent_divergent():
    a = {"m": np.zeros((10, 1), dtype=np.uint8)}
    b = {"autre": np.zeros((10, 1), dtype=np.uint8)}
    assert compare_samples(a, b).verdict is Verdict.DIVERGENT


def test_les_verdicts_sont_ordonnes():
    assert (
        Verdict.BIT_EXACT
        < Verdict.NUMERICALLY_EQUIVALENT
        < Verdict.STATISTICALLY_COMPATIBLE
        < Verdict.DIVERGENT
    )


def test_chi2_sur_deux_echantillons_identiques_donne_p_eleve():
    c = {0: 500, 1: 500}
    assert chi2_homogeneity_pvalue(c, dict(c)) > 0.9


def test_chi2_sur_deux_echantillons_opposes_donne_p_faible():
    assert chi2_homogeneity_pvalue({0: 1000, 1: 0}, {0: 0, 1: 1000}) < 1e-10
