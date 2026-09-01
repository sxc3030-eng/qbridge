"""Le backend IBM, teste hors ligne.

Un backend FACTICE d'IBM expose exactement la meme interface qu'un vrai pour
`transpile` et `SamplerV2`. Tout ce fichier s'execute donc sans compte et sans
reseau : le seul pas non couvert est l'obtention du backend reel aupres du
service, qui demande un jeton.
"""

from __future__ import annotations

import numpy as np
import pytest

cirq = pytest.importorskip("cirq")
pytest.importorskip("qiskit_ibm_runtime")

from qiskit_ibm_runtime.fake_provider import FakeManilaV2  # noqa: E402

from qbridge import Verdict, capture, replay_record, verify_archival  # noqa: E402
from qbridge.backends.ibm_runtime import IbmRuntimeBackend  # noqa: E402
from qbridge.calibration import synthetic_snapshot  # noqa: E402
from qbridge.record import RunRecord  # noqa: E402
from qbridge.verdict import bitstring_counts  # noqa: E402


@pytest.fixture
def backend():
    return IbmRuntimeBackend(FakeManilaV2())


def _ghz():
    q = cirq.LineQubit.range(3)
    return cirq.Circuit(
        [cirq.H(q[0]), cirq.CNOT(q[0], q[1]), cirq.CNOT(q[1], q[2]),
         cirq.measure(*q, key="m")]
    )


# ---------- le contrat d'une machine ----------


def test_le_backend_declare_ne_pas_etre_bit_exact(backend):
    """Sur du vrai materiel ce n'est plus un contrat prudent, c'est la
    physique."""
    assert backend.is_bit_exact_replayable() is False
    assert IbmRuntimeBackend.BIT_EXACT_REPLAYABLE is False


def test_simulate_est_impossible(backend):
    with pytest.raises(NotImplementedError, match="no-cloning"):
        backend.simulate(_ghz(), seed=7, options={})


def test_les_options_qsim_sont_refusees(backend):
    """Elles decrivent un simulateur. Les accepter en silence laisserait croire
    qu'elles ont influence une execution materielle."""
    with pytest.raises(ValueError, match="n'accepte pas les options qsim"):
        backend.sample(_ghz(), repetitions=10, seed=7, options={"cpu_threads": 4})


def test_un_modele_de_bruit_externe_est_refuse(backend):
    with pytest.raises(ValueError, match="propre bruit"):
        backend.sample(
            _ghz(), repetitions=10, seed=7, options={}, noise=cirq.depolarize(0.1)
        )


# ---------- la conversion cirq -> Qiskit ----------


def test_la_conversion_preserve_le_circuit(backend):
    qc = backend._vers_qiskit(_ghz())
    assert qc.num_qubits == 3
    assert qc.num_clbits == 3
    operations = dict(qc.count_ops())
    assert operations["h"] == 1
    assert operations["cx"] == 2
    assert operations["measure"] == 3


def test_le_depaquetage_concorde_avec_les_comptages_de_qiskit(backend):
    """LE controle de la conversion des resultats. Qiskit rend des octets
    empaquetes ; qbridge attend un bit par colonne. Une erreur d'ordre de bits
    passerait inapercue sans confronter aux comptages de Qiskit lui-meme."""
    from qiskit import transpile
    from qiskit_ibm_runtime import SamplerV2

    qc = backend._vers_qiskit(_ghz())
    tqc = transpile(qc, backend=FakeManilaV2(), optimization_level=1,
                    seed_transpiler=7)
    donnees = SamplerV2(mode=FakeManilaV2()).run([tqc], shots=400).result()[0].data
    champ = donnees.m_m

    depaquete = backend._depaqueter(champ)
    assert depaquete.shape == (400, 3)
    assert depaquete.dtype == np.uint8

    poids = 1 << np.arange(2, -1, -1, dtype=np.int64)
    miens = {}
    for valeur in (depaquete.astype(np.int64) * poids).sum(axis=1):
        miens[int(valeur)] = miens.get(int(valeur), 0) + 1
    ceux_de_qiskit = {int(k, 2): v for k, v in champ.get_counts().items()}
    assert miens == ceux_de_qiskit


# ---------- le cycle qbridge complet ----------


def test_capture_accepte_une_INSTANCE_de_backend(backend):
    """Un backend exigeant des identifiants ne peut pas etre construit depuis
    une chaine : c'est pourquoi il n'est pas dans le registre."""
    run = capture(_ghz(), backend=backend, seed=7, repetitions=400)
    assert run.manifest.backend_name == "ibm-runtime"
    assert "manila" in run.manifest.backend_version


def test_une_calibration_separee_est_refusee(backend):
    """Le backend porte deja son etat d'appareil."""
    with pytest.raises(ValueError, match="porte son propre"):
        capture(
            _ghz(),
            backend=backend,
            seed=7,
            repetitions=10,
            calibration=synthetic_snapshot(tuple(cirq.LineQubit.range(3))),
        )


def test_le_cycle_archive_complet(backend, tmp_path):
    run = capture(_ghz(), backend=backend, seed=7, repetitions=400)
    record = RunRecord.from_capture(run)
    record.save(tmp_path / "ibm")

    relu = RunRecord.load(tmp_path / "ibm")
    relu.verify_integrity()
    rapport = verify_archival(relu)
    assert rapport.manifest_intact and rapport.results_intact
    assert rapport.total_shots == 400


def test_le_ghz_sort_bien_du_backend(backend):
    run = capture(_ghz(), backend=backend, seed=7, repetitions=400)
    comptes = bitstring_counts(run.samples["m"])
    domine = comptes.get(0b000, 0) + comptes.get(0b111, 0)
    assert domine / sum(comptes.values()) > 0.9


def test_le_verdict_est_plafonne_pour_un_backend_materiel(backend, tmp_path):
    """Le nom `ibm-runtime` n'est PAS dans le registre. La regle « backend de
    capture inconnu = suppose non reproductible » doit donc s'appliquer, et
    c'est exactement le comportement voulu."""
    run = capture(_ghz(), backend=backend, seed=7, repetitions=200)
    RunRecord.from_capture(run).save(tmp_path / "a")
    rapport = replay_record(RunRecord.load(tmp_path / "a"), backend=backend)
    assert rapport.verdict is not Verdict.BIT_EXACT


def test_rejouer_sans_instance_echoue_clairement(backend, tmp_path):
    """On ne peut pas reconstruire depuis le nom un backend qui exige des
    identifiants. Le message doit le dire plutot que de laisser un KeyError
    obscur."""
    run = capture(_ghz(), backend=backend, seed=7, repetitions=50)
    RunRecord.from_capture(run).save(tmp_path / "b")
    with pytest.raises(KeyError, match="passe en objet"):
        replay_record(RunRecord.load(tmp_path / "b"))


# ---------- provenance de transpilation ----------


def test_la_transpilation_est_tracee(backend):
    """Deux transpilations differentes du meme circuit logique ne sont pas la
    meme experience physique. Le placement est une donnee de provenance de
    premier ordre sur du vrai materiel."""
    capture(_ghz(), backend=backend, seed=7, repetitions=100)
    trace = backend.derniere_transpilation
    assert trace is not None
    assert trace["seed_transpiler"] == 7
    assert trace["initial_layout"] is not None
    assert trace["depth"] > 0
    assert trace["gate_counts"]["measure"] == 3


def test_la_transpilation_est_reproductible_a_graine_fixe(backend):
    """La graine ne pilote pas le materiel — une machine quantique n'en accepte
    pas — mais elle rend la transpilation reproductible, ce qui est deja
    beaucoup."""
    capture(_ghz(), backend=backend, seed=7, repetitions=50)
    a = dict(backend.derniere_transpilation)
    capture(_ghz(), backend=backend, seed=7, repetitions=50)
    b = dict(backend.derniere_transpilation)
    assert a["initial_layout"] == b["initial_layout"]
    assert a["gate_counts"] == b["gate_counts"]


# ---------- l'aide a la connexion ne reclame jamais de jeton ----------


def test_backend_reel_explique_sans_reclamer_de_jeton(monkeypatch):
    """Sans compte configure, le message doit indiquer la marche a suivre —
    jamais demander le jeton, jamais le lire."""
    from qbridge.backends import ibm_runtime

    class ServiceQuiEchoue:
        def __init__(self, *a, **k):
            raise RuntimeError("aucun compte")

    import qiskit_ibm_runtime

    monkeypatch.setattr(
        qiskit_ibm_runtime, "QiskitRuntimeService", ServiceQuiEchoue
    )
    with pytest.raises(RuntimeError) as info:
        ibm_runtime.backend_reel()

    message = str(info.value)
    assert "save_account" in message
    assert "ne lit, ne stocke et n'affiche jamais ce jeton" in message
