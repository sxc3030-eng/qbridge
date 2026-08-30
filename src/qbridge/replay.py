"""replay() : re-executer depuis un manifeste et rendre un verdict."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from qbridge.backends import BACKENDS
from qbridge.capture import capture
from qbridge.fingerprint import environment_fingerprint, kernel_fingerprint
from qbridge.manifest import Manifest
from qbridge.record import RunRecord
from qbridge.tiers import Tier, split_options
from qbridge.verdict import (
    ComparisonResult,
    Verdict,
    compare_samples,
    compare_state_vectors,
)


def _plafonner_selon_le_backend(
    comparaison: ComparisonResult, impl: Any
) -> ComparisonResult:
    """Empeche un backend non deterministe d'annoncer un verdict trop fort.

    Un backend materiel ne peut PAS reproduire un resultat bit-pour-bit : le
    bruit physique n'est pas rejouable. S'il ressort BIT_EXACT, c'est un
    artefact, pas une preuve. On plafonne a STATISTICALLY_COMPATIBLE.
    """
    if impl.is_bit_exact_replayable():
        return comparaison
    if comparaison.verdict >= Verdict.STATISTICALLY_COMPATIBLE:
        return comparaison
    return ComparisonResult(
        Verdict.STATISTICALLY_COMPATIBLE,
        f"{comparaison.detail} (plafonne : le backend {impl.name} ne garantit "
        "pas la reproductibilite bit-pour-bit)",
        infidelity=comparaison.infidelity,
        p_value=comparaison.p_value,
        max_abs_delta=comparaison.max_abs_delta,
    )


@dataclass(frozen=True)
class ReplayReport:
    verdict: Verdict
    detail: str
    comparison: ComparisonResult
    original_backend: str
    replay_backend: str
    kernel_changed: bool
    environment_drift: Dict[str, Any]


def _verifier_integrite(manifest: Manifest) -> None:
    """Refuse un manifeste incoherent. Delegue a `Manifest.verify_self`, qui
    controle le circuit, le placement des options par seau, et le hash."""
    manifest.verify_self()


def _derive_environnement(manifest: Manifest) -> Dict[str, Any]:
    actuel = environment_fingerprint()
    return {
        cle: {"capture": manifest.environment.get(cle), "replay": actuel.get(cle)}
        for cle in sorted(set(manifest.environment) | set(actuel))
        if manifest.environment.get(cle) != actuel.get(cle)
    }


def replay(
    manifest: Manifest,
    *,
    backend: Optional[str] = None,
    override_performance: Optional[Dict[str, Any]] = None,
) -> ReplayReport:
    """Rejoue l'execution decrite par `manifest`.

    AVERTISSEMENT — cette fonction prouve moins qu'il n'y parait.

    Elle re-execute la capture d'origine pour obtenir sa reference. Dans le cas
    par defaut c'est donc la MEME fonction, avec les MEMES arguments, dans le
    MEME processus, a quelques microsecondes d'intervalle : elle mesure le
    determinisme de qsim, pas la conformite au resultat d'origine. Tout bug
    present des deux cotes reste invisible.

    Pour une comparaison qui prouve quelque chose sur la duree, archiver un
    `RunRecord` et utiliser `replay_record()`, dont la reference vient du
    disque.

    `override_performance` ne peut contenir que des options qui sont de niveau
    PERFORMANCE POUR LE MODE DU MANIFESTE. C'est ce qui permet de rejouer sur
    une machine ayant un autre nombre de coeurs — sauf en mode midcircuit, ou
    `cpu_threads` fait partie de l'algorithme de mesure.
    """
    _verifier_integrite(manifest)

    nom = backend or manifest.backend_name
    if nom not in BACKENDS:
        raise KeyError(f"Backend inconnu : {nom!r}. Disponibles : {sorted(BACKENDS)}")

    mode = manifest.execution_mode()
    options = dict(manifest.all_options())

    if override_performance:
        parts = split_options(override_performance, mode)
        interdits = {**parts[Tier.SEMANTIC], **parts[Tier.NUMERIC]}
        if interdits:
            raise ValueError(
                f"En mode {mode.value}, ces options ne sont pas de niveau "
                f"PERFORMANCE et ne peuvent pas etre surchargees : {sorted(interdits)}"
            )
        options.update(override_performance)

    if nom == "cirq-reference":
        options = {}  # l'oracle n'accepte aucune option d'execution

    impl = BACKENDS[nom]()
    circuit = manifest.circuit()
    bruit = manifest.noise()

    if manifest.repetitions is None:
        rejoue = impl.simulate(
            circuit, seed=manifest.seed, options=options, noise=bruit
        )
    else:
        rejoue = impl.sample(
            circuit,
            repetitions=manifest.repetitions,
            seed=manifest.seed,
            options=options,
            noise=bruit,
        )

    origine = capture(
        circuit,
        backend=manifest.backend_name,
        seed=manifest.seed,
        repetitions=manifest.repetitions,
        options=manifest.all_options(),
        noise=bruit,
    )

    if manifest.repetitions is None:
        comparaison = compare_state_vectors(origine.state_vector, rejoue)
    else:
        comparaison = compare_samples(origine.samples, rejoue)

    comparaison = _plafonner_selon_le_backend(comparaison, impl)
    return ReplayReport(
        verdict=comparaison.verdict,
        detail=comparaison.detail,
        comparison=comparaison,
        original_backend=manifest.backend_name,
        replay_backend=nom,
        kernel_changed=kernel_fingerprint() != manifest.kernel,
        environment_drift=_derive_environnement(manifest),
    )


@dataclass(frozen=True)
class ArchivalReport:
    """Resultat d'une verification archivistique — zero ressource quantique."""

    manifest_intact: bool
    results_intact: bool
    measurement_keys: list
    total_shots: int
    detail: str


def verify_archival(record: "RunRecord") -> ArchivalReport:
    """Verifie un enregistrement SANS executer quoi que ce soit.

    C'est la garantie qui sera reellement exercee dans cinq ans : prouver que
    les bitstrings archives sont bien ceux qui ont ete scelles, et pouvoir en
    recalculer tous les agregats publies — sans simulateur, sans machine
    quantique, sans meme qsim installe.

    Ne peut pas echouer faute de materiel. C'est le point.
    """
    # Les deux controles sont faits SEPAREMENT : cette fonction sert a
    # l'attribution medico-legale. Dire "le manifeste a ete falsifie" alors que
    # c'est l'archive des tirages qui l'est enverrait sur une fausse piste.
    manifeste_ok, resultats_ok = True, True
    motifs = []
    try:
        record.manifest.verify_self()
    except ValueError as e:
        manifeste_ok = False
        motifs.append(f"manifeste : {e}")

    if record.samples is not None:
        from qbridge.capture import hash_samples

        if hash_samples(record.samples) != record.result_hash:
            resultats_ok = False
            motifs.append(
                "resultats : les bitstrings archives ne correspondent pas au "
                f"hash scelle ({record.result_hash[:16]}...)"
            )

    if motifs:
        return ArchivalReport(
            manifest_intact=manifeste_ok,
            results_intact=resultats_ok,
            measurement_keys=sorted(record.samples) if record.samples else [],
            total_shots=0,
            detail=" | ".join(motifs),
        )

    cles = sorted(record.samples) if record.samples else []
    tirages = (
        sum(int(record.samples[k].shape[0]) for k in cles) if record.samples else 0
    )
    return ArchivalReport(
        manifest_intact=True,
        results_intact=True,
        measurement_keys=cles,
        total_shots=tirages,
        detail=(
            f"{len(cles)} cle(s) de mesure, {tirages} tirages archives, "
            "hashes conformes"
        ),
    )


def replay_record(
    record: "RunRecord",
    *,
    backend: Optional[str] = None,
    override_performance: Optional[Dict[str, Any]] = None,
) -> ReplayReport:
    """Rejoue et compare au RESULTAT ARCHIVE, pas a une re-execution.

    Difference essentielle avec `replay(manifest)` : celui-la re-execute la
    capture d'origine pour obtenir sa reference, ce qui ne peut pas detecter un
    bug present dans les DEUX executions. Ici la reference vient du disque,
    scellee avant. C'est la seule des deux comparaisons qui prouve quelque
    chose sur la duree.
    """
    record.verify_integrity()
    manifest = record.manifest

    nom = backend or manifest.backend_name
    if nom not in BACKENDS:
        raise KeyError(f"Backend inconnu : {nom!r}. Disponibles : {sorted(BACKENDS)}")

    mode = manifest.execution_mode()
    options = dict(manifest.all_options())

    if override_performance:
        parts = split_options(override_performance, mode)
        interdits = {**parts[Tier.SEMANTIC], **parts[Tier.NUMERIC]}
        if interdits:
            raise ValueError(
                f"En mode {mode.value}, ces options ne sont pas de niveau "
                f"PERFORMANCE et ne peuvent pas etre surchargees : {sorted(interdits)}"
            )
        options.update(override_performance)

    if nom == "cirq-reference":
        options = {}

    impl = BACKENDS[nom]()
    circuit = manifest.circuit()
    bruit = manifest.noise()

    if manifest.repetitions is None:
        rejoue = impl.simulate(circuit, seed=manifest.seed, options=options, noise=bruit)
        # Le vecteur d'origine n'est pas archive (2^n * 8 octets). On ne dispose
        # que de son hash : la comparaison est donc binaire, sans mesure de
        # l'ecart. C'est une limite assumee, pas un oubli.
        from qbridge.digest import sha256_of_array

        identique = sha256_of_array(rejoue) == record.state_vector_hash
        comparaison = ComparisonResult(
            Verdict.BIT_EXACT if identique else Verdict.DIVERGENT,
            (
                "hash du vecteur d'etat conforme a l'archive"
                if identique
                else "hash du vecteur d'etat different de l'archive ; l'ecart fin "
                "est incalculable, le vecteur d'origine n'est pas archive"
            ),
        )
    else:
        rejoue = impl.sample(
            circuit,
            repetitions=manifest.repetitions,
            seed=manifest.seed,
            options=options,
            noise=bruit,
        )
        comparaison = compare_samples(record.samples, rejoue)

    comparaison = _plafonner_selon_le_backend(comparaison, impl)
    return ReplayReport(
        verdict=comparaison.verdict,
        detail=comparaison.detail,
        comparison=comparaison,
        original_backend=manifest.backend_name,
        replay_backend=nom,
        kernel_changed=kernel_fingerprint() != manifest.kernel,
        environment_drift=_derive_environnement(manifest),
    )
