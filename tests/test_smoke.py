def test_package_importable():
    import qbridge

    assert qbridge.__version__ == "0.10.0"


def test_api_publique_exposee():
    import qbridge

    for nom in (
        "capture",
        "replay",
        "replay_record",
        "verify_archival",
        "Manifest",
        "RunRecord",
        "CaptureRun",
        "ReplayReport",
        "ArchivalReport",
        "Verdict",
        "Tier",
        "ExecutionMode",
    ):
        assert hasattr(qbridge, nom), f"{nom} absent de l'API publique"


def test_tout_all_est_reellement_exporte():
    import qbridge

    for nom in qbridge.__all__:
        assert hasattr(qbridge, nom), f"{nom} dans __all__ mais absent du module"
