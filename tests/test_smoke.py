def test_package_importable():
    import qbridge

    assert qbridge.__version__ == "0.2.0"


def test_api_publique_exposee():
    import qbridge

    for nom in ("capture", "replay", "Manifest", "Verdict", "Tier", "ExecutionMode"):
        assert hasattr(qbridge, nom), f"{nom} absent de l'API publique"
