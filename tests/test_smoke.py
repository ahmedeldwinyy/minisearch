from minisearch import main


def test_cli_entrypoint_is_importable() -> None:
    assert callable(main)
