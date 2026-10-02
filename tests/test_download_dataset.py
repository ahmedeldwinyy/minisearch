import ssl
from io import BytesIO
from pathlib import Path
from unittest.mock import ANY, Mock

import pytest

from scripts import download_dataset


def test_download_dataset_saves_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "raw" / "scifact.zip"
    mock_urlopen = Mock(return_value=BytesIO(b"archive contents"))
    monkeypatch.setattr(download_dataset, "urlopen", mock_urlopen)

    result = download_dataset.download_dataset(destination)

    assert result == destination
    assert destination.read_bytes() == b"archive contents"
    assert not destination.with_suffix(".zip.part").exists()
    mock_urlopen.assert_called_once_with(
        download_dataset.DATASET_URL, timeout=60, context=ANY
    )
    ssl_context = mock_urlopen.call_args.kwargs["context"]
    assert ssl_context.verify_mode == ssl.CERT_REQUIRED


def test_download_dataset_reuses_existing_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "scifact.zip"
    destination.write_bytes(b"existing archive")
    mock_urlopen = Mock(side_effect=AssertionError("archive should be reused"))
    monkeypatch.setattr(download_dataset, "urlopen", mock_urlopen)

    result = download_dataset.download_dataset(destination)

    assert result == destination
    assert destination.read_bytes() == b"existing archive"
    mock_urlopen.assert_not_called()
