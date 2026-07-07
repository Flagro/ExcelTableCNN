import hashlib

import pytest

from excel_table_cnn.data.loader import DatasetLoader


@pytest.fixture()
def loader_with_archive(tmp_path):
    loader = DatasetLoader(save_path=str(tmp_path))
    archive = tmp_path / "VEnron2.7z"
    archive.write_bytes(b"definitely an archive")
    return loader, archive


def test_verify_archive_accepts_matching_md5(loader_with_archive):
    loader, archive = loader_with_archive
    loader.known_md5["VEnron2"] = hashlib.md5(archive.read_bytes()).hexdigest()
    assert loader.verify_archive("VEnron2") is True


def test_verify_archive_rejects_mismatch(loader_with_archive):
    loader, _ = loader_with_archive
    loader.known_md5["VEnron2"] = "0" * 32
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        loader.verify_archive("VEnron2")


def test_known_checksums_cover_all_datasets():
    loader = DatasetLoader(save_path=".")
    assert set(loader.known_md5) == set(loader.datasets)
    assert all(len(v) == 32 for v in loader.known_md5.values())


def test_cli_helps_exit_cleanly():
    from excel_table_cnn.evaluation.evaluate import main as eval_main
    from excel_table_cnn.inference import main as detect_main
    from excel_table_cnn.training.train import main as train_main

    for main in (train_main, eval_main, detect_main):
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0
