import io

import pytest

from pdfword.storage import atomic_write, atomic_write_stream, ensure_disk_space


class ChunkLimitedStream(io.BytesIO):
    def read(self, size=-1):
        assert 0 < size <= 1024 * 1024
        return super().read(size)


def test_atomic_write_stream_copies_upload_in_bounded_chunks(tmp_path):
    payload = b"%PDF-" + (b"x" * (2 * 1024 * 1024))
    source = ChunkLimitedStream(payload)
    target = tmp_path / "large.pdf"

    atomic_write_stream(target, source)

    assert target.read_bytes() == payload
    assert source.tell() == 0


def test_atomic_write_removes_temporary_file_when_replace_fails(tmp_path, monkeypatch):
    target = tmp_path / "result.pdf"

    def fail_replace(_source, _target):
        raise OSError("replace failed")

    monkeypatch.setattr("pdfword.storage.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        atomic_write(target, b"%PDF-test")

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_disk_space_check_reports_required_and_available_space(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "pdfword.storage.shutil.disk_usage",
        lambda _path: type("Usage", (), {"free": 10 * 1024 * 1024})(),
    )

    with pytest.raises(OSError, match="مساحة القرص غير كافية"):
        ensure_disk_space(tmp_path, required_bytes=20 * 1024 * 1024, reserve_bytes=0)
