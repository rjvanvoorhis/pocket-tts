from pocket_tts.utils import utils


def test_download_http_cache_suffix_ignores_query_string(monkeypatch, tmp_path):
    class Response:
        content = b"data"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(utils, "make_cache_directory", lambda: tmp_path)
    monkeypatch.setattr(utils.requests, "get", lambda url: Response())

    cached_file = utils.download_if_necessary("https://example.com/audio.wav?download=true")

    assert cached_file.parent == tmp_path
    assert cached_file.suffix == ".wav"
    assert "?" not in cached_file.name
    assert cached_file.read_bytes() == b"data"


def test_download_http_uses_content_disposition_when_url_has_no_suffix(monkeypatch, tmp_path):
    class Response:
        content = b"data"
        headers = {"content-disposition": 'attachment; filename="abc123.safetensors"'}

        def raise_for_status(self):
            pass

    monkeypatch.setattr(utils, "make_cache_directory", lambda: tmp_path)
    monkeypatch.setattr(utils.requests, "get", lambda url: Response())

    cached_file = utils.download_if_necessary("http://localhost:8000/voices/abc123/data")

    assert cached_file.parent == tmp_path
    assert cached_file.suffix == ".safetensors"
    assert cached_file.read_bytes() == b"data"
