import pytest

from ccvr.remote import huggingface_endpoint


def test_huggingface_endpoint_is_https_and_trims_slash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.example/")
    assert huggingface_endpoint() == "https://hf-mirror.example"
    monkeypatch.setenv("HF_ENDPOINT", "http://unsafe.example")
    with pytest.raises(ValueError, match="HTTPS"):
        huggingface_endpoint()
