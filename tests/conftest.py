"""Keep tests independent of any model registered on this machine."""

import pytest


@pytest.fixture(autouse=True)
def _no_registered_model(monkeypatch, tmp_path_factory):
    monkeypatch.setenv("AUTHORSHIP_REGISTRY", str(tmp_path_factory.getbasetemp() / "no-registry.json"))
