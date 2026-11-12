from visionlens.sysutil import resolve_device


def test_device_choice_order(monkeypatch):
    monkeypatch.delenv("VISIONLENS_DEVICE", raising=False)
    assert resolve_device("cpu") == "cpu"
    monkeypatch.setenv("VISIONLENS_DEVICE", "cpu")
    assert resolve_device() == "cpu"
    assert resolve_device("cuda") == "cuda"  # an explicit argument beats the environment
    monkeypatch.delenv("VISIONLENS_DEVICE")
    assert resolve_device() in {"cpu", "cuda"}
