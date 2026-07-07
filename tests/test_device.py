import torch

from excel_table_cnn.device import resolve_amp, resolve_device


def test_explicit_devices_pass_through():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(torch.device("cpu")) == torch.device("cpu")


def test_auto_prefers_cuda_else_cpu_never_mps():
    device = resolve_device("auto")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert device.type == expected
    assert resolve_device(None) == device


def test_mps_is_explicit_opt_in():
    # Constructing the device object works regardless of availability.
    assert resolve_device("mps").type == "mps"


def test_amp_auto_only_on_cuda():
    assert resolve_amp(None, torch.device("cuda")) is True
    assert resolve_amp(None, torch.device("cpu")) is False
    assert resolve_amp(None, torch.device("mps")) is False


def test_amp_never_forced_off_cuda():
    # Even an explicit True is clamped off-CUDA: GradScaler needs CUDA.
    assert resolve_amp(True, torch.device("cpu")) is False
    assert resolve_amp(True, torch.device("mps")) is False
    assert resolve_amp(False, torch.device("cuda")) is False
