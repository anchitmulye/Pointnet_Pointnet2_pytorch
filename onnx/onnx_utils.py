"""Shared ONNX export helpers."""
import torch


def default_opset() -> int:
    """
    Return the minimum opset version supported by the installed PyTorch.
    Requesting a lower opset than this causes PyTorch to silently upgrade it,
    creating a mismatch between what the code says and what lands in the file.
    """
    if hasattr(torch.onnx, '_constants') and hasattr(torch.onnx._constants, 'ONNX_DEFAULT_OPSET'):
        return torch.onnx._constants.ONNX_DEFAULT_OPSET
    if hasattr(torch.onnx.utils, '_default_onnx_opset_version'):
        return torch.onnx.utils._default_onnx_opset_version()
    # Fallback: derive from PyTorch version
    major, minor = (int(x) for x in torch.__version__.split('.')[:2])
    if (major, minor) >= (2, 1):
        return 18
    if (major, minor) >= (2, 0):
        return 17
    return 13
