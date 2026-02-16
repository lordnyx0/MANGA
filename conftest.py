"""
Root-level conftest.py — Fixes torch/__spec__ for pytest.

The CUDA wheels for torch, torchvision, torchaudio install with
__spec__ = None under Python 3.10 + pytest, which causes
diffusers and other libraries to crash at importlib.util.find_spec().

This conftest monkey-patches find_spec to handle that ValueError
gracefully: if the module is already in sys.modules, we reconstruct
a valid ModuleSpec using the module's own __loader__ and __path__,
preserving the real loader to avoid metaclass conflicts in downstream
imports.
"""
import importlib
import importlib.util
import importlib.machinery
import sys

_original_find_spec = importlib.util.find_spec


def _patched_find_spec(name, package=None):
    """
    Wrapper that handles modules whose __spec__ is None.
    Instead of raising ValueError, we check sys.modules and
    reconstruct a proper ModuleSpec using the module's real loader.
    """
    try:
        return _original_find_spec(name, package)
    except ValueError as e:
        if "__spec__ is not set" in str(e) and name in sys.modules:
            # Module is loaded but __spec__ is None  (CUDA wheel bug).
            # Reconstruct a spec from the module's actual attributes,
            # crucially preserving the real loader to avoid downstream
            # metaclass conflicts in libraries like diffusers.
            mod = sys.modules[name]
            origin = getattr(mod, "__file__", None)
            loader = getattr(mod, "__loader__", None)
            submodule_search_locations = getattr(mod, "__path__", None)

            spec = importlib.machinery.ModuleSpec(
                name,
                loader=loader,
                origin=origin,
            )
            if submodule_search_locations is not None:
                spec.submodule_search_locations = list(submodule_search_locations)

            # Fix the module in-place so future calls don't need the workaround
            mod.__spec__ = spec
            return spec
        raise


importlib.util.find_spec = _patched_find_spec
