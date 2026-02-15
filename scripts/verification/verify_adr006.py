import sys
import os
from pathlib import Path
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.generation.engines.sd15_lineart_engine import SD15LineartEngine
from core.logging.setup import setup_logging

# Setup minimal logging to avoid file output spam
import logging
logging.basicConfig(level=logging.INFO)

def verify():
    print("Verifying ADR 006 (SD 1.5 Engine)...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Force float32 for CPU if CUDA not available to avoid half issues
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Device: {device}, Dtype: {dtype}")
    
    try:
        print("Attempting to load SD15LineartEngine...")
        # Note: This will try to load models. If models are missing, it should fail.
        engine = SD15LineartEngine(device=device, dtype=dtype)
        print("PASS: SD15LineartEngine instantiated successfully.")
        
        # Verify methods exist
        if hasattr(engine, 'generate_page') and hasattr(engine, 'compose_final'):
            print("PASS: Engine interface methods found.")
        else:
            print("FAIL: Engine missing required methods.")
            return False
            
    except Exception as e:
        print(f"FAIL: Failed to instantiate engine: {e}")
        # Check if it's a known error (e.g. model not found)
        if "Repository Not Found" in str(e) or "models" in str(e).lower():
            print("Hint: Run scripts/download_models_v3.py to download required models.")
        return False

    return True

if __name__ == "__main__":
    if verify():
        print("Verification Successful!")
        sys.exit(0)
    else:
        print("Verification Failed!")
        sys.exit(1)
