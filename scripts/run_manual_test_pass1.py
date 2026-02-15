
import sys
import os
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

# Ensure tests/integration is a package
(Path("tests") / "__init__.py").touch()
(Path("tests") / "integration" / "__init__.py").touch()

print("Importing test_pass1...")
try:
    from tests.integration.test_pass1 import test_pass1_full_pipeline
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)

# Mock tmp_path
tmp_path = Path("./tmp_test_isolation_pass1")
if tmp_path.exists():
    shutil.rmtree(tmp_path)
tmp_path.mkdir(parents=True, exist_ok=True)

print("Running test_pass1_full_pipeline manually...")
try:
    test_pass1_full_pipeline(tmp_path)
    print("MATCH: Test passed manually!")
except Exception as e:
    import traceback
    with open("manual_traceback.txt", "w") as f:
        traceback.print_exc(file=f)
    print(f"MATCH: Test failed manually: {e}")
