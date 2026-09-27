"""Purpose: Provision uv and HippUnfold through Apptainer on Linux or Docker elsewhere.

Author: Qirui Zhang
Created: 2026-09-27
Bootstrap with any Python 3.11 or newer. Only the standard library is required.
"""

from pathlib import Path
import argparse
import platform
import shutil
import subprocess
import sys
from utils.runtime_config import HIPPUNFOLD_IMAGE, HIPPUNFOLD_PLATFORM, HIPPUNFOLD_SIF, PYTHON_VERSION

# %% Resolve host tools without changing system applications
root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--check", action="store_true", help="Check the environment and image without installing anything.")
parser.add_argument("--python", default=PYTHON_VERSION, help="Python version or explicit uv-managed interpreter path for initial setup.")
parser.add_argument("--check-gui", action="store_true", help="Also create a hidden Tk window; requires an active desktop session.")
args = parser.parse_args()
uv = shutil.which("uv")
if sys.platform not in {"linux", "win32", "darwin"}:
    raise SystemExit("Supported host systems are Linux, Windows, and macOS.")
runtime_name = "apptainer" if sys.platform == "linux" else "docker"
runtime = shutil.which(runtime_name)
if uv is None:
    raise SystemExit("Install uv first: https://docs.astral.sh/uv/getting-started/installation/")
if runtime is None:
    raise SystemExit(f"Install {runtime_name} first; see the installation instructions in README.md.")
if sys.platform == "linux" and platform.machine().lower() not in {"x86_64", "amd64"}:
    raise SystemExit("The fixed HippUnfold Apptainer image requires an x86_64 Linux host.")
python = root / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

# %% Synchronize the reviewed lockfile into the project-local environment
if not args.check:
    subprocess.run([uv, "sync", "--locked", "--managed-python", "--python", args.python, "--link-mode", "copy"], cwd=root, check=True)
if not python.is_file():
    raise SystemExit("The project .venv is missing. Run Step 0 without --check first.")
subprocess.run([uv, "lock", "--check", "--offline", "--python", str(python)], cwd=root, check=True)
subprocess.run([uv, "pip", "check", "--python", str(python)], cwd=root, check=True)
probe = (
    "import sys, tkinter, numpy, scipy, pandas, nibabel, matplotlib, PIL, "
    "brainspace, vtk, openpyxl, xlsxwriter; "
    "from hippomaps.plotting import _apply_unfold_transforms, _rotate_in_place; "
    f"assert sys.version.split()[0] == '{PYTHON_VERSION}'; "
    "interpreter = tkinter.Tcl(); "
    "print('Python imports and Tcl initialization passed.')"
)
subprocess.run([str(python), "-B", "-c", probe], cwd=root, check=True)
if args.check_gui:
    subprocess.run([str(python), "-B", "-c",
                    "import tkinter; window=tkinter.Tk(); window.withdraw(); window.update(); window.destroy(); print('Tk window creation passed.')"],
                   cwd=root, check=True)

# %% Provision the host-specific container from the same fixed registry digest
if runtime_name == "apptainer":
    subprocess.run([runtime, "--version"], check=True)
    sif_path = root / HIPPUNFOLD_SIF
    if not sif_path.is_file():
        if args.check:
            raise SystemExit("The HippUnfold SIF image is missing. Run Step 0 without --check.")
        sif_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([runtime, "pull", str(sif_path), f"docker://{HIPPUNFOLD_IMAGE}"], check=True)
    definition = subprocess.check_output([runtime, "inspect", "--deffile", str(sif_path)], text=True)
    sources = [line.split(":", 1)[1].strip() for line in definition.splitlines()
               if line.strip().lower().startswith("from:")]
    if sources != [HIPPUNFOLD_IMAGE]:
        raise SystemExit("The local SIF was not built from the required HippUnfold digest.")
    print(f"Apptainer image ready: {HIPPUNFOLD_SIF}")
else:
    server_os = subprocess.check_output([runtime, "info", "--format", "{{.OSType}}"], text=True).strip()
    if server_os != "linux":
        raise SystemExit("HippUnfold requires Docker Desktop to use Linux containers.")
    installed = subprocess.run([runtime, "image", "inspect", HIPPUNFOLD_IMAGE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if installed.returncode:
        if args.check:
            raise SystemExit("The fixed HippUnfold image is missing. Run Step 0 without --check.")
        subprocess.run([runtime, "pull", "--platform", HIPPUNFOLD_PLATFORM, HIPPUNFOLD_IMAGE], check=True)
    subprocess.run([runtime, "image", "inspect", HIPPUNFOLD_IMAGE, "--format", "{{.Id}}"], check=True)
print("Runtime ready. Continue with Step1_LocalizationQC.py.")
