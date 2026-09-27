"""Purpose: Run reviewed native T1 images through the fixed HippUnfold container.

Author: Qirui Zhang
Created: 2026-09-27
"""

from pathlib import Path
import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from utils.pipeline_inputs import read_coordinates, reviewed_subjects
from utils.runtime_config import HIPPUNFOLD_IMAGE, HIPPUNFOLD_PLATFORM, HIPPUNFOLD_SIF

# %% Read the single coordinate input and completed localization review
root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--subjects", nargs="+", help="Anonymous IDs; default: all participants retained after review.")
parser.add_argument("--dry-run", action="store_true", help="Build the workflow without running segmentation jobs.")
parser.add_argument("--external-qc", action="store_true", help="Confirm external QC of all current inputs and use results/localization_qc/QC_fail.csv without Step2 records.")
args = parser.parse_args()
contacts = read_coordinates(root)
subjects = reviewed_subjects(root, contacts, args.subjects, external_qc=args.external_qc)
output_dir = root / ("tmp/hippunfold_dry_run" if args.dry_run else "results/hippunfold")
cache_dir = root / "tmp/hippunfold_cache"
output_dir.mkdir(parents=True, exist_ok=True)
cache_dir.mkdir(parents=True, exist_ok=True)
options = ["--force-output", "--modality", "T1w", "--output-spaces", "T1w", "corobl",  # Accept this dedicated output directory containing our provenance file.
           "--output-density", "512", "2k", "--cores", "12",
           "--path-T1w", "/input/sub-{subject}_T1w.nii.gz"]

# %% Use Apptainer on Linux and Docker Desktop on Windows or macOS
if sys.platform not in {"linux", "win32", "darwin"}:
    raise SystemExit("Supported host systems are Linux, Windows, and macOS.")
runtime_name = "apptainer" if sys.platform == "linux" else "docker"
runtime = shutil.which(runtime_name)
if runtime is None:
    raise SystemExit(f"{runtime_name} is unavailable. Complete Step0_SetupRuntime.py first.")
sif_sha256 = None
if runtime_name == "apptainer":
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        raise SystemExit("The fixed HippUnfold Apptainer image requires an x86_64 Linux host.")
    sif_path = root / HIPPUNFOLD_SIF
    if not sif_path.is_file():
        raise SystemExit("The HippUnfold SIF image is missing. Complete Step0_SetupRuntime.py first.")
    definition = subprocess.check_output([runtime, "inspect", "--deffile", str(sif_path)], text=True)
    sources = [line.split(":", 1)[1].strip() for line in definition.splitlines()
               if line.strip().lower().startswith("from:")]
    if sources != [HIPPUNFOLD_IMAGE]:
        raise SystemExit("The local SIF was not built from the required HippUnfold digest.")
    with sif_path.open("rb") as stream:
        sif_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    command = [runtime, "run", "--cleanenv", "--no-eval", "--pwd", "/src",
               "--bind", f"{root / 'data/native_T1'}:/input:ro",
               "--bind", f"{output_dir}:/output:rw",
               "--bind", f"{cache_dir}:/hippunfold_cache:rw",
               "--env", "HIPPUNFOLD_CACHE_DIR=/hippunfold_cache", str(sif_path),
               "hippunfold", "/input", "/output", "participant"]
else:
    command = [runtime, "run", "--rm", "--pull=never", "--platform", HIPPUNFOLD_PLATFORM, "--cpus=12",
               "--mount", f"type=bind,source={root / 'data/native_T1'},target=/input,readonly",
               "--mount", f"type=bind,source={output_dir},target=/output",
               "--mount", f"type=bind,source={cache_dir},target=/hippunfold_cache",
               "--env", "HIPPUNFOLD_CACHE_DIR=/hippunfold_cache", HIPPUNFOLD_IMAGE,
               "hippunfold", "/input", "/output", "participant"]

# %% Record anonymous, project-relative scientific provenance
source_paths = [root / "data/coordinates.csv",
                root / "results/localization_qc/QC_fail.csv",
                *[root / "data/native_T1" / f"{subject}_T1w.nii.gz" for subject in subjects]]
if not args.external_qc:
    source_paths.append(root / "results/localization_qc/manual_qc.csv")
config = {"qc_source": "external" if args.external_qc else "Step2", "container_runtime": runtime_name, "image_digest": HIPPUNFOLD_IMAGE, "subjects": subjects,
          "parallel_participants": 1, "hippunfold_options": options,
          "source_sha256": {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in source_paths}}
if runtime_name == "apptainer":
    config.update(sif_path=HIPPUNFOLD_SIF, sif_sha256=sif_sha256)
(output_dir / "segmentation_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

# %% Process participants sequentially; stop on the first container failure
for subject in subjects:
    participant_command = command + options + ["--participant-label", subject.removeprefix("sub-")]
    if args.dry_run:
        participant_command.append("--dry-run")
    print(f"HippUnfold participant: {subject} | cores=12 | sequential", flush=True)
    subprocess.run(participant_command, check=True)
    print(f"Completed {subject}", flush=True)
