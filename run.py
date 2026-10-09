#!/usr/bin/env python3
"""
BIDS-app entrypoint for SSL-QALAS-crossvendor.

Runs the upstream run_ssl.sh + submit_CPU.sh workflow directly inside
the container. No Slurm/sbatch is required.

Usage:
    run.py bids_dir output_dir participant
        [--participant_label ...]
        [--session_label ...]
        [-w work_dir]
        [--scanner_checkpoints DIR]
        [--no_b1_map]
        [--n_cpus N]
        [--force]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


TOOL_SRC = Path("/opt/ssl_qalas")
VERSION = os.environ.get("SSL_QALAS_VERSION", "unknown")


# -------------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "SSL-QALAS cross-vendor: "
            "3D-QALAS -> T1/T2/PD/IE maps"
        )
    )

    p.add_argument("bids_dir", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument(
        "analysis_level",
        choices=["participant"],
    )

    p.add_argument(
        "--participant_label",
        nargs="+",
        help="Participant labels without 'sub-'",
    )

    p.add_argument(
        "--session_label",
        nargs="+",
        help="Session labels without 'ses-'",
    )

    p.add_argument(
        "-w",
        "--work_dir",
        type=Path,
        default=Path("/tmp/ssl_qalas_work"),
    )

    p.add_argument(
        "--scanner_checkpoints",
        type=Path,
        help=(
            "Persistent directory for per-scanner baseline weights. "
            "Default: <output_dir>/.ssl_qalas_checkpoints"
        ),
    )

    p.add_argument(
        "--no_b1_map",
        action="store_true",
        help=(
            "No B1 maps acquired. Skip B1 matching/coregistration "
            "and use uniform B1=1."
        ),
    )

    p.add_argument(
        "--n_cpus",
        type=int,
        default=4,
    )

    p.add_argument(
        "--force",
        action="store_true",
        help="Reprocess sessions that already have maps.",
    )

    p.add_argument(
        "--version",
        action="version",
        version=f"ssl-qalas-bidsapp {VERSION}",
    )

    return p.parse_args()


# -------------------------------------------------------------------------
# Generic command runner
# -------------------------------------------------------------------------

def run_cmd(
    cmd,
    *,
    cwd=None,
    log_file=None,
    env=None,
):
    """
    Run a command directly.

    stdout/stderr can optionally be written to log_file while also being
    displayed on the container stdout/stderr.
    """

    cmd = [str(x) for x in cmd]

    print(
        "\n$ " + " ".join(cmd),
        flush=True,
    )

    if log_file is None:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            check=False,
        )
    else:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        with log_file.open("w") as fh:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                check=False,
            )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: "
            f"{' '.join(cmd)}"
        )

    return result


# -------------------------------------------------------------------------
# BIDS/session discovery
# -------------------------------------------------------------------------

def find_sessions(bids, labels, ses_labels):
    out = []

    for sub in sorted(bids.glob("sub-*")):
        if labels and sub.name[4:] not in labels:
            continue

        for ses in sorted(sub.glob("ses-*")):
            if ses_labels and ses.name[4:] not in ses_labels:
                continue

            if list((ses / "anat").glob("*QALAS.json")):
                out.append(f"{sub.name}/{ses.name}")

    return out


def find_qalas_jsons(bids, sub_ses):
    """
    Mirror run_ssl.sh:

        *inv-0*_QALAS.json

    followed by:

        *_QALAS.json
    """

    anat = bids / sub_ses / "anat"

    files = sorted(anat.glob("*inv-0*_QALAS.json"))

    if not files:
        files = sorted(anat.glob("*_QALAS.json"))

    return files


def json_to_nii(json_file):
    """
    Convert a QALAS/B1 JSON filename to its corresponding NIfTI filename.

    Prefer .nii.gz, then .nii.
    """

    candidate_gz = json_file.with_suffix("").with_suffix(".nii.gz")

    # For e.g. foo.json:
    # with_suffix("") -> foo
    # with_suffix(".nii.gz") -> foo.nii.gz

    if candidate_gz.exists():
        return candidate_gz

    candidate_nii = json_file.with_suffix(".nii")

    if candidate_nii.exists():
        return candidate_nii

    return candidate_gz


def relative_bids_path(bids, path):
    return Path(path).resolve().relative_to(bids.resolve())


# -------------------------------------------------------------------------
# Scanner information
# -------------------------------------------------------------------------

def scanner_id(bids, sub_ses):
    """
    Mirror submit_CPU.sh:

        DeviceSerialNumber + alphanumeric characters
    """

    for j in find_qalas_jsons(bids, sub_ses):
        try:
            data = json.loads(j.read_text())
        except Exception:
            continue

        value = data.get("DeviceSerialNumber")

        if value is not None:
            return (
                "DeviceSerialNumber"
                + "".join(c for c in str(value) if c.isalnum())
            )

    return None


# -------------------------------------------------------------------------
# QALAS / B1 metadata
# -------------------------------------------------------------------------

def read_json(path):
    with Path(path).open() as f:
        return json.load(f)


def shim_setting(path):
    try:
        return read_json(path).get("ShimSetting")
    except Exception:
        return None


def acquisition_datetime(path):
    try:
        value = read_json(path).get("AcquisitionDateTime")

        if not value:
            return None

        # Handle common ISO-8601 values.
        value = str(value).replace("Z", "+00:00")

        return datetime.fromisoformat(value)

    except Exception:
        return None


def run_number(path):
    """
    Extract run-N from filename.
    """

    match = re.search(
        r"run-(\d+)(?=_)",
        Path(path).name,
    )

    return int(match.group(1)) if match else None


def date_run_check(jsons):
    """
    Mirror run_ssl.sh date_run_check().

    Returns True if a later AcquisitionDateTime has a lower run number.
    """

    for current in jsons:
        current_date = acquisition_datetime(current)
        current_run = run_number(current)

        if current_date is None or current_run is None:
            continue

        for other in jsons:
            if other == current:
                continue

            other_date = acquisition_datetime(other)
            other_run = run_number(other)

            if other_date is None or other_run is None:
                continue

            if (
                current_date > other_date
                and current_run < other_run
            ):
                return True

    return False


def check_identical_shims(fmap_jsons, current):
    """
    Mirror check_identical_shims().

    Returns True if another B1 map has the same ShimSetting.
    """

    current_shim = shim_setting(current)

    for other in fmap_jsons:
        if other == current:
            continue

        other_shim = shim_setting(other)

        if other_shim == current_shim:
            print(
                "An identical ShimSetting has been encountered in "
                f"{current} and {other}"
            )
            return True

    return False


# -------------------------------------------------------------------------
# AFI
# -------------------------------------------------------------------------

def estimate_afi(
    tool,
    fmap1,
    fmap2,
    output,
    env,
):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        return

    run_cmd(
        [
            sys.executable,
            str(tool / "calculate_afi_b1.py"),
            str(fmap1),
            str(fmap2),
            str(output),
        ],
        cwd=tool,
        env=env,
    )


# -------------------------------------------------------------------------
# Coregistration
# -------------------------------------------------------------------------

def coregister_b1(
    tool,
    bids,
    sub_ses,
    f_qalas,
    f_fmap,
    fmap_contrast,
    path_precoreg,
    fmap_coreg_output,
    env,
):
    """
    Mirror run_ssl.sh run_coregistration_and_submit().
    """

    fmap_coreg_output = Path(fmap_coreg_output)

    if fmap_coreg_output.exists():
        return

    fmap_coreg_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_cmd(
        [
            sys.executable,
            str(tool / "coreg_b1.py"),
            str(bids / fmap_contrast),
            str(bids / sub_ses / "anat" / f_qalas),
            str(
                path_precoreg
                / sub_ses
                / "fmap"
                / f_fmap
            ),
            str(fmap_coreg_output),
        ],
        cwd=tool,
        env=env,
    )


# -------------------------------------------------------------------------
# FreeSurfer / skull stripping
# -------------------------------------------------------------------------

def synthstrip_qalas(
    tool,
    bids,
    sub_ses,
    f_qalas,
    freesurfer,
    env,
):
    """
    Mirror the skull-stripping section of submit_CPU.sh.
    """

    qalas_mask_input = f_qalas.replace(
        "inv-0",
        "inv-2",
    )

    source = (
        bids
        / sub_ses
        / "anat"
        / qalas_mask_input
    )

    mask_dir = tool / "synthstrip_mask"
    mask_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask = mask_dir / qalas_mask_input

    if mask.exists():
        print(
            "\nA mask was detected and used:"
        )
        print(mask)
        return

    if not source.exists():
        raise FileNotFoundError(
            f"Cannot find QALAS image required for "
            f"skull stripping: {source}"
        )

    nframes_result = subprocess.run(
        [
            str(freesurfer / "bin" / "mri_info"),
            "--nframes",
            str(source),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    nframes = int(nframes_result.stdout.strip())

    if nframes > 1:
        print(
            "4D image detected. Extracting Frame 2..."
        )

        frame = mask_dir / f_qalas.replace(
            "QALAS",
            "QALAS_frame",
        )

        run_cmd(
            [
                str(freesurfer / "bin" / "mri_convert"),
                str(source),
                "--frame",
                "2",
                str(frame),
            ],
            env=env,
        )

        try:
            run_cmd(
                [
                    str(freesurfer / "bin" / "mri_synthstrip"),
                    "-i",
                    str(frame),
                    "-m",
                    str(mask),
                    "-b",
                    "2",
                ],
                env=env,
            )
        finally:
            frame.unlink(missing_ok=True)

    else:
        run_cmd(
            [
                str(freesurfer / "bin" / "mri_synthstrip"),
                "-i",
                str(source),
                "-m",
                str(mask),
                "-b",
                "2",
            ],
            env=env,
        )


# -------------------------------------------------------------------------
# Checkpoint helpers
# -------------------------------------------------------------------------

def newest_ckpt(directory):
    directory = Path(directory)

    checkpoints = sorted(
        directory.glob("epoch*.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    return checkpoints[0] if checkpoints else None


def checkpoint_epoch(checkpoint):
    """
    Equivalent to submit_CPU.sh epochs_after().
    """

    import torch

    try:
        ck = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        ck = torch.load(
            checkpoint,
            map_location="cpu",
        )

    return int(ck.get("epoch", 0))


# -------------------------------------------------------------------------
# Actual submit_CPU processing
# -------------------------------------------------------------------------

def process_qalas_run(
    *,
    sub_ses,
    f_qalas,
    f_fmap,
    bids,
    tool,
    freesurfer,
    env,
):
    """
    Direct Python implementation of submit_CPU.sh.

    This is the part that previously ran inside sbatch.
    """

    sub_ses_run = (
        f"{sub_ses}/"
        f"{run_number(f_qalas) if run_number(f_qalas) is not None else ''}"
    )

    # The upstream code uses run-N directly.
    run_match = re.search(
        r"(run-\d+)(?=_)",
        f_qalas,
    )

    if not run_match:
        raise RuntimeError(
            f"Could not determine run number from {f_qalas}"
        )

    run_label = run_match.group(1)

    sub_ses_run = f"{sub_ses}/{run_label}"

    # -------------------------------------------------------------
    # B1 availability
    # -------------------------------------------------------------

    if f_fmap == "NONE":
        print(
            "\nNo B1+ map provided for this run "
            "(no_b1_map=1) - a uniform B1map of 1.0 "
            "will be used instead."
        )

    # -------------------------------------------------------------
    # Skull stripping
    # -------------------------------------------------------------

    synthstrip_qalas(
        tool,
        bids,
        sub_ses,
        f_qalas,
        freesurfer,
        env,
    )

    # -------------------------------------------------------------
    # Training configuration
    # -------------------------------------------------------------

    EPOCHS_FRESH = 500
    EPOCHS_WARMSTART = 100
    EPOCHS_RESUME = 100

    run_ckpt_dir = (
        tool
        / "qalas_log"
        / sub_ses_run
        / "checkpoints"
    )

    scanner_ckpt_dir = None
    scanner_ckpt = None

    # -------------------------------------------------------------
    # Scanner checkpoint
    # -------------------------------------------------------------

    qalas_json = (
        bids
        / sub_ses
        / "anat"
        / Path(f_qalas).with_suffix("").with_suffix(".json").name
    )

    dsn = None

    if qalas_json.exists():
        try:
            dsn_value = read_json(qalas_json).get(
                "DeviceSerialNumber"
            )

            if dsn_value is not None:
                dsn = "".join(
                    c for c in str(dsn_value)
                    if c.isalnum()
                )

        except Exception:
            pass

    if dsn:
        scanner_ckpt_dir = (
            tool
            / "qalas_log"
            / "scanner_checkpoints"
            / f"DeviceSerialNumber{dsn}"
        )

        scanner_ckpt = newest_ckpt(
            scanner_ckpt_dir
        )

    else:
        print(
            f"WARNING: no DeviceSerialNumber in "
            f"{qalas_json} -- scanner-specific "
            "weights are disabled for this run."
        )

    run_ckpt = newest_ckpt(run_ckpt_dir)

    # -------------------------------------------------------------
    # Data -> H5
    # -------------------------------------------------------------

    def save_h5():
        run_cmd(
            [
                sys.executable,
                str(tool / "main_data" / "ssl_qalas_save_h5.py"),
                sub_ses,
                f_qalas,
                f_fmap,
                str(bids),
                str(tool),
            ],
            cwd=tool / "main_data",
            env=env,
        )

    h5_data_dir = (
        tool
        / "main_data"
        / "h5_data"
        / sub_ses_run.replace("-", "")
    )

    # -------------------------------------------------------------
    # Scanner warm start
    # -------------------------------------------------------------

    if scanner_ckpt is not None:

        print(
            "\nDATA FROM THIS SCANNER HAS ALREADY BEEN "
            "PROCESSED, INITIALISING FROM ITS WEIGHTS"
        )

        print(scanner_ckpt)

        save_h5()

        run_cmd(
            [
                sys.executable,
                str(tool / "train_qalas.py"),
                "--data_path",
                str(h5_data_dir),
                "--check_val_every_n_epoch",
                "4",
                "--max_epochs",
                str(EPOCHS_WARMSTART),
                "--default_root_dir",
                str(tool / "qalas_log" / sub_ses_run),
                "--use_dataset_cache_file",
                "False",
                "--init_from_checkpoint",
                str(scanner_ckpt),
            ],
            cwd=tool,
            env=env,
        )

        print(
            "PROCESSING WAS DONE STARTING FROM "
            "SCANNER-SPECIFIC WEIGHTS PRODUCED ON A PREVIOUS RUN"
        )

    # -------------------------------------------------------------
    # Resume interrupted same-run processing
    # -------------------------------------------------------------

    elif run_ckpt is not None:

        print(
            "\nCHECKPOINT FOUND FOR THIS RUN, "
            "RESUMING PROCESSING"
        )

        epoch = checkpoint_epoch(run_ckpt)
        max_epochs = epoch + EPOCHS_RESUME

        print(
            f"Resuming at checkpoint epoch {epoch}; "
            f"running {EPOCHS_RESUME} more epochs "
            f"(--max_epochs {max_epochs})"
        )

        run_cmd(
            [
                sys.executable,
                str(tool / "train_qalas.py"),
                "--data_path",
                str(h5_data_dir),
                "--check_val_every_n_epoch",
                "4",
                "--max_epochs",
                str(max_epochs),
                "--default_root_dir",
                str(tool / "qalas_log" / sub_ses_run),
                "--use_dataset_cache_file",
                "False",
                "--resume_from_checkpoint",
                str(run_ckpt),
            ],
            cwd=tool,
            env=env,
        )

    # -------------------------------------------------------------
    # Fresh processing
    # -------------------------------------------------------------

    else:

        save_h5()

        run_cmd(
            [
                sys.executable,
                str(tool / "train_qalas.py"),
                "--data_path",
                str(h5_data_dir),
                "--check_val_every_n_epoch",
                "4",
                "--max_epochs",
                str(EPOCHS_FRESH),
                "--default_root_dir",
                str(tool / "qalas_log" / sub_ses_run),
                "--use_dataset_cache_file",
                "False",
            ],
            cwd=tool,
            env=env,
        )

    # -------------------------------------------------------------
    # Re-resolve checkpoint
    # -------------------------------------------------------------

    run_ckpt = newest_ckpt(run_ckpt_dir)

    if run_ckpt is None:
        raise RuntimeError(
            f"Training wrote no checkpoint in {run_ckpt_dir}"
        )

    print(
        "\nUsing checkpoint for inference:"
    )
    print(run_ckpt)

    # -------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------

    multicoil_val = (
        h5_data_dir
        / "multicoil_val"
    )

    run_cmd(
        [
            sys.executable,
            str(tool / "inference_qalas_map.py"),
            "--data_path",
            str(multicoil_val),
            "--state_dict_file",
            str(run_ckpt),
            "--output_path",
            str(h5_data_dir),
        ],
        cwd=tool,
        env=env,
    )

    # -------------------------------------------------------------
    # Move checkpoint to old/
    # -------------------------------------------------------------

    old_dir = run_ckpt_dir / "old"
    old_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_name = run_ckpt.name

    for ckpt in run_ckpt_dir.glob("epoch*.ckpt"):
        shutil.move(
            str(ckpt),
            str(old_dir / ckpt.name),
        )

    if qalas_json.exists():
        shutil.copy(
            qalas_json,
            old_dir / qalas_json.name,
        )

    last_ckpt = old_dir / checkpoint_name

    # -------------------------------------------------------------
    # Extract maps
    # -------------------------------------------------------------

    run_cmd(
        [
            sys.executable,
            str(tool / "main_data" / "h5_to_maps.py"),
            sub_ses,
            f_qalas,
            str(bids),
        ],
        cwd=tool / "main_data",
        env=env,
    )

    # -------------------------------------------------------------
    # Save scanner-specific weights
    # -------------------------------------------------------------

    if dsn and scanner_ckpt_dir is not None:

        if newest_ckpt(scanner_ckpt_dir) is None:

            scanner_ckpt_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.copy(
                last_ckpt,
                scanner_ckpt_dir / last_ckpt.name,
            )

            print(
                "SCANNER-SPECIFIC WEIGHTS WERE SAVED FOR "
                f"{dsn} IN qalas_log/scanner_checkpoints"
            )

    print(
        f"Processing {sub_ses_run} is done.",
        flush=True,
    )


# -------------------------------------------------------------------------
# Process no-B1 session
# -------------------------------------------------------------------------

def process_subject_no_b1map(
    *,
    sub_ses,
    bids,
    tool,
    freesurfer,
    env,
):
    qalas_jsons = find_qalas_jsons(
        bids,
        sub_ses,
    )

    if not qalas_jsons:
        print(
            f"No 3D-QALAS image found for {sub_ses}"
        )
        return

    for json_qalas in qalas_jsons:

        qalas_path = json_to_nii(json_qalas)

        if not qalas_path.exists():
            raise FileNotFoundError(
                f"Could not find NIfTI corresponding to "
                f"{json_qalas}"
            )

        f_qalas = qalas_path.name

        print(
            "\nno_b1_map=1: processing 3D-QALAS "
            "run without a B1 map:"
        )
        print(f_qalas)

        process_qalas_run(
            sub_ses=sub_ses,
            f_qalas=f_qalas,
            f_fmap="NONE",
            bids=bids,
            tool=tool,
            freesurfer=freesurfer,
            env=env,
        )


# -------------------------------------------------------------------------
# Normal B1 processing
# -------------------------------------------------------------------------

def process_subject(
    *,
    sub_ses,
    bids,
    tool,
    freesurfer,
    env,
    no_b1_map,
):
    if no_b1_map:
        process_subject_no_b1map(
            sub_ses=sub_ses,
            bids=bids,
            tool=tool,
            freesurfer=freesurfer,
            env=env,
        )
        return

    # -------------------------------------------------------------
    # Find B1 candidates
    # -------------------------------------------------------------

    session_dir = bids / sub_ses

    patterns = [
        "*tr1_run-*TB1AFI.json",
        "*acq-famp_run-*TB1TFL.json",
        "*part-phase_TB1TFL.json",
    ]

    fmap_jsons = []

    for pattern in patterns:
        fmap_jsons = sorted(
            (session_dir / "fmap").glob(pattern)
        )

        if fmap_jsons:
            break

    if not fmap_jsons:
        print(
            f"No B1 map found for {sub_ses}"
        )
        return

    # -------------------------------------------------------------
    # Find QALAS
    # -------------------------------------------------------------

    qalas_jsons = find_qalas_jsons(
        bids,
        sub_ses,
    )

    if not qalas_jsons:
        print(
            f"No 3D-QALAS image found for {sub_ses}"
        )
        return

    # -------------------------------------------------------------
    # Shim matching
    # -------------------------------------------------------------

    qalas_shims = [
        shim_setting(x)
        for x in qalas_jsons
    ]

    matching_shim_flag = any(
        shim is not None
        and shim in qalas_shims
        for shim in (
            shim_setting(x)
            for x in fmap_jsons
        )
    )

    # -------------------------------------------------------------
    # Process each B1 map
    # -------------------------------------------------------------

    for json_fmap in fmap_jsons:

        shim_fmap = shim_setting(
            json_fmap
        )

        fmap_path = json_to_nii(
            json_fmap
        )

        if not fmap_path.exists():
            print(
                f"WARNING: B1 NIfTI not found for "
                f"{json_fmap}"
            )
            continue

        f_fmap = fmap_path.name

        # ---------------------------------------------------------
        # AFI
        # ---------------------------------------------------------

        if "TB1AFI.nii" in f_fmap:

            fmap_tr2 = Path(
                str(fmap_path).replace(
                    "acq-tr1",
                    "acq-tr2",
                )
            )

            output_name = f_fmap.replace(
                "acq-tr1",
                "acq-est",
            )

            afi_out = (
                tool
                / "afi_b1_maps"
                / sub_ses
                / "fmap"
                / output_name
            )

            estimate_afi(
                tool,
                fmap_path,
                fmap_tr2,
                afi_out,
                env,
            )

            fmap_contrast = (
                relative_bids_path(
                    bids,
                    fmap_path,
                )
            )

            f_fmap = afi_out.name
            path_precoreg = (
                tool / "afi_b1_maps"
            )

        else:

            fmap_contrast_path = Path(
                str(fmap_path)
                .replace(
                    "acq-famp",
                    "acq-anat",
                )
                .replace(
                    "part-phase",
                    "part-mag",
                )
            )

            fmap_contrast = (
                relative_bids_path(
                    bids,
                    fmap_contrast_path,
                )
            )

            path_precoreg = bids

        # ---------------------------------------------------------
        # Match to QALAS
        # ---------------------------------------------------------

        for json_qalas in qalas_jsons:

            shim_qalas = shim_setting(
                json_qalas
            )

            qalas_path = json_to_nii(
                json_qalas
            )

            if not qalas_path.exists():
                continue

            f_qalas = qalas_path.name

            current_shim_b1 = shim_setting(
                json_fmap
            )

            # -----------------------------------------------------
            # Shim filtering
            # -----------------------------------------------------

            if (
                matching_shim_flag
                and current_shim_b1 not in qalas_shims
            ):
                continue

            shim_identical_flag = False
            run_time_difference_flag = False

            # -----------------------------------------------------
            # Multiple B1 maps
            # -----------------------------------------------------

            if (
                len(fmap_jsons) != 1
                and current_shim_b1 is not None
            ):

                shim_identical_flag = (
                    check_identical_shims(
                        fmap_jsons,
                        json_fmap,
                    )
                )

                run_time_difference_flag = (
                    date_run_check(
                        fmap_jsons
                    )
                )

                if run_time_difference_flag:
                    print(
                        f"{sub_ses} has B1 maps in a wrong "
                        "order, may require manual inspection"
                    )

            # -----------------------------------------------------
            # Multiple QALAS
            # -----------------------------------------------------

            if len(qalas_jsons) != 1:

                run_time_difference_flag = (
                    date_run_check(
                        qalas_jsons
                    )
                )

                if run_time_difference_flag:
                    print(
                        f"{sub_ses} has 3D-QALAS runs in a "
                        "wrong order, may require manual inspection"
                    )

            # -----------------------------------------------------
            # Run numbers
            # -----------------------------------------------------

            run_fmap = run_number(
                json_fmap
            )

            run_qalas = run_number(
                json_qalas
            )

            # -----------------------------------------------------
            # Match
            # -----------------------------------------------------

            matched = False

            # One B1 + one QALAS
            if (
                len(qalas_jsons) == 1
                and len(fmap_jsons) == 1
            ):
                print(
                    "\nOnly one pair of 3D-QALAS "
                    "and B1 map was found:"
                )
                print(f_qalas)
                print(f_fmap)
                matched = True

            # Unique ShimSetting match
            elif (
                shim_fmap == shim_qalas
                and shim_qalas is not None
                and not shim_identical_flag
            ):
                print(
                    "\nThe match was made based on the "
                    "unique ShimSetting of 3D-QALAS and B1 map:"
                )
                print(f_qalas)
                print(f_fmap)
                matched = True

            # Run-number match
            elif (
                run_fmap == run_qalas
                and not run_time_difference_flag
            ):
                print(
                    "\nThe match was made based on the "
                    "run number of 3D-QALAS and B1 map:"
                )
                print(f_qalas)
                print(f_fmap)
                matched = True

            # -----------------------------------------------------
            # Process matched pair
            # -----------------------------------------------------

            if matched:

                # Same naming logic as run_ssl.sh.
                coreg_name = (
                    f_fmap
                    .replace(
                        "acq-famp",
                        "acq-coreg",
                    )
                    .replace(
                        "acq-est",
                        "acq-coreg",
                    )
                    .replace(
                        "part-phase",
                        "part-coreg",
                    )
                )

                fmap_coreg_output = (
                    tool
                    / "coreg_b1_maps"
                    / sub_ses
                    / "fmap"
                    / coreg_name
                )

                coregister_b1(
                    tool=tool,
                    bids=bids,
                    sub_ses=sub_ses,
                    f_qalas=f_qalas,
                    f_fmap=f_fmap,
                    fmap_contrast=fmap_contrast,
                    path_precoreg=path_precoreg,
                    fmap_coreg_output=fmap_coreg_output,
                    env=env,
                )

                process_qalas_run(
                    sub_ses=sub_ses,
                    f_qalas=f_qalas,
                    f_fmap=fmap_coreg_output.name,
                    bids=bids,
                    tool=tool,
                    freesurfer=freesurfer,
                    env=env,
                )

                # Record successful matching.
                submitted = (
                    tool
                    / "overview"
                    / "multi_submitted.txt"
                )

                submitted.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                with submitted.open("a") as fh:
                    fh.write(
                        f"{json_fmap};{json_qalas}\n"
                    )

            else:

                no_match = (
                    tool
                    / "overview"
                    / "no_clear_match.txt"
                )

                no_match.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                with no_match.open("a") as fh:
                    fh.write(
                        f"{json_fmap};{json_qalas}\n"
                    )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def has_maps(out, sub_ses):
    return bool(
        list(
            (
                out
                / sub_ses
                / "anat"
            ).glob("*_T1map.nii*")
        )
    )


def main():
    args = parse_args()

    bids = args.bids_dir.resolve()
    out = args.output_dir.resolve()

    if not bids.exists():
        sys.exit(f"BIDS directory does not exist: {bids}")

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Runtime environment
    # -------------------------------------------------------------

    env = os.environ.copy()

    env["OMP_NUM_THREADS"] = str(
        args.n_cpus
    )
    env["MKL_NUM_THREADS"] = str(
        args.n_cpus
    )

    # Container already has the conda environment activated through
    # PATH, so Python subprocesses use the same environment.
    env["PYTHONNOUSERSITE"] = "1"

    freesurfer = Path(
        env.get(
            "FREESURFER_HOME",
            "/opt/fslite",
        )
    )

    # -------------------------------------------------------------
    # Sessions
    # -------------------------------------------------------------

    sessions = find_sessions(
        bids,
        args.participant_label,
        args.session_label,
    )

    if not sessions:
        sys.exit(
            "No sub-*/ses-* with *QALAS.json found "
            "(sessionless datasets are not supported)."
        )

    if not args.force:

        done = [
            s
            for s in sessions
            if has_maps(out, s)
        ]

        sessions = [
            s
            for s in sessions
            if s not in done
        ]

        if done:
            print(
                f"Skipping {len(done)} session(s) "
                "with existing maps (use --force)"
            )

    if not sessions:
        return

    # -------------------------------------------------------------
    # Working copy
    # -------------------------------------------------------------

    tool = (
        args.work_dir.resolve()
        / "tool"
    )

    if tool.exists():
        shutil.rmtree(tool)

    shutil.copytree(
        TOOL_SRC,
        tool,
        ignore=shutil.ignore_patterns(".git"),
    )

    # Upstream directories that contain runtime state.
    for directory in (
        "logs",
        "main_data/maps",
        "main_data/h5_data",
        "qalas_log",
        "overview",
        "synthstrip_mask",
        "afi_b1_maps",
        "coreg_b1_maps",
    ):
        shutil.rmtree(
            tool / directory,
            ignore_errors=True,
        )

    # -------------------------------------------------------------
    # Scanner checkpoints
    # -------------------------------------------------------------

    ckpt = (
        args.scanner_checkpoints
        or out / ".ssl_qalas_checkpoints"
    ).resolve()

    ckpt.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        tool
        / "qalas_log"
        / "scanner_checkpoints"
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        tool
        / "qalas_log"
        / "scanner_checkpoints"
    ).symlink_to(
        ckpt
    )

    # Runtime directories.
    for directory in (
        "logs",
        "overview",
        "qalas_log",
        "main_data/maps",
        "main_data/h5_data",
    ):
        (
            tool / directory
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    # -------------------------------------------------------------
    # Baseline vs remaining
    # -------------------------------------------------------------

    baseline = []
    remaining = []

    seen = set()

    for session in sessions:

        sid = scanner_id(
            bids,
            session,
        )

        if (
            sid
            and sid not in seen
            and not any(
                ckpt.glob(
                    f"{sid}/epoch*.ckpt"
                )
            )
        ):
            baseline.append(session)

        if sid:
            seen.add(sid)

    baseline_set = set(baseline)

    remaining = [
        s
        for s in sessions
        if s not in baseline_set
    ]

    # -------------------------------------------------------------
    # Phase 1
    # -------------------------------------------------------------

    if baseline:

        print(
            f"\n=== Phase 'baseline': "
            f"{len(baseline)} session(s) ===",
            flush=True,
        )

        for session in baseline:

            print(
                f"\n========== {session} ==========",
                flush=True,
            )

            process_subject(
                sub_ses=session,
                bids=bids,
                tool=tool,
                freesurfer=freesurfer,
                env=env,
                no_b1_map=args.no_b1_map,
            )

    # -------------------------------------------------------------
    # Phase 2
    # -------------------------------------------------------------

    if remaining:

        print(
            f"\n=== Phase 'remaining': "
            f"{len(remaining)} session(s) ===",
            flush=True,
        )

        for session in remaining:

            print(
                f"\n========== {session} ==========",
                flush=True,
            )

            process_subject(
                sub_ses=session,
                bids=bids,
                tool=tool,
                freesurfer=freesurfer,
                env=env,
                no_b1_map=args.no_b1_map,
            )

    # -------------------------------------------------------------
    # Collect maps
    # -------------------------------------------------------------

    maps = (
        tool
        / "main_data"
        / "maps"
    )

    if maps.exists():
        shutil.copytree(
            maps,
            out,
            dirs_exist_ok=True,
        )

    # -------------------------------------------------------------
    # Logs
    # -------------------------------------------------------------

    out_logs = out / "logs"
    out_logs.mkdir(
        parents=True,
        exist_ok=True,
    )

    tool_logs = tool / "logs"

    if tool_logs.exists():

        for log in tool_logs.glob("*.log"):
            shutil.copy(
                log,
                out_logs / log.name,
            )

    no_match = (
        tool
        / "overview"
        / "no_clear_match.txt"
    )

    if (
        no_match.exists()
        and no_match.stat().st_size
    ):
        shutil.copy(
            no_match,
            out_logs / "no_clear_match.txt",
        )

    # -------------------------------------------------------------
    # BIDS derivative metadata
    # -------------------------------------------------------------

    (
        out
        / "dataset_description.json"
    ).write_text(
        json.dumps(
            {
                "Name": (
                    "SSL-QALAS cross-vendor "
                    "quantitative maps"
                ),
                "BIDSVersion": "1.9.0",
                "DatasetType": "derivative",
                "GeneratedBy": [
                    {
                        "Name": "ssl-qalas-bidsapp",
                        "Version": VERSION,
                        "CodeURL": (
                            "https://github.com/DCAN-Labs/"
                            "HBCD-SSL-QALAS-crossvendor"
                        ),
                    }
                ],
            },
            indent=2,
        )
    )

    (
        out / ".bidsignore"
    ).write_text(
        "logs/\n"
        ".ssl_qalas_checkpoints/\n"
    )

    # -------------------------------------------------------------
    # Final validation
    # -------------------------------------------------------------

    unmatched = [
        session
        for session in sessions
        if not has_maps(
            out,
            session,
        )
    ]

    if unmatched:

        print(
            "\nSessions without maps:",
            *unmatched,
            sep="\n  ",
        )

        sys.exit(1)

    print(
        "\nSSL-QALAS processing completed successfully.",
        flush=True,
    )


if __name__ == "__main__":
    main()