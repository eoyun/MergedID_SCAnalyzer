"""Generate HTCondor submit files + per-job arg lists from categories.py."""
import argparse
from pathlib import Path

from pipeline import categories as cat

TRANSFER_MODULES = [
    "train_resnet_image_classifier.py",
    "track_point_transformer.py",
    "train_point_transformer_track_classifier.py",
    "train_fusion_ensemble.py",
    "build_event_level_weights.py",
    "pipeline",  # package dir: compact_manifest.py etc. (import pipeline.*)
]


def write_arglist(path, cats):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for c in cats:
            fh.write(" ".join(c["argv"]) + "\n")
    return path


SUBMIT_TEMPLATE = """\
universe                = vanilla
executable              = condor/run_job.sh
arguments               = $(arguments)
# Forward the submitter's Kerberos credential so workers can read/write EOS.
MY.SendCredential       = True
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = {transfers}
# All outputs are staged to EOS via xrdcp inside run_job.sh. Disable Condor's
# automatic output transfer so the compact .h5 staged into scratch is NOT copied
# back to the AFS submit dir (blows the AFS quota: errno 122).
transfer_output_files   = ""
output = {logdir}/{stage}.$(Cluster).$(Process).out
error  = {logdir}/{stage}.$(Cluster).$(Process).err
log    = {logdir}/{stage}.$(Cluster).$(Process).log
request_cpus   = {cpus}
request_memory = {mem}
request_disk   = {disk}
{gpu_line}+JobFlavour = "{flavour}"

queue arguments from {listfile}
"""


def write_submit(path, stage, listfile, log_dir="condor/logs", request_gpus=1,
                 cpus=8, mem="16 GB", disk="8 GB", flavour="nextweek",
                 min_gpu_capability=8.0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The LCG_109_cuda PyTorch supports CUDA capability 7.5-9.0, so exclude
    # V100/V100S (cc 7.0). require_gpus ANDs with the auto GPU matching without
    # clobbering the default requirements.
    gpu_line = ""
    if request_gpus:
        gpu_line = (f"request_gpus   = {request_gpus}\n"
                    f"require_gpus   = (Capability >= {min_gpu_capability})\n")
    text = SUBMIT_TEMPLATE.format(
        transfers=", ".join(TRANSFER_MODULES),
        stage=stage, cpus=cpus, mem=mem, disk=disk, logdir=log_dir,
        gpu_line=gpu_line, flavour=flavour, listfile=listfile)
    path.write_text(text)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="condor", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Group logs by the runs version tag (last component of RUNS_ROOT, e.g. "v3")
    # so condor/logs doesn't become an undifferentiated pile across submissions.
    tag = cat.RUNS_ROOT.rstrip("/").split("/")[-1]
    log_dir = args.out_dir / "logs" / tag
    log_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("train_track", cat.track_categories()),
        ("train_image", cat.image_categories()),
        ("fusion", cat.fusion_categories()),
    ]
    for stage, cats in specs:
        listfile = args.out_dir / f"{stage}.txt"
        write_arglist(listfile, cats)
        write_submit(args.out_dir / f"{stage}.sub", stage, listfile, log_dir=str(log_dir))
        print(f"[{stage}] {len(cats)} jobs -> {args.out_dir / (stage + '.sub')}  (logs: {log_dir})")


if __name__ == "__main__":
    main()
