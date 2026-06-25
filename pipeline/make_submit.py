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
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = {transfers}
output = condor/logs/{stage}.$(Cluster).$(Process).out
error  = condor/logs/{stage}.$(Cluster).$(Process).err
log    = condor/logs/{stage}.$(Cluster).$(Process).log
request_cpus   = {cpus}
request_memory = {mem}
request_disk   = {disk}
{gpu_line}+JobFlavour = "{flavour}"

queue arguments from {listfile}
"""


def write_submit(path, stage, listfile, request_gpus=1, cpus=2, mem="8 GB",
                 disk="8 GB", flavour="tomorrow"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gpu_line = f"request_gpus   = {request_gpus}\n" if request_gpus else ""
    text = SUBMIT_TEMPLATE.format(
        transfers=", ".join(TRANSFER_MODULES),
        stage=stage, cpus=cpus, mem=mem, disk=disk,
        gpu_line=gpu_line, flavour=flavour, listfile=listfile)
    path.write_text(text)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="condor", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "logs").mkdir(exist_ok=True)
    specs = [
        ("train_track", cat.track_categories()),
        ("train_image", cat.image_categories()),
        ("fusion", cat.fusion_categories()),
    ]
    for stage, cats in specs:
        listfile = args.out_dir / f"{stage}.txt"
        write_arglist(listfile, cats)
        write_submit(args.out_dir / f"{stage}.sub", stage, listfile)
        print(f"[{stage}] {len(cats)} jobs -> {args.out_dir / (stage + '.sub')}")


if __name__ == "__main__":
    main()
