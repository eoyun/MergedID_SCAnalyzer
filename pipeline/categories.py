"""Single source of truth for the 18/12/18 training categories.

Axes: tier (AOD/MiniAOD) x detector (EB/EE) x [EE only: ES image on/off] x
hasADD (all/eq0/eq1). ES affects only the image model, so EE ES-on/off pairs
share one track model -> 12 track, 18 image, 18 fusion.
"""

WEIGHTS_ROOT = "/eos/user/y/yeo/4l/weights/categories"
COMPACT_ROOT = "/eos/user/y/yeo/4l/compact"
RUNS_ROOT = "/eos/user/y/yeo/4l/runs/v5"


def compact_h5(tier_dir, det, mode):
    return f"{COMPACT_ROOT}/{tier_dir}_{det}_{mode}.h5"

# (short, weight-dir tier, track_types)
TIERS = [("aod", "AOD", "GenTrk"), ("mini", "MiniAOD", "Lost,PF,GSF")]
DETECTORS = ("eb", "ee")
HASADD = ("all", "eq0", "eq1")
ES_OPTS = ("es", "noes")

# Training hyperparameters (edit here; one knob for the whole sweep).
EPOCHS = 20
NUM_WORKERS = "8"   # parallel dataloader workers (EOS reads are the bottleneck)
IMAGE = {"model": "resnet50", "batch_size": "16", "lr": "1e-4"}
TRACK = {"batch_size": "64", "lr": "1e-4", "max_points": "16",
         "embed_dim": "128", "depth": "4", "num_heads": "4",
         "mlp_ratio": "4.0", "dropout": "0.1"}


def weight_h5(tier_dir, mode):
    return f"{WEIGHTS_ROOT}/{tier_dir}_{mode}/category_weights.h5"


def track_categories():
    out = []
    for ts, tdir, tt in TIERS:
        for det in DETECTORS:
            for mode in HASADD:
                name = f"{ts}_{det}_{mode}"
                outdir = f"{RUNS_ROOT}/track_{name}"
                argv = [
                    "train_point_transformer_track_classifier.py",
                    "--detector", det,
                    "--weight-h5", weight_h5(tdir, mode),
                    "--compact", compact_h5(tdir, det, mode),
                    "--output-dir", outdir,
                    "--track-types", tt,
                    "--epochs", str(EPOCHS),
                    "--num-workers", NUM_WORKERS,
                    "--batch-size", TRACK["batch_size"],
                    "--lr", TRACK["lr"],
                    "--max-points", TRACK["max_points"],
                    "--embed-dim", TRACK["embed_dim"],
                    "--depth", TRACK["depth"],
                    "--num-heads", TRACK["num_heads"],
                    "--mlp-ratio", TRACK["mlp_ratio"],
                    "--dropout", TRACK["dropout"],
                ]
                out.append({"name": name, "tier": ts, "detector": det,
                            "hasadd": mode, "track_types": tt,
                            "outdir": outdir, "argv": argv})
    return out


def _image_variants():
    """Yield (name, det, mode, track_types, tier_dir, include_es)."""
    for ts, tdir, tt in TIERS:
        for mode in HASADD:
            yield (f"{ts}_eb_{mode}", "eb", mode, tt, tdir, True)
            for es in ES_OPTS:
                yield (f"{ts}_ee_{es}_{mode}", "ee", mode, tt, tdir, es == "es")


def image_categories():
    out = []
    for name, det, mode, tt, tdir, include_es in _image_variants():
        outdir = f"{RUNS_ROOT}/image_{name}"
        argv = [
            "train_resnet_image_classifier.py",
            "--detector", det,
            "--weight-h5", weight_h5(tdir, mode),
            "--compact", compact_h5(tdir, det, mode),
            "--output-dir", outdir,
            "--track-types", tt,
            "--model", IMAGE["model"],
            "--epochs", str(EPOCHS),
            "--num-workers", NUM_WORKERS,
            "--batch-size", IMAGE["batch_size"],
            "--lr", IMAGE["lr"],
        ]
        if det == "ee" and not include_es:
            argv.append("--no-es")
        out.append({"name": name, "detector": det, "hasadd": mode,
                    "track_types": tt, "include_es": include_es,
                    "outdir": outdir, "argv": argv})
    return out


def fusion_categories():
    out = []
    for img in image_categories():
        name = img["name"]
        # shared track name: strip the es/noes token for EE
        parts = name.split("_")
        ts, det, mode = parts[0], parts[1], parts[-1]
        track_name = f"{ts}_{det}_{mode}"
        image_run = f"{RUNS_ROOT}/image_{name}"
        track_run = f"{RUNS_ROOT}/track_{track_name}"
        outdir = f"{RUNS_ROOT}/fusion_{name}"
        argv = [
            "train_fusion_ensemble.py",
            "--image-run-dir", image_run,
            "--track-run-dir", track_run,
            "--output-dir", outdir,
            "--prediction-source", "csv",
            "--num-workers", NUM_WORKERS,
        ]
        out.append({"name": name, "image_run": image_run,
                    "track_run": track_run, "outdir": outdir, "argv": argv})
    return out
