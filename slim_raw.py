#!/usr/bin/env python3
"""Slim a raw HDF5 by dropping the FULL-detector image arrays we never use, while
keeping everything else byte-for-byte (cropped SC/seed/ES_seed images, sparse
tracks, all scalar/meta arrays, and file attrs).

DROP = full-detector calorimeter/HCAL/preshower images:
  ECAL_energy, HBHE_energy(_EB), EE_minus/plus_energy, ES_minus/plus_plane1/2.
KEEP = everything else, notably the cropped images that may be used later:
  SC_energy/energyT/energyZ/time, EE_seed_energy(T/Z), ES_seed_plane1/2, tracks...

Copies each kept dataset with h5py.copy so dtype/compression(lzf)/chunks/attrs are
preserved exactly. Non-destructive: writes a new file; the caller decides when to
replace the original (only after verification).
"""
import argparse
import h5py
import numpy as np
from pathlib import Path

DROP = {
    "ECAL_energy",
    "HBHE_energy", "HBHE_energy_EB",
    "EE_minus_energy", "EE_plus_energy",
    "ES_minus_plane1_energy", "ES_minus_plane2_energy",
    "ES_plus_plane1_energy", "ES_plus_plane2_energy",
}


def slim_file(src, dst):
    src, dst = str(src), str(dst)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        for k, v in fin.attrs.items():
            fout.attrs[k] = v
        for name in fin.keys():
            if name in DROP:
                continue
            fin.copy(name, fout)          # preserves dtype/compression/chunks/attrs
    return dst


def _equal(a, b):
    if a.dtype == object:                 # vlen (tracks): compare row by row
        if len(a) != len(b):
            return False
        return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))
    return np.array_equal(a, b)


def verify(src, dst):
    """Return (ok, report). Checks every kept dataset is byte-identical, drops gone,
    file attrs preserved."""
    problems = []
    with h5py.File(src, "r") as fin, h5py.File(dst, "r") as fout:
        # dropped ones must be absent
        for name in DROP:
            if name in fin and name in fout:
                problems.append(f"{name}: still present in slim")
        # every kept dataset must be identical
        for name in fin.keys():
            if name in DROP:
                continue
            if name not in fout:
                problems.append(f"{name}: MISSING in slim")
                continue
            if not _equal(fin[name][:], fout[name][:]):
                problems.append(f"{name}: DATA MISMATCH")
        # file attrs
        for k in fin.attrs:
            if k not in fout.attrs:
                problems.append(f"attr {k}: missing")
    return (len(problems) == 0, problems)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    slim_file(args.src, args.dst)
    if args.verify:
        ok, probs = verify(args.src, args.dst)
        print("VERIFY", "OK" if ok else "FAIL")
        for p in probs:
            print("  ", p)


if __name__ == "__main__":
    main()
