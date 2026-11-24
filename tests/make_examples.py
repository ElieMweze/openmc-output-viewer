"""
Developer script: (re)create the example files in ../examples.

It needs h5py and numpy (only for the developer, NOT for using the viewer).
The layout follows the OpenMC statepoint / summary file formats.
"""
import os

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "examples")
rng = np.random.default_rng(42)


def s(x):
    return np.bytes_(x)


def strings(lst):
    n = max(len(x) for x in lst)
    return np.array([x.encode() for x in lst], dtype="S%d" % n)


def statepoint(path):
    nb, ninact, nreal = 100, 20, 80
    with h5py.File(path, "w") as f:
        f.attrs["filetype"] = s("statepoint")
        f.attrs["version"] = np.array([18, 1], dtype="i4")
        f.attrs["openmc_version"] = np.array([0, 15, 0], dtype="i4")
        f.attrs["date_and_time"] = s("2026-11-24 10:00:00")
        f.attrs["path"] = s("./")
        f.attrs["seed"] = np.int64(1)
        f.attrs["energy_mode"] = s("continuous-energy")
        f["run_mode"] = s("eigenvalue")
        f["n_particles"] = np.int64(10000)
        f["n_batches"] = np.int32(nb)
        f["current_batch"] = np.int32(nb)
        f["n_inactive"] = np.int32(ninact)
        f["generations_per_batch"] = np.int32(1)
        k = 1.02 + 0.004 * rng.standard_normal(nb)
        f["k_generation"] = k
        f["entropy"] = 7.5 + 0.2 * np.exp(-np.arange(nb) / 6.0) + 0.01 * rng.standard_normal(nb)
        f["k_col_abs"] = 1.021
        f["k_col_tra"] = 1.022
        f["k_abs_tra"] = 1.020
        f["k_combined"] = np.array([1.02113, 0.00041])
        f["n_realizations"] = np.int32(nreal)
        gt = np.zeros((4, 3))
        for i, m in enumerate([1.021, 1.020, 1.022, 0.031]):
            x = m + 0.004 * rng.standard_normal(nreal)
            gt[i, 1], gt[i, 2] = x.sum(), (x * x).sum()
        f["global_tallies"] = gt
        rt = f.create_group("runtime")
        for name, v in [("total initialization", 0.8), ("simulation", 42.1),
                        ("transport", 40.9), ("inactive batches", 8.2),
                        ("active batches", 32.7), ("writing statepoints", 0.05),
                        ("total", 43.0)]:
            rt[name] = v

        t = f.create_group("tallies")
        m = t.create_group("meshes")
        m.attrs["n_meshes"] = np.int32(1)
        m.attrs["ids"] = np.array([1], dtype="i4")
        mg = m.create_group("mesh 1")
        mg["type"] = s("regular")
        mg["dimension"] = np.array([4, 4], dtype="i4")
        mg["lower_left"] = np.array([-10.0, -10.0])
        mg["upper_right"] = np.array([10.0, 10.0])
        mg["width"] = np.array([5.0, 5.0])

        fl = t.create_group("filters")
        fl.attrs["n_filters"] = np.int32(3)
        fl.attrs["ids"] = np.array([1, 2, 3], dtype="i4")
        g = fl.create_group("filter 1")
        g["type"] = s("cell"); g["n_bins"] = np.int32(3)
        g["bins"] = np.array([1, 2, 3], dtype="i4")
        g = fl.create_group("filter 2")
        g["type"] = s("energy"); g["n_bins"] = np.int32(3)
        g["bins"] = np.array([0.0, 0.625, 2.0e4, 2.0e7])
        g = fl.create_group("filter 3")
        g["type"] = s("mesh"); g["n_bins"] = np.int32(16)
        g["bins"] = np.array([1], dtype="i4")

        tallies = [(1, "fuel reaction rates", [1, 2], ["U235", "U238", "total"],
                    ["fission", "absorption", "nu-fission"]),
                   (2, "flux map", [3], ["total"], ["flux"]),
                   (3, "", [], ["total"], ["flux", "total"])]
        t.attrs["n_tallies"] = np.int32(len(tallies))
        t.attrs["ids"] = np.array([x[0] for x in tallies], dtype="i4")
        nbins_of = {1: 3, 2: 3, 3: 16}
        for tid, name, filt, nucs, scores in tallies:
            g = t.create_group("tally %d" % tid)
            g.attrs["internal"] = np.int32(0)
            if name:
                g["name"] = s(name)
            g["estimator"] = s("tracklength")
            g["n_realizations"] = np.int32(nreal)
            g["n_filters"] = np.int32(len(filt))
            if filt:
                g["filters"] = np.array(filt, dtype="i4")
            g["nuclides"] = strings(nucs)
            g["n_score_bins"] = np.int32(len(scores))
            g["score_bins"] = strings(scores)
            nf = int(np.prod([nbins_of[i] for i in filt])) if filt else 1
            ns = len(nucs) * len(scores)
            mean = rng.uniform(0.01, 2.0, size=(nf, ns))
            x = mean[None] * (1 + 0.02 * rng.standard_normal((nreal, nf, ns)))
            res = np.stack([x.sum(0), (x * x).sum(0)], axis=-1)
            g["results"] = res

        # a compound dataset like the source bank
        pt = np.dtype([("x", "f8"), ("y", "f8"), ("z", "f8")])
        bank = np.dtype([("r", pt), ("u", pt), ("E", "f8"), ("time", "f8"),
                         ("wgt", "f8"), ("delayed_group", "i4"),
                         ("surf_id", "i4"), ("particle", "i4")])
        n = 200
        b = np.zeros(n, dtype=bank)
        for c in "xyz":
            b["r"][c] = rng.uniform(-10, 10, n)
        mu = rng.uniform(-1, 1, n)
        b["u"]["z"] = mu
        b["u"]["x"] = np.sqrt(1 - mu ** 2)
        b["E"] = rng.exponential(2e6, n)
        b["wgt"] = 1.0
        f.create_dataset("source_bank", data=b)


def summary(path):
    with h5py.File(path, "w") as f:
        f.attrs["filetype"] = s("summary")
        f.attrs["version"] = np.array([6, 0], dtype="i4")
        mats = f.create_group("materials")
        mats.attrs["n_materials"] = np.int32(3)
        data = [(1, "UO2 fuel", 10.3, ["U235", "U238", "O16"], [7.2e-4, 2.2e-2, 4.6e-2]),
                (2, "Zircaloy", 6.55, ["Zr90", "Zr91", "Zr92", "Zr94"],
                 [2.2e-2, 4.8e-3, 7.3e-3, 7.4e-3]),
                (3, "Water", 0.74, ["H1", "O16", "B10"], [4.9e-2, 2.5e-2, 1.0e-5])]
        for mid, name, rho, nucs, dens in data:
            g = mats.create_group("material %d" % mid)
            g["name"] = s(name)
            g["depletable"] = np.int32(1 if mid == 1 else 0)
            g["atom_density"] = float(sum(dens))
            g["nuclides"] = strings(nucs)
            g["nuclide_densities"] = np.array(dens)
            g["temperature"] = 293.6
        geo = f.create_group("geometry")
        geo.attrs["n_cells"] = np.int32(3)
        cells = geo.create_group("cells")
        for cid, name, mat, reg in [(1, "fuel", 1, "-1"), (2, "clad", 2, "1 -2"),
                                    (3, "moderator", 3, "2 3 -4 5 -6")]:
            g = cells.create_group("cell %d" % cid)
            g["name"] = s(name)
            g["universe"] = np.int32(0)
            g["fill_type"] = s("material")
            g["material"] = np.int32(mat)
            g["region"] = s(reg)
            g["temperature"] = np.array([293.6])
        surfs = geo.create_group("surfaces")
        for sid, typ, coef, bc in [(1, "z-cylinder", [0, 0, 0.39], "transmission"),
                                   (2, "z-cylinder", [0, 0, 0.46], "transmission"),
                                   (3, "x-plane", [-0.63], "reflective"),
                                   (4, "x-plane", [0.63], "reflective"),
                                   (5, "y-plane", [-0.63], "reflective"),
                                   (6, "y-plane", [0.63], "reflective")]:
            g = surfs.create_group("surface %d" % sid)
            g["type"] = s(typ)
            g["coefficients"] = np.array(coef, dtype="f8")
            g["boundary_type"] = s(bc)
        nuc = f.create_group("nuclides")
        nuc.attrs["n_nuclides"] = np.int32(9)
        nuc["names"] = strings(["U235", "U238", "O16", "Zr90", "Zr91", "Zr92", "Zr94",
                                "H1", "B10"])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    statepoint(os.path.join(OUT, "statepoint.100.h5"))
    summary(os.path.join(OUT, "summary.h5"))
    print("written to", os.path.abspath(OUT))
