"""
Turn HDF5 / XML content into simple tables (columns + rows) for display.

Every function here returns a ``Table``.  Values inside rows are kept as
real Python numbers / strings so they can be sorted correctly; ``fmt()``
converts them to text for display.
"""

import math

from .hdf5 import Dataset, Group, natural_key


class Table(object):
    def __init__(self, title, columns, rows, note=""):
        self.title = title
        self.columns = list(columns)
        self.rows = rows
        self.note = note


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------

def fmt(v, precise=False):
    """Convert a value to display text."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        if v != v:
            return "nan"
        if precise:
            return repr(v)
        if v == 0:
            return "0"
        a = abs(v)
        if 1e-4 <= a < 1e7:
            return "%.6g" % v
        return "%.5e" % v
    if isinstance(v, (list, tuple)):
        if len(v) > 12 and not precise:
            return "[" + ", ".join(fmt(x) for x in v[:12]) + ", ... (%d values)]" % len(v)
        return "[" + ", ".join(fmt(x, precise) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join("%s: %s" % (k, fmt(x, precise)) for k, x in v.items()) + "}"
    return str(v)


def _flatten_dict(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = prefix + str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key + "."))
        else:
            out[key] = v
    return out


def _scalar(v):
    """Unwrap single-element lists."""
    while isinstance(v, list) and len(v) == 1:
        v = v[0]
    return v


def _read_scalar(group, name, default=None):
    node = group.get(name) if isinstance(group, Group) else None
    if isinstance(node, Dataset):
        try:
            return _scalar(node.read())
        except Exception:
            return default
    return default


def _read_list(group, name):
    node = group.get(name) if isinstance(group, Group) else None
    if isinstance(node, Dataset):
        v = node.read()
        return v
    return []


def describe_shape(node):
    if isinstance(node, Dataset):
        if node.shape is None:
            return "empty"
        if node.shape == ():
            return "single value"
        return " x ".join(str(s) for s in node.shape)
    return ""


# --------------------------------------------------------------------------
# Generic HDF5 tables
# --------------------------------------------------------------------------

def dataset_table(ds):
    """Show the values of a dataset as a table."""
    title = "Dataset %s" % ds.path
    if ds.shape is None:
        return Table(title, ["value"], [], "This dataset is empty.")
    flat = ds.read()
    shape = ds.shape
    note = "Shape: %s   Type: %s" % (describe_shape(ds), ds.dtype)

    is_comp = bool(flat) and isinstance(flat[0], dict)
    if is_comp:
        flat = [_flatten_dict(v) for v in flat]
        fields = list(flat[0].keys()) if flat else []

    if shape == ():
        if is_comp:
            return Table(title, ["field", "value"],
                         [[k, v] for k, v in flat[0].items()], note)
        return Table(title, ["value"], [[flat[0]]], note)

    if len(shape) == 1 or is_comp:
        idx_cols = ["index"] if len(shape) == 1 else \
            ["index %d" % (i + 1) for i in range(len(shape))]
        cols = idx_cols + (fields if is_comp else ["value"])
        rows = []
        for n, v in enumerate(flat):
            idx = _unravel(n, shape)
            rows.append(list(idx) + ([v.get(k) for k in fields] if is_comp else [v]))
        return Table(title, cols, rows, note)

    # 2-D and higher: the last dimension becomes the columns
    last = shape[-1]
    lead = shape[:-1]
    if len(lead) == 1:
        idx_cols = ["row"]
    else:
        idx_cols = ["index %d" % (i + 1) for i in range(len(lead))]
    cols = idx_cols + ["column %d" % j for j in range(last)]
    rows = []
    nlead = len(flat) // last if last else 0
    for r in range(nlead):
        rows.append(list(_unravel(r, lead)) + flat[r * last:(r + 1) * last])
    return Table(title, cols, rows, note +
                 "   (rows = first dimension(s), columns = last dimension; "
                 "indexes start at 0)")


def _unravel(n, shape):
    idx = []
    for s in reversed(shape):
        n, r = divmod(n, s) if s else (n, 0)
        idx.append(r)
    return tuple(reversed(idx))


def _preview(node):
    if isinstance(node, Dataset):
        try:
            if node.size <= 8:
                v = node.read()
                return _scalar(v) if node.shape == () or node.size == 1 else v
            return "(%d values - click to open)" % node.size
        except Exception as e:
            return "(cannot read: %s)" % e
    if isinstance(node, Group):
        return "(%d items)" % len(node.links())
    return getattr(node, "reason", "")


def group_table(group):
    rows = []
    for name, node in group.items():
        kind = {"group": "Folder (group)", "dataset": "Data (dataset)"}.get(
            node.kind, node.kind)
        rows.append([name, kind, describe_shape(node),
                     str(node.dtype) if isinstance(node, Dataset) else "",
                     _preview(node)])
    return Table("Contents of %s" % group.path,
                 ["name", "kind", "size", "data type", "value / preview"], rows,
                 "%d items. Click an item in the left panel to open it." % len(rows))


def attributes_table(node):
    rows = [[k, v] for k, v in sorted(node.attrs.items(), key=lambda kv: natural_key(kv[0]))]
    return Table("Attributes of %s" % node.path, ["attribute", "value"], rows,
                 "Attributes are small pieces of information attached to an item.")


def combine_subgroups(group, title=None):
    """One row per sub-group; columns = its attributes and small datasets."""
    cols = ["folder"]
    rows = []
    for name, node in group.items():
        if not isinstance(node, Group):
            continue
        rec = {"folder": name}
        for k, v in node.attrs.items():
            rec["@" + k] = _scalar(v)
        for cname, child in node.items():
            if isinstance(child, Dataset):
                try:
                    if child.size <= 16:
                        v = child.read()
                        rec[cname] = _scalar(v) if child.size <= 1 else v
                    else:
                        rec[cname] = "(%s values)" % describe_shape(child)
                except Exception as e:
                    rec[cname] = "(cannot read: %s)" % e
            elif isinstance(child, Group):
                rec[cname] = "(folder, %d items)" % len(child.links())
        for k in rec:
            if k not in cols:
                cols.append(k)
        rows.append(rec)
    table_rows = [[r.get(c) for c in cols] for r in rows]
    return Table(title or "Combined table of the folders in %s" % group.path,
                 cols, table_rows,
                 "One row per sub-folder. Columns starting with @ are attributes.")


# --------------------------------------------------------------------------
# OpenMC specific "quick views"
# --------------------------------------------------------------------------

def file_type(f):
    ft = f.root.attrs.get("filetype")
    if isinstance(ft, bytes):
        ft = ft.decode()
    if ft:
        return str(_scalar(ft))
    keys = set(f.root.keys())
    if {"eigenvalues", "time", "number"} <= keys:
        return "depletion results"
    return ""


def quick_views(f):
    """List of (title, description, function returning a Table)."""
    views = []
    ft = file_type(f).lower()
    root = f.root
    views.append(("File information",
                  "General information stored in the file (attributes).",
                  lambda: file_info_table(f)))
    if ft == "statepoint" or "tallies" in root:
        if "k_combined" in root or "k_generation" in root:
            views.append(("k-effective by batch",
                          "Multiplication factor for every batch.",
                          lambda: keff_table(f)))
        if "global_tallies" in root:
            views.append(("Global tallies", "k-collision, k-absorption, "
                          "k-tracklength and leakage with uncertainties.",
                          lambda: global_tallies_table(f)))
        tallies = root.get("tallies")
        if isinstance(tallies, Group):
            views.append(("All tallies (overview)",
                          "One row per tally: name, filters, nuclides, scores.",
                          lambda: tallies_overview(f)))
            for name, node in tallies.items():
                if isinstance(node, Group) and name.startswith("tally "):
                    tname = _read_scalar(node, "name", "") or ""
                    label = "Tally %s%s" % (name[6:], (" - " + str(tname)) if tname else "")
                    views.append((label, "Mean and standard deviation for every "
                                  "filter bin, nuclide and score.",
                                  (lambda n=node: tally_table(f, n))))
            meshes = tallies.get("meshes")
            if isinstance(meshes, Group) and meshes.links():
                views.append(("Meshes", "Meshes used by the tallies.",
                              lambda m=meshes: combine_subgroups(m, "Meshes")))
            filters = tallies.get("filters")
            if isinstance(filters, Group) and filters.links():
                views.append(("Filters", "Filters used by the tallies.",
                              lambda m=filters: combine_subgroups(m, "Filters")))
    if "materials" in root and isinstance(root["materials"], Group):
        views.append(("Material compositions",
                      "One row per nuclide of every material.",
                      lambda: materials_composition(f)))
        views.append(("Materials (overview)", "One row per material.",
                      lambda: combine_subgroups(root["materials"], "Materials")))
    for path, label in (("geometry/cells", "Cells"), ("geometry/surfaces", "Surfaces"),
                        ("geometry/universes", "Universes"),
                        ("geometry/lattices", "Lattices"), ("nuclides", "Nuclides"),
                        ("tally_derivatives", "Tally derivatives")):
        g = root.get(path)
        if isinstance(g, Group) and any(isinstance(n, Group) for _, n in g.items()):
            views.append((label, "One row per item in /%s." % path,
                          (lambda g=g, label=label: combine_subgroups(g, label))))
    if ft == "depletion results" or {"eigenvalues", "time"} <= set(root.keys()):
        views.append(("Depletion: k-effective vs time",
                      "Eigenvalue at every depletion step.",
                      lambda: depletion_table(f)))
    return views


def file_info_table(f):
    rows = [["file name", f.filename], ["file type", file_type(f) or "(not stated)"]]
    for k, v in sorted(f.root.attrs.items()):
        rows.append([k, _scalar(v)])
    for name, node in f.root.items():
        if isinstance(node, Dataset) and node.shape is not None and node.size <= 3:
            try:
                rows.append([name, _scalar(node.read())])
            except Exception:
                pass
    return Table("File information", ["property", "value"], rows)


def _mean_std(s, s2, n):
    if not n:
        return None, None, None
    mean = s / n
    if n > 1:
        var = (s2 / n - mean * mean) / (n - 1)
        std = math.sqrt(var) if var > 0 else 0.0
    else:
        std = None
    rel = (100.0 * std / abs(mean)) if (std is not None and mean) else None
    return mean, std, rel


def keff_table(f):
    root = f.root
    kg = _read_list(root, "k_generation")
    ent = _read_list(root, "entropy")
    ninact = _read_scalar(root, "n_inactive", 0) or 0
    gpb = _read_scalar(root, "generations_per_batch", 1) or 1
    rows = []
    cols = ["batch", "generation", "k (this generation)", "active?"]
    if ent:
        cols.append("Shannon entropy")
    for i, k in enumerate(kg):
        batch = i // gpb + 1
        row = [batch, i + 1, k, "active" if batch > ninact else "inactive"]
        if ent:
            row.append(ent[i] if i < len(ent) else None)
        rows.append(row)
    note = ""
    kc = _read_list(root, "k_combined")
    if len(kc) == 2:
        note = "Combined k-effective = %.5f +/- %.5f" % (kc[0], kc[1])
    return Table("k-effective by batch", cols, rows, note)


def global_tallies_table(f):
    ds = f.root["global_tallies"]
    shape = ds.shape
    data = ds.read()
    n = _read_scalar(f.root, "n_realizations", 0)
    names = ["k-collision", "k-absorption", "k-tracklength", "leakage"]
    ncol = shape[-1] if shape else 1
    i_s, i_s2 = (1, 2) if ncol >= 3 else (0, 1)
    rows = []
    for r in range(shape[0] if shape else 0):
        vals = data[r * ncol:(r + 1) * ncol]
        mean, std, rel = _mean_std(vals[i_s], vals[i_s2], n)
        rows.append([names[r] if r < len(names) else "global %d" % r, mean, std, rel])
    return Table("Global tallies", ["quantity", "mean", "std. dev.", "rel. error (%)"],
                 rows, "Number of realizations (active batches): %s" % n)


def _filter_info(f, fid):
    g = f.root.get("tallies/filters/filter %d" % fid)
    if not isinstance(g, Group):
        return "filter %d" % fid, None
    ftype = _read_scalar(g, "type", "filter") or "filter"
    n_bins = _read_scalar(g, "n_bins", None)
    bins_ds = g.get("bins")
    bins = bins_ds.read() if isinstance(bins_ds, Dataset) else []
    bshape = bins_ds.shape if isinstance(bins_ds, Dataset) else ()
    labels = None
    ftype_l = str(ftype).lower()

    if ftype_l in ("mesh", "meshborn", "meshsurface") and bins:
        mesh = f.root.get("tallies/meshes/mesh %d" % int(bins[0]))
        dims = _read_list(mesh, "dimension") if isinstance(mesh, Group) else []
        if dims and ftype_l != "meshsurface":
            labels = []
            total = 1
            for d in dims:
                total *= d
            for n in range(total):
                idx, m = [], n
                for d in dims:            # x varies fastest
                    m, r = divmod(m, d)
                    idx.append(r + 1)
                labels.append("(" + ", ".join(str(i) for i in idx) + ")")
    elif ftype_l == "legendre":
        order = _read_scalar(g, "order", None)
        if order is not None:
            labels = ["P%d" % i for i in range(int(order) + 1)]
    elif ftype_l == "distribcell" and n_bins:
        labels = ["instance %d" % i for i in range(int(n_bins))]
    if labels is None and n_bins:
        n_bins = int(n_bins)
        if len(bshape) == 2 and bshape[0] == n_bins:
            w = bshape[1]
            labels = [fmt(bins[i * w:(i + 1) * w]) for i in range(n_bins)]
        elif len(bins) == n_bins + 1 and all(isinstance(b, float) for b in bins):
            labels = ["%s - %s" % (fmt(bins[i]), fmt(bins[i + 1])) for i in range(n_bins)]
        elif len(bins) == n_bins:
            labels = list(bins)
    if labels is None:
        labels = ["bin %d" % (i + 1) for i in range(int(n_bins or 0))]
    unit = {"energy": " [eV]", "energyout": " [eV]", "time": " [s]",
            "mesh": " (x, y, z)", "meshborn": " (x, y, z)"}.get(ftype_l, "")
    return "%s%s" % (ftype, unit), labels


def tally_table(f, tg):
    """Readable tally results: one row per filter bin x nuclide x score."""
    tid = tg.name.replace("tally ", "")
    n = _read_scalar(tg, "n_realizations", None)
    if n is None:
        n = _read_scalar(f.root, "n_realizations", 0)
    fids = _read_list(tg, "filters") if (_read_scalar(tg, "n_filters", 0) or 0) > 0 else []
    if isinstance(fids, int):
        fids = [fids]
    nuclides = _read_list(tg, "nuclides") or ["total"]
    scores = _read_list(tg, "score_bins") or _read_list(tg, "scores")
    res = tg.get("results")
    if not isinstance(res, Dataset) or not res.shape or len(res.shape) != 3:
        t = group_table(tg)
        t.note = "No results stored for this tally (it may not have been run yet)."
        return t
    nfb, nsc, nv = res.shape
    if not scores:
        scores = ["score %d" % i for i in range(nsc // max(len(nuclides), 1))]
    data = res.read()
    i_s, i_s2 = (1, 2) if nv == 3 else (0, 1)

    fnames, flabels, sizes = [], [], []
    for fid in fids:
        name, labels = _filter_info(f, int(fid))
        if name in fnames:
            name = "%s #%s" % (name, fid)
        fnames.append(name)
        flabels.append(labels)
        sizes.append(len(labels))
    # if bin bookkeeping does not match, fall back to plain bin numbers
    total = 1
    for s in sizes:
        total *= s
    if total != nfb:
        fnames, flabels, sizes = ["filter bin"], [["bin %d" % (i + 1) for i in range(nfb)]], [nfb]

    cols = fnames + ["nuclide", "score", "mean", "std. dev.", "rel. error (%)"]
    rows = []
    for fb in range(nfb):
        idx = _unravel(fb, sizes) if sizes else ()
        flab = [flabels[k][i] for k, i in enumerate(idx)]
        for j in range(nsc):
            nuc_i, sc_i = divmod(j, len(scores))
            base = (fb * nsc + j) * nv
            mean, std, rel = _mean_std(data[base + i_s], data[base + i_s2], n)
            rows.append(flab + [nuclides[nuc_i] if nuc_i < len(nuclides) else "?",
                                scores[sc_i], mean, std, rel])
    name = _read_scalar(tg, "name", "") or ""
    est = _read_scalar(tg, "estimator", "") or ""
    note = "Tally %s%s   estimator: %s   realizations: %s" % (
        tid, (" '%s'" % name) if name else "", est or "?", n)
    return Table("Tally %s results" % tid, cols, rows, note)


def tallies_overview(f):
    tallies = f.root["tallies"]
    rows = []
    for name, tg in tallies.items():
        if not (isinstance(tg, Group) and name.startswith("tally ")):
            continue
        fids = _read_list(tg, "filters") if (_read_scalar(tg, "n_filters", 0) or 0) > 0 else []
        ftypes = []
        for fid in fids if isinstance(fids, list) else [fids]:
            fg = f.root.get("tallies/filters/filter %d" % int(fid))
            ftypes.append("%s (%d)" % (_read_scalar(fg, "type", "?"), int(fid)))
        res = tg.get("results")
        rows.append([
            int(name[6:]) if name[6:].isdigit() else name,
            _read_scalar(tg, "name", ""),
            _read_scalar(tg, "estimator", ""),
            ", ".join(ftypes),
            ", ".join(str(x) for x in _read_list(tg, "nuclides")),
            ", ".join(str(x) for x in (_read_list(tg, "score_bins") or _read_list(tg, "scores"))),
            _read_scalar(tg, "n_realizations", None),
            res.shape[0] * res.shape[1] if isinstance(res, Dataset) and res.shape else None,
        ])
    return Table("All tallies", ["tally id", "name", "estimator", "filters", "nuclides",
                                 "scores", "realizations", "number of results"], rows)


def materials_composition(f):
    mats = f.root["materials"]
    rows = []
    for name, mg in mats.items():
        if not isinstance(mg, Group):
            continue
        mid = name.replace("material ", "")
        mname = _read_scalar(mg, "name", "") or mg.attrs.get("name", "")
        nucs = _read_list(mg, "nuclides")
        dens = _read_list(mg, "nuclide_densities")
        for i, nuc in enumerate(nucs):
            rows.append([int(mid) if mid.isdigit() else mid, _scalar(mname), nuc,
                         dens[i] if i < len(dens) else None])
        for extra in ("sab_names", "macroscopics"):
            for x in _read_list(mg, extra):
                rows.append([int(mid) if mid.isdigit() else mid, _scalar(mname), x, None])
    return Table("Material compositions",
                 ["material id", "material name", "nuclide", "density [atom/b-cm]"], rows)


def depletion_table(f):
    root = f.root
    t = root["time"]
    e = root["eigenvalues"]
    times = t.read()
    eig = e.read()
    nsteps = t.shape[0] if t.shape else 0
    tw = t.shape[1] if t.shape and len(t.shape) > 1 else 1
    per = len(eig) // nsteps if nsteps else 0
    rows = []
    for i in range(nsteps):
        t0 = times[i * tw]
        t1 = times[i * tw + tw - 1]
        k = eig[i * per] if per else None
        unc = eig[i * per + 1] if per >= 2 else None
        rows.append([i, t0, t1, t0 / 86400.0, k, unc])
    return Table("Depletion: k-effective vs time",
                 ["step", "start time [s]", "end time [s]", "start time [days]",
                  "k-effective", "uncertainty"], rows)


# --------------------------------------------------------------------------
# XML
# --------------------------------------------------------------------------

def _localname(tag):
    return tag.split("}", 1)[-1] if isinstance(tag, str) else str(tag)


def xml_children_table(elem):
    """Rows = child elements, columns = their attributes / simple sub-elements."""
    cols = ["element"]
    recs = []
    for child in elem:
        if not isinstance(child.tag, str):
            continue
        rec = {"element": _localname(child.tag)}
        for k, v in child.attrib.items():
            rec[_localname(k)] = _number(v)
        text = (child.text or "").strip()
        if text:
            rec["text"] = _number(text)
        counts = {}
        for sub in child:
            if not isinstance(sub.tag, str):
                continue
            name = _localname(sub.tag)
            counts[name] = counts.get(name, 0) + 1
        for sub in child:
            if not isinstance(sub.tag, str):
                continue
            name = _localname(sub.tag)
            if counts[name] == 1 and len(sub) == 0:
                stext = (sub.text or "").strip()
                if sub.attrib and not stext:
                    rec[name] = " ".join("%s=%s" % (_localname(k), v)
                                         for k, v in sub.attrib.items())
                else:
                    rec[name] = _number(stext)
            else:
                rec[name] = "(%d x <%s>)" % (counts[name], name)
        recs.append(rec)
        for k in rec:
            if k not in cols:
                cols.append(k)
    rows = [[r.get(c) for c in cols] for r in recs]
    return Table("Elements inside <%s>" % _localname(elem.tag), cols, rows,
                 "One row per element inside <%s>." % _localname(elem.tag))


def xml_values_table(elem):
    """Element text that is a list of numbers (e.g. energy bins) as a column."""
    parts = (elem.text or "").split()
    rows = [[i, _number(p)] for i, p in enumerate(parts)]
    return Table("Values of <%s>" % _localname(elem.tag), ["index", "value"], rows)


def xml_attributes_table(elem):
    rows = [[_localname(k), _number(v)] for k, v in elem.attrib.items()]
    text = (elem.text or "").strip()
    if text:
        rows.append(["(text)", _number(text) if len(text) < 2000 else text[:2000] + "..."])
    return Table("Attributes of <%s>" % _localname(elem.tag), ["attribute", "value"], rows)


def _number(s):
    if not isinstance(s, str):
        return s
    t = s.strip()
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        return s
