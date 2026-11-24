"""
Graphical interface (tkinter, part of the Python standard library).
"""

import csv
import math
import os
import xml.etree.ElementTree as ET

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

from . import __version__
from . import tables as T
from .hdf5 import Dataset, File, Group, HDF5Error, is_hdf5, natural_key

APP_NAME = "OpenMC Output Viewer"
MAX_SHOWN = 20000          # rows put on screen at once (export always saves all)

HELP_TEXT = """How to use %s

1. Click "Open file" and choose an OpenMC output file
   (statepoint.*.h5, summary.h5, depletion_results.h5, *.xml, ...).

2. In the left panel:
   - "Quick views" are ready-made, readable tables
     (k-effective, tally results with mean and standard deviation,
     material compositions, cells, ...).
   - "Raw file structure" shows everything stored in the file.
     Folders (groups) can be opened with the small arrow.

3. Click any item: its data appears as a table on the right.
   Buttons above the table offer other ways to look at that item.

4. Organise the table with the buttons on top:
   - Click a column title to sort (click again to reverse).
   - "Search" keeps only the rows containing your text.
     Choose a column and type  >0.5  or  <=1e-3  to filter numbers.
   - "Columns" hides or shows columns.
   - "Statistics" gives min / max / average / sum of numeric columns.
   - "Plot" draws a quick graph of a column.
   - "Export CSV" saves the table for Excel / LibreOffice Calc.
   - "Copy" puts the table on the clipboard.
   - "Reset view" undoes the sorting, search and hidden columns.

Keyboard: Ctrl+O open,  Ctrl+F search,  Ctrl+E export,  Ctrl+Shift+C copy.
""" % APP_NAME


class Tooltip(object):
    """Small yellow help text shown when the mouse rests on a widget."""

    def __init__(self, widget, text, status=None):
        self.widget, self.text, self.status = widget, text, status
        self.tip = None
        self.after_id = None
        widget.bind("<Enter>", self._enter, add="+")
        widget.bind("<Leave>", self._leave, add="+")
        widget.bind("<ButtonPress>", self._leave, add="+")

    def _enter(self, _e):
        if self.status is not None:
            self.status.set(self.text)
        self.after_id = self.widget.after(600, self._show)

    def _show(self):
        if self.tip:
            return
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry("+%d+%d" % (x, y))
        tk.Label(self.tip, text=self.text, justify="left", background="#ffffe0",
                 relief="solid", borderwidth=1, wraplength=360,
                 padx=6, pady=3).pack()

    def _leave(self, _e):
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None
        if self.tip:
            self.tip.destroy()
            self.tip = None
        if self.status is not None:
            self.status.set("")


def _sort_key(v):
    if v is None or v == "":
        return (2, 0)
    if isinstance(v, bool):
        return (0, int(v))
    if isinstance(v, (int, float)):
        if v != v:
            return (1, [(1, 0, "nan")])
        return (0, v)
    return (1, natural_key(T.fmt(v)))


def _as_number(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and v == v:
        return float(v)
    if isinstance(v, str):
        try:
            x = float(v)
            return x if x == x else None
        except ValueError:
            return None
    return None


class App(object):
    def __init__(self, root, paths=()):
        self.root = root
        root.title(APP_NAME)
        root.geometry("1280x780")
        root.minsize(800, 500)

        self.items = {}          # tree item id -> (kind, payload, file)
        self.open_files = {}     # tree item id of file -> file object
        self.table = None
        self.order = []          # row indices after search + sort
        self.sort_col = None
        self.sort_desc = False
        self.hidden = set()
        self._search_job = None
        self.status = tk.StringVar()

        self._setup_style()
        self._build_toolbar()
        self._build_body()
        self._build_statusbar()
        self._bind_keys()
        self._show_welcome()

        for p in paths:
            self.open_path(p)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _setup_style(self):
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        base = tkfont.nametofont("TkDefaultFont")
        size = base.cget("size") or 10
        self.font = base
        self.bold = base.copy()
        self.bold.configure(weight="bold")
        self.title_font = base.copy()
        self.title_font.configure(weight="bold", size=abs(size) + 3)
        rowh = int(self.font.metrics("linespace") * 1.45)
        style.configure("Treeview", rowheight=rowh)
        style.configure("Treeview.Heading", font=self.bold, padding=(4, 4))
        style.configure("Tool.TButton", padding=(6, 4))
        style.configure("Action.TButton", padding=(8, 3))
        style.configure("Status.TLabel", padding=(8, 3), background="#e8ecf1")
        style.configure("Note.TLabel", foreground="#444444")

    def _build_toolbar(self):
        bar = ttk.Frame(self.root, padding=(6, 6, 6, 2))
        bar.pack(side="top", fill="x")
        self.buttons = {}

        def add(text, cmd, tip, key=None):
            b = ttk.Button(bar, text=text, command=cmd, style="Tool.TButton",
                           width=-1)
            b.pack(side="left", padx=1)
            Tooltip(b, tip, self.status)
            if key:
                self.buttons[key] = b
            return b

        def sep():
            ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=5)

        add("Open file...", self.open_dialog,
            "Open an OpenMC output file (.h5 or .xml). You can open several files.")
        add("Close file", self.close_file,
            "Close the file that is selected in the left panel.")
        sep()
        add("Sort...", self.sort_dialog,
            "Sort the rows by one column. Tip: you can also click on a column title.",
            "sort")
        add("Columns...", self.columns_dialog,
            "Choose which columns are shown in the table.", "columns")
        add("Statistics", self.stats_dialog,
            "Count, minimum, maximum, average and sum of every numeric column "
            "(only the rows currently shown).", "stats")
        add("Plot", self.plot_dialog,
            "Draw a simple graph of one column (for example the mean of a tally "
            "or k-effective per batch).", "plot")
        add("Reset view", self.reset_view,
            "Show all rows and columns again, in the original order.", "reset")
        sep()
        add("Export CSV...", self.export_csv,
            "Save the table (as currently sorted and searched) to a CSV file that "
            "opens in LibreOffice Calc or Excel.", "export")
        add("Copy", self.copy_table,
            "Copy the table to the clipboard. Paste it into a spreadsheet.", "copy")
        sep()
        add("Help", self.show_help, "How to use this program.")

        # search line
        sbar = ttk.Frame(self.root, padding=(8, 2, 6, 4))
        sbar.pack(side="top", fill="x")
        ttk.Label(sbar, text="Search rows:").pack(side="left")
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(sbar, textvariable=self.search_var, width=30)
        self.search_entry.pack(side="left", padx=4)
        Tooltip(self.search_entry, "Type a word or number: only rows containing it "
                "are kept. With a column chosen you can also type >0.5, <1e-3, "
                ">=10, =92 or !=0 to filter numbers.", self.status)
        self.search_var.trace_add("write", lambda *a: self._schedule_search())
        ttk.Label(sbar, text="in column:").pack(side="left", padx=(8, 2))
        self.search_col = ttk.Combobox(sbar, state="readonly", width=24,
                                       values=["All columns"])
        self.search_col.set("All columns")
        self.search_col.pack(side="left")
        self.search_col.bind("<<ComboboxSelected>>", lambda e: self.apply_view())
        b = ttk.Button(sbar, text="Clear search", command=self.clear_search,
                       style="Action.TButton")
        b.pack(side="left", padx=6)
        Tooltip(b, "Remove the search and show all rows again.", self.status)
        self.count_label = ttk.Label(sbar, text="", style="Note.TLabel")
        self.count_label.pack(side="right", padx=6)

    def _build_body(self):
        pw = ttk.PanedWindow(self.root, orient="horizontal")
        pw.pack(side="top", fill="both", expand=True, padx=6, pady=2)

        # left: file tree
        left = ttk.LabelFrame(pw, text=" 1. Choose what to view ", padding=4)
        self.tree = ttk.Treeview(left, show="tree", selectmode="browse")
        ysb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        xsb = ttk.Scrollbar(left, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        xsb.grid(row=1, column=0, sticky="ew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.tree.column("#0", width=330, stretch=True)
        self.tree.tag_configure("file", font=self.bold)
        self.tree.tag_configure("section", foreground="#1a4f8b", font=self.bold)
        self.tree.tag_configure("view", foreground="#1a4f8b")
        self.tree.tag_configure("attrs", foreground="#7a5a00")
        self.tree.tag_configure("broken", foreground="#999999")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<<TreeviewOpen>>", self._on_open)
        pw.add(left, weight=1)

        # right: table
        right = ttk.LabelFrame(pw, text=" 2. Table ", padding=4)
        self.title_label = ttk.Label(right, text="", font=self.title_font)
        self.title_label.pack(side="top", anchor="w")
        self.note_label = ttk.Label(right, text="", style="Note.TLabel",
                                    wraplength=800, justify="left")
        self.note_label.pack(side="top", anchor="w", pady=(0, 2))
        right.bind("<Configure>", lambda e: self.note_label.configure(
            wraplength=max(300, e.width - 20)))
        self.actions = ttk.Frame(right)
        self.actions.pack(side="top", fill="x", pady=(0, 4))

        tframe = ttk.Frame(right)
        tframe.pack(side="top", fill="both", expand=True)
        self.grid = ttk.Treeview(tframe, show="headings", selectmode="extended")
        gy = ttk.Scrollbar(tframe, orient="vertical", command=self.grid.yview)
        gx = ttk.Scrollbar(tframe, orient="horizontal", command=self.grid.xview)
        self.grid.configure(yscrollcommand=gy.set, xscrollcommand=gx.set)
        self.grid.grid(row=0, column=0, sticky="nsew")
        gy.grid(row=0, column=1, sticky="ns")
        gx.grid(row=1, column=0, sticky="ew")
        tframe.rowconfigure(0, weight=1)
        tframe.columnconfigure(0, weight=1)
        self.grid.tag_configure("odd", background="#f2f5f9")
        self.grid.bind("<Control-c>", lambda e: self.copy_selection())
        pw.add(right, weight=4)

    def _build_statusbar(self):
        ttk.Label(self.root, textvariable=self.status, style="Status.TLabel",
                  anchor="w").pack(side="bottom", fill="x")

    def _bind_keys(self):
        r = self.root
        r.bind("<Control-o>", lambda e: self.open_dialog())
        r.bind("<Control-f>", lambda e: self.search_entry.focus_set())
        r.bind("<Control-e>", lambda e: self.export_csv())
        r.bind("<Control-C>", lambda e: self.copy_table())
        r.bind("<F1>", lambda e: self.show_help())

    def _show_welcome(self):
        self.set_table(T.Table(
            "Welcome", ["step", "what to do"],
            [[1, 'Click "Open file..." and choose an OpenMC output file (.h5 or .xml).'],
             [2, 'In the left panel, open "Quick views" for ready-made tables, or '
                 '"Raw file structure" to see everything in the file.'],
             [3, "Click an item to see its data here."],
             [4, "Click a column title to sort. Use Search, Columns, Statistics, "
                 "Plot and Export CSV to organise the data."]],
            "Version %s - uses only Python (no extra packages needed)." % __version__))

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------
    def open_dialog(self):
        paths = filedialog.askopenfilenames(
            parent=self.root, title="Open OpenMC output file",
            filetypes=[("OpenMC files", "*.h5 *.H5 *.hdf5 *.hdf *.xml *.XML"),
                       ("HDF5 files", "*.h5 *.H5 *.hdf5 *.hdf"),
                       ("XML files", "*.xml *.XML"), ("All files", "*")])
        for p in paths or ():
            self.open_path(p)

    def open_path(self, path):
        path = os.path.abspath(path)
        self.status.set("Opening %s ..." % path)
        self.root.update_idletasks()
        try:
            if is_hdf5(path):
                f = File(path)
                fid = self._add_hdf5(f)
            else:
                fid = self._add_xml(path)
        except (HDF5Error, ET.ParseError, OSError) as e:
            messagebox.showerror(APP_NAME, "Could not open\n%s\n\n%s" % (path, e),
                                 parent=self.root)
            self.status.set("")
            return
        except Exception as e:
            messagebox.showerror(APP_NAME, "Could not open\n%s\n\nUnexpected error: %s"
                                 % (path, e), parent=self.root)
            self.status.set("")
            return
        self.tree.selection_set(fid)
        self.tree.see(fid)
        self.status.set("Opened %s" % path)

    def _add_hdf5(self, f):
        ftype = T.file_type(f)
        label = f.filename + ("   [%s]" % ftype if ftype else "")
        fid = self.tree.insert("", "end", text=label, open=True, tags=("file",))
        self.items[fid] = ("file-h5", f, f)
        self.open_files[fid] = f
        views = T.quick_views(f)
        qid = self.tree.insert(fid, "end", text="Quick views (readable tables)",
                               open=True, tags=("section",))
        self.items[qid] = ("views", views, f)
        for title, desc, func in views:
            vid = self.tree.insert(qid, "end", text=title, tags=("view",))
            self.items[vid] = ("view", (title, desc, func), f)
        rid = self.tree.insert(fid, "end", text="Raw file structure",
                               tags=("section",))
        self.items[rid] = ("group", f.root, f)
        self._add_placeholder(rid)
        return fid

    def _add_xml(self, path):
        tree = ET.parse(path)
        root = tree.getroot()
        label = os.path.basename(path) + "   [XML]"
        fid = self.tree.insert("", "end", text=label, open=True, tags=("file",))
        self.items[fid] = ("file-xml", root, path)
        self.open_files[fid] = None
        eid = self.tree.insert(fid, "end", text=self._xml_label(root), open=True)
        self.items[eid] = ("xml", root, path)
        self._fill_xml(eid, root)
        return fid

    def close_file(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo(APP_NAME, "Select a file in the left panel first.",
                                parent=self.root)
            return
        item = sel[0]
        while self.tree.parent(item):
            item = self.tree.parent(item)
        f = self.open_files.pop(item, None)
        self._forget(item)
        self.tree.delete(item)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
        self._show_welcome()

    def _forget(self, item):
        for child in self.tree.get_children(item):
            self._forget(child)
        self.items.pop(item, None)

    # ------------------------------------------------------------------
    # Tree
    # ------------------------------------------------------------------
    def _add_placeholder(self, iid):
        pid = self.tree.insert(iid, "end", text="(loading...)")
        self.items[pid] = ("placeholder", None, None)

    def _on_open(self, _e=None):
        iid = self.tree.focus()
        children = self.tree.get_children(iid)
        if len(children) == 1 and self.items.get(children[0], ("",))[0] == "placeholder":
            self.tree.delete(children[0])
            self.items.pop(children[0], None)
            kind, obj, f = self.items[iid]
            try:
                if kind == "group":
                    self._fill_group(iid, obj, f)
            except Exception as e:
                eid = self.tree.insert(iid, "end", text="(error: %s)" % e, tags=("broken",))
                self.items[eid] = ("error", str(e), f)

    def _fill_group(self, iid, group, f):
        if group.attrs:
            aid = self.tree.insert(iid, "end", text="Attributes (%d)" % len(group.attrs),
                                   tags=("attrs",))
            self.items[aid] = ("attrs", group, f)
        for name, node in group.items():
            if isinstance(node, Group):
                cid = self.tree.insert(iid, "end", text=name + "  /")
                self.items[cid] = ("group", node, f)
                self._add_placeholder(cid)
            elif isinstance(node, Dataset):
                info = T.describe_shape(node)
                cid = self.tree.insert(iid, "end", text="%s   (%s)" % (name, info))
                self.items[cid] = ("dataset", node, f)
            else:
                cid = self.tree.insert(iid, "end", text="%s   (%s)" % (
                    name, getattr(node, "reason", node.kind)), tags=("broken",))
                self.items[cid] = ("broken", node, f)

    def _xml_label(self, elem):
        tag = T._localname(elem.tag)
        extra = []
        for key in ("id", "name", "type", "scores", "filename"):
            if key in elem.attrib:
                extra.append("%s=%s" % (key, elem.attrib[key]))
        if not extra and len(elem) == 0 and elem.text and elem.text.strip():
            txt = " ".join(elem.text.split())
            extra.append(txt if len(txt) < 40 else txt[:40] + "...")
        return "<%s>%s" % (tag, ("  " + "  ".join(extra)) if extra else "")

    def _fill_xml(self, iid, elem):
        for child in elem:
            if not isinstance(child.tag, str):
                continue
            cid = self.tree.insert(iid, "end", text=self._xml_label(child))
            self.items[cid] = ("xml", child, None)
            if len(child):
                self._fill_xml(cid, child)

    def _on_select(self, _e=None):
        sel = self.tree.selection()
        if not sel:
            return
        kind, obj, f = self.items.get(sel[0], (None, None, None))
        self.status.set("Loading...")
        self.root.update_idletasks()
        try:
            self._show_item(kind, obj, f, sel[0])
            self.status.set("")
        except Exception as e:
            self._clear_actions()
            self.set_table(T.Table("Cannot show this item", ["problem"], [[str(e)]]))
            self.status.set("Error: %s" % e)

    def _clear_actions(self):
        for w in self.actions.winfo_children():
            w.destroy()

    def _action(self, text, func, tip):
        b = ttk.Button(self.actions, text=text, style="Action.TButton",
                       command=lambda: self._run(func))
        b.pack(side="left", padx=(0, 4))
        Tooltip(b, tip, self.status)

    def _run(self, func):
        self.status.set("Loading...")
        self.root.update_idletasks()
        try:
            self.set_table(func())
            self.status.set("")
        except Exception as e:
            self.set_table(T.Table("Cannot show this item", ["problem"], [[str(e)]]))

    def _show_item(self, kind, obj, f, iid):
        self._clear_actions()
        if kind == "file-h5":
            views = T.quick_views(obj)
            self.set_table(T.Table(
                obj.filename, ["quick view", "what it shows"],
                [[t, d] for t, d, _ in views],
                "File: %s. Click a quick view in the left panel." % obj.path))
        elif kind == "file-xml":
            self._show_item("xml", obj, f, iid)
        elif kind == "views":
            self.set_table(T.Table("Quick views", ["quick view", "what it shows"],
                                   [[t, d] for t, d, _ in obj]))
        elif kind == "view":
            title, desc, func = obj
            t = func()
            if desc and not t.note:
                t.note = desc
            self.set_table(t)
        elif kind == "group":
            self._action("Contents", lambda: T.group_table(obj),
                         "List everything inside this folder.")
            self._action("Combine sub-folders into one table",
                         lambda: T.combine_subgroups(obj),
                         "Make one table with one row per sub-folder and their "
                         "values as columns (useful for materials, cells, tallies...).")
            if obj.attrs:
                self._action("Attributes", lambda: T.attributes_table(obj),
                             "Show the attributes of this folder.")
            self.set_table(T.group_table(obj))
        elif kind == "dataset":
            self._action("Values", lambda: T.dataset_table(obj),
                         "Show the values stored in this dataset.")
            if obj.attrs:
                self._action("Attributes", lambda: T.attributes_table(obj),
                             "Show the attributes of this dataset.")
            self.set_table(T.dataset_table(obj))
        elif kind == "attrs":
            self.set_table(T.attributes_table(obj))
        elif kind == "xml":
            elem = obj
            has_children = len(elem) > 0
            words = (elem.text or "").split()
            if has_children:
                self._action("Child elements", lambda: T.xml_children_table(elem),
                             "One row for each element inside this one.")
            self._action("Attributes", lambda: T.xml_attributes_table(elem),
                         "Show the attributes and text of this element.")
            if len(words) > 1:
                self._action("List of values", lambda: T.xml_values_table(elem),
                             "Show the text of this element as a list of values "
                             "(e.g. energy bins).")
            if has_children:
                self.set_table(T.xml_children_table(elem))
            elif len(words) > 1:
                self.set_table(T.xml_values_table(elem))
            else:
                self.set_table(T.xml_attributes_table(elem))
        elif kind == "broken":
            self.set_table(T.Table(obj.path, ["problem"], [[obj.reason]]))
        elif kind == "error":
            self.set_table(T.Table("Error", ["problem"], [[obj]]))

    # ------------------------------------------------------------------
    # Table display
    # ------------------------------------------------------------------
    def set_table(self, table):
        self.table = table
        self.sort_col = None
        self.sort_desc = False
        self.hidden = set()
        cols = ["All columns"] + table.columns
        self.search_col.configure(values=cols)
        self.search_col.set("All columns")
        if self.search_var.get():
            self.search_var.set("")          # a new table starts unfiltered
            if self._search_job:
                self.root.after_cancel(self._search_job)
                self._search_job = None
        self.title_label.configure(text=table.title)
        self.note_label.configure(text=table.note or "")
        ids = ["c%d" % i for i in range(len(table.columns))]
        self.grid.configure(columns=ids, displaycolumns=ids)
        sample = table.rows[:200]
        for i, (cid, name) in enumerate(zip(ids, table.columns)):
            self.grid.heading(cid, text=name, anchor="w",
                              command=lambda i=i: self.click_heading(i))
            w = self.bold.measure(name) + 30
            for r in sample:
                if i < len(r):
                    w = max(w, self.font.measure(T.fmt(r[i])) + 16)
            numeric = any(isinstance(r[i], (int, float)) and not isinstance(r[i], bool)
                          for r in sample if i < len(r))
            self.grid.column(cid, width=min(max(w, 70), 420), minwidth=40,
                             stretch=False, anchor="e" if numeric else "w")
        self.apply_view()

    def _schedule_search(self):
        if self._search_job:
            self.root.after_cancel(self._search_job)
        self._search_job = self.root.after(300, self.apply_view)

    def _row_filter(self):
        text = self.search_var.get().strip()
        if not text or self.table is None:
            return None
        col = self.search_col.get()
        cidx = self.table.columns.index(col) if col in self.table.columns else None
        for op in (">=", "<=", "!=", ">", "<", "="):
            if text.startswith(op) and cidx is not None:
                try:
                    ref = float(text[len(op):].strip())
                except ValueError:
                    break
                test = {">=": lambda x: x >= ref, "<=": lambda x: x <= ref,
                        "!=": lambda x: x != ref, ">": lambda x: x > ref,
                        "<": lambda x: x < ref, "=": lambda x: x == ref}[op]

                def match(row, cidx=cidx, test=test):
                    x = _as_number(row[cidx]) if cidx < len(row) else None
                    return x is not None and test(x)
                return match
        low = text.lower()
        if cidx is not None:
            return lambda row: cidx < len(row) and low in T.fmt(row[cidx]).lower()
        return lambda row: any(low in T.fmt(v).lower() for v in row)

    def apply_view(self):
        """Apply search + sort and redraw the table."""
        self._search_job = None
        t = self.table
        if t is None:
            return
        match = self._row_filter()
        order = [i for i, r in enumerate(t.rows) if match is None or match(r)]
        if self.sort_col is not None:
            c = self.sort_col
            order.sort(key=lambda i: _sort_key(t.rows[i][c] if c < len(t.rows[i]) else None),
                       reverse=self.sort_desc)
        self.order = order
        for i, name in enumerate(t.columns):
            arrow = ""
            if i == self.sort_col:
                arrow = "  ▼" if self.sort_desc else "  ▲"
            self.grid.heading("c%d" % i, text=name + arrow)
        self.grid.configure(displaycolumns=[
            "c%d" % i for i in range(len(t.columns)) if i not in self.hidden] or ["c0"])
        self.grid.delete(*self.grid.get_children())
        ins = self.grid.insert
        for n, i in enumerate(order[:MAX_SHOWN]):
            ins("", "end", values=[T.fmt(v) for v in t.rows[i]],
                tags=("odd",) if n % 2 else ())
        total, shown = len(t.rows), len(order)
        txt = "%d rows" % total if shown == total else "%d of %d rows match" % (shown, total)
        if shown > MAX_SHOWN:
            txt += "  (first %d shown - use Search or Export CSV for all)" % MAX_SHOWN
        if self.hidden:
            txt += ",  %d column(s) hidden" % len(self.hidden)
        self.count_label.configure(text=txt)

    def click_heading(self, i):
        if self.sort_col == i:
            if self.sort_desc:
                self.sort_col, self.sort_desc = None, False   # third click: original
            else:
                self.sort_desc = True
        else:
            self.sort_col, self.sort_desc = i, False
        self.apply_view()

    def clear_search(self):
        self.search_var.set("")
        self.search_col.set("All columns")
        self.apply_view()

    def reset_view(self):
        if self.table is None:
            return
        self.sort_col, self.sort_desc = None, False
        self.hidden = set()
        self.search_var.set("")
        self.search_col.set("All columns")
        self.apply_view()

    def visible_columns(self):
        return [i for i in range(len(self.table.columns)) if i not in self.hidden]

    # ------------------------------------------------------------------
    # Dialogs
    # ------------------------------------------------------------------
    def _need_table(self):
        if self.table is None or not self.table.columns:
            messagebox.showinfo(APP_NAME, "Open a file and choose a table first.",
                                parent=self.root)
            return False
        return True

    def _dialog(self, title):
        d = tk.Toplevel(self.root)
        d.title(title)
        d.transient(self.root)
        d.configure(padx=12, pady=10)
        return d

    def sort_dialog(self):
        if not self._need_table():
            return
        d = self._dialog("Sort rows")
        ttk.Label(d, text="Sort the rows by this column:").grid(row=0, column=0,
                                                                columnspan=2, sticky="w")
        col = ttk.Combobox(d, state="readonly", values=self.table.columns, width=40)
        col.set(self.table.columns[self.sort_col if self.sort_col is not None else 0])
        col.grid(row=1, column=0, columnspan=2, sticky="we", pady=4)
        direction = tk.StringVar(value="desc" if self.sort_desc else "asc")
        ttk.Radiobutton(d, text="Smallest first (A to Z)", value="asc",
                        variable=direction).grid(row=2, column=0, sticky="w")
        ttk.Radiobutton(d, text="Largest first (Z to A)", value="desc",
                        variable=direction).grid(row=3, column=0, sticky="w")

        def ok():
            self.sort_col = self.table.columns.index(col.get())
            self.sort_desc = direction.get() == "desc"
            d.destroy()
            self.apply_view()

        def original():
            self.sort_col = None
            d.destroy()
            self.apply_view()
        bf = ttk.Frame(d)
        bf.grid(row=4, column=0, columnspan=2, pady=(10, 0), sticky="e")
        ttk.Button(bf, text="Original order", command=original).pack(side="left", padx=3)
        ttk.Button(bf, text="Cancel", command=d.destroy).pack(side="left", padx=3)
        ttk.Button(bf, text="Sort", command=ok).pack(side="left", padx=3)

    def columns_dialog(self):
        if not self._need_table():
            return
        d = self._dialog("Choose columns")
        ttk.Label(d, text="Tick the columns you want to see:").pack(anchor="w")
        outer = ttk.Frame(d)
        outer.pack(fill="both", expand=True, pady=6)
        canvas = tk.Canvas(outer, height=min(400, 26 * len(self.table.columns) + 4),
                           width=340, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        vars_ = []
        for i, name in enumerate(self.table.columns):
            v = tk.BooleanVar(value=i not in self.hidden)
            ttk.Checkbutton(inner, text=name, variable=v).pack(anchor="w", pady=1)
            vars_.append(v)

        def set_all(val):
            for v in vars_:
                v.set(val)

        def ok():
            hidden = set(i for i, v in enumerate(vars_) if not v.get())
            if len(hidden) == len(vars_):
                messagebox.showwarning(APP_NAME, "Keep at least one column.", parent=d)
                return
            self.hidden = hidden
            d.destroy()
            self.apply_view()
        bf = ttk.Frame(d)
        bf.pack(fill="x")
        ttk.Button(bf, text="Show all", command=lambda: set_all(True)).pack(side="left")
        ttk.Button(bf, text="Hide all", command=lambda: set_all(False)).pack(side="left", padx=4)
        ttk.Button(bf, text="Apply", command=ok).pack(side="right")
        ttk.Button(bf, text="Cancel", command=d.destroy).pack(side="right", padx=4)

    def stats_dialog(self):
        if not self._need_table():
            return
        t = self.table
        rows = []
        for c in self.visible_columns():
            nums = [x for x in (_as_number(t.rows[i][c]) for i in self.order
                                if c < len(t.rows[i])) if x is not None]
            if not nums:
                continue
            n = len(nums)
            mean = sum(nums) / n
            sd = math.sqrt(sum((x - mean) ** 2 for x in nums) / (n - 1)) if n > 1 else None
            rows.append([t.columns[c], n, min(nums), max(nums), mean, sd, sum(nums)])
        if not rows:
            messagebox.showinfo(APP_NAME, "This table has no numeric columns.",
                                parent=self.root)
            return
        d = self._dialog("Statistics")
        ttk.Label(d, text="Statistics of the %d rows currently shown" % len(self.order),
                  font=self.bold).pack(anchor="w", pady=(0, 6))
        cols = ["column", "count", "minimum", "maximum", "average", "std. deviation", "sum"]
        tv = ttk.Treeview(d, columns=cols, show="headings", height=min(len(rows), 15))
        for c in cols:
            tv.heading(c, text=c, anchor="w")
            tv.column(c, width=170 if c == "column" else 110, anchor="w" if c == "column" else "e")
        for r in rows:
            tv.insert("", "end", values=[T.fmt(v) for v in r])
        tv.pack(fill="both", expand=True)

        def copy():
            self._to_clipboard("\t".join(cols) + "\n" + "\n".join(
                "\t".join(T.fmt(v, True) for v in r) for r in rows))
        bf = ttk.Frame(d)
        bf.pack(fill="x", pady=(8, 0))
        ttk.Button(bf, text="Close", command=d.destroy).pack(side="right")
        ttk.Button(bf, text="Copy", command=copy).pack(side="right", padx=4)

    def plot_dialog(self):
        if not self._need_table():
            return
        t = self.table
        numeric = [t.columns[c] for c in range(len(t.columns))
                   if any(_as_number(r[c]) is not None for r in t.rows[:500] if c < len(r))]
        if not numeric:
            messagebox.showinfo(APP_NAME, "This table has no numeric column to plot.",
                                parent=self.root)
            return
        PlotWindow(self, numeric)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _table_text_rows(self, precise=True):
        cols = self.visible_columns()
        yield [self.table.columns[c] for c in cols]
        for i in self.order:
            r = self.table.rows[i]
            yield [T.fmt(r[c], precise) if c < len(r) else "" for c in cols]

    def export_csv(self):
        if not self._need_table():
            return
        name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in self.table.title)
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Save table as CSV", defaultextension=".csv",
            initialfile=name.strip("_") + ".csv",
            filetypes=[("CSV file", "*.csv"), ("All files", "*")])
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                for row in self._table_text_rows():
                    w.writerow(row)
        except OSError as e:
            messagebox.showerror(APP_NAME, "Could not save the file:\n%s" % e,
                                 parent=self.root)
            return
        self.status.set("Saved %d rows to %s" % (len(self.order), path))

    def _to_clipboard(self, text):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def copy_table(self):
        if not self._need_table():
            return
        self._to_clipboard("\n".join("\t".join(r) for r in self._table_text_rows()))
        self.status.set("Copied %d rows to the clipboard." % len(self.order))

    def copy_selection(self):
        sel = self.grid.selection()
        if not sel:
            return self.copy_table()
        cols = self.visible_columns()
        lines = ["\t".join(self.table.columns[c] for c in cols)]
        for iid in sel:
            vals = self.grid.item(iid, "values")
            lines.append("\t".join(str(vals[c]) for c in cols if c < len(vals)))
        self._to_clipboard("\n".join(lines))
        self.status.set("Copied %d selected rows." % len(sel))

    def show_help(self):
        d = self._dialog("Help")
        txt = tk.Text(d, width=78, height=34, wrap="word", relief="flat",
                      font=self.font, padx=6, pady=6)
        txt.insert("1.0", HELP_TEXT)
        txt.configure(state="disabled")
        txt.pack(fill="both", expand=True)
        ttk.Button(d, text="Close", command=d.destroy).pack(anchor="e", pady=(6, 0))


class PlotWindow(object):
    """A very simple line / point graph drawn on a tkinter Canvas."""

    def __init__(self, app, numeric):
        self.app = app
        t = app.table
        self.d = d = app._dialog("Plot - %s" % t.title)
        d.geometry("860x560")
        top = ttk.Frame(d)
        top.pack(fill="x")
        ttk.Label(top, text="Y (vertical):").pack(side="left")
        prefer = [c for c in ("mean", "k (this generation)", "k-effective", "value")
                  if c in numeric]
        self.y = ttk.Combobox(top, state="readonly", values=numeric, width=22)
        self.y.set(prefer[0] if prefer else numeric[-1])
        self.y.pack(side="left", padx=4)
        ttk.Label(top, text="X (horizontal):").pack(side="left", padx=(10, 0))
        self.x = ttk.Combobox(top, state="readonly", values=["row number"] + numeric,
                              width=22)
        self.x.set("row number")
        self.x.pack(side="left", padx=4)
        self.logy = tk.BooleanVar()
        ttk.Checkbutton(top, text="log scale (Y)", variable=self.logy,
                        command=self.draw).pack(side="left", padx=8)
        self.err = tk.BooleanVar(value="std. dev." in t.columns)
        if "std. dev." in t.columns:
            ttk.Checkbutton(top, text="error bars (std. dev.)", variable=self.err,
                            command=self.draw).pack(side="left")
        for cb in (self.x, self.y):
            cb.bind("<<ComboboxSelected>>", lambda e: self.draw())
        ttk.Label(d, text="The graph uses the rows currently shown in the table "
                  "(after search and sorting).", style="Note.TLabel").pack(anchor="w")
        self.c = tk.Canvas(d, background="white", highlightthickness=0)
        self.c.pack(fill="both", expand=True, pady=4)
        self.c.bind("<Configure>", lambda e: self.draw())

    def draw(self):
        c = self.c
        c.delete("all")
        t = self.app.table
        W, H = c.winfo_width(), c.winfo_height()
        if W < 50 or H < 50:
            return
        yi = t.columns.index(self.y.get())
        xi = t.columns.index(self.x.get()) if self.x.get() in t.columns else None
        ei = t.columns.index("std. dev.") if (self.err.get() and "std. dev." in t.columns) else None
        logy = self.logy.get()
        pts = []
        for n, i in enumerate(self.app.order):
            r = t.rows[i]
            yv = _as_number(r[yi]) if yi < len(r) else None
            xv = float(n + 1) if xi is None else (_as_number(r[xi]) if xi < len(r) else None)
            if yv is None or xv is None or (logy and yv <= 0):
                continue
            ev = _as_number(r[ei]) if ei is not None and ei < len(r) else None
            pts.append((xv, yv, ev or 0.0))
        if not pts:
            c.create_text(W / 2, H / 2, text="Nothing to plot (no numeric values).")
            return
        tr = (lambda v: math.log10(v)) if logy else (lambda v: v)
        xs = [p[0] for p in pts]
        ylo = [tr(p[1] - p[2]) if (not logy or p[1] - p[2] > 0) else tr(p[1]) for p in pts]
        yhi = [tr(p[1] + p[2]) for p in pts]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ylo), max(yhi)
        if x1 == x0:
            x0, x1 = x0 - 1, x1 + 1
        if y1 == y0:
            y0, y1 = y0 - 1, y1 + 1
        pad = (y1 - y0) * 0.05
        y0, y1 = y0 - pad, y1 + pad
        L, R, TOP, B = 90, 20, 20, 50

        def px(x):
            return L + (x - x0) / (x1 - x0) * (W - L - R)

        def py(y):
            return H - B - (y - y0) / (y1 - y0) * (H - TOP - B)
        c.create_rectangle(L, TOP, W - R, H - B, outline="#888888")
        for k in range(6):
            yv = y0 + (y1 - y0) * k / 5
            yy = py(yv)
            c.create_line(L, yy, W - R, yy, fill="#eeeeee")
            c.create_text(L - 6, yy, anchor="e", text=T.fmt(10 ** yv if logy else yv))
            xv = x0 + (x1 - x0) * k / 5
            xx = px(xv)
            c.create_line(xx, TOP, xx, H - B, fill="#eeeeee")
            c.create_text(xx, H - B + 12, anchor="n", text=T.fmt(xv))
        c.create_text((L + W - R) / 2, H - 12, text=self.x.get())
        c.create_text(12, (TOP + H - B) / 2, text=self.y.get(), angle=90)
        color = "#1f5fa8"
        coords = []
        for (xv, yv, ev), lo, hi in zip(pts, ylo, yhi):
            X, Y = px(xv), py(tr(yv))
            coords.extend((X, Y))
            if ev:
                c.create_line(X, py(lo), X, py(hi), fill="#9bb8dd")
        if len(pts) > 1 and len(pts) <= 5000:
            c.create_line(*coords, fill=color, width=1.5)
        r = 2.5 if len(pts) < 300 else 1.2
        for k in range(0, len(coords), 2):
            X, Y = coords[k], coords[k + 1]
            c.create_oval(X - r, Y - r, X + r, Y + r, fill=color, outline=color)


def main(paths=()):
    root = tk.Tk()
    App(root, paths)
    root.mainloop()
