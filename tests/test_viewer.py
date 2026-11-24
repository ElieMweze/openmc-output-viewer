"""Tests (standard library only):  python3 -m unittest discover tests"""
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
EX = os.path.join(HERE, "..", "examples")

from openmc_viewer import tables as T  # noqa: E402
from openmc_viewer.hdf5 import File, is_hdf5  # noqa: E402


class StatepointTests(unittest.TestCase):
    def setUp(self):
        self.f = File(os.path.join(EX, "statepoint.100.h5"))

    def tearDown(self):
        self.f.close()

    def test_detect(self):
        self.assertTrue(is_hdf5(os.path.join(EX, "statepoint.100.h5")))
        self.assertFalse(is_hdf5(os.path.join(EX, "materials.xml")))
        self.assertEqual(T.file_type(self.f), "statepoint")

    def test_structure(self):
        self.assertIn("tallies", self.f.root.keys())
        self.assertEqual(self.f["n_batches"].read(), [100])
        self.assertEqual(self.f["tallies/tally 1/nuclides"].read(), ["U235", "U238", "total"])
        self.assertEqual(self.f["tallies/tally 1/results"].shape, (9, 9, 2))
        self.assertEqual(list(self.f["tallies"].attrs["ids"]), [1, 2, 3])

    def test_tally_statistics(self):
        tg = self.f["tallies/tally 1"]
        data = tg["results"].read()
        n = 80
        mean = data[0] / n
        std = math.sqrt((data[1] / n - mean ** 2) / (n - 1))
        t = T.tally_table(self.f, tg)
        self.assertEqual(len(t.rows), 81)
        row = t.rows[0]
        self.assertEqual(row[:4], [1, "0 - 0.625", "U235", "fission"])
        self.assertAlmostEqual(row[4], mean)
        self.assertAlmostEqual(row[5], std)

    def test_mesh_labels(self):
        t = T.tally_table(self.f, self.f["tallies/tally 2"])
        self.assertEqual([r[0] for r in t.rows[:5]],
                         ["(1, 1)", "(2, 1)", "(3, 1)", "(4, 1)", "(1, 2)"])

    def test_all_quick_views(self):
        for title, desc, func in T.quick_views(self.f):
            t = func()
            self.assertTrue(t.columns, title)

    def test_compound(self):
        t = T.dataset_table(self.f["source_bank"])
        self.assertIn("r.x", t.columns)
        self.assertEqual(len(t.rows), 200)


class SummaryTests(unittest.TestCase):
    def test_materials(self):
        with File(os.path.join(EX, "summary.h5")) as f:
            t = T.materials_composition(f)
            self.assertEqual(t.rows[0], [1, "UO2 fuel", "U235", 7.2e-4])
            cells = T.combine_subgroups(f["geometry/cells"])
            self.assertEqual(cells.columns[0], "folder")
            self.assertEqual(len(cells.rows), 3)


class XMLTests(unittest.TestCase):
    def test_children_table(self):
        import xml.etree.ElementTree as ET
        root = ET.parse(os.path.join(EX, "materials.xml")).getroot()
        t = T.xml_children_table(root)
        self.assertEqual(len(t.rows), 3)
        mat = T.xml_children_table(root[0])
        self.assertEqual([r[t.columns.index("element")] for r in mat.rows].count("nuclide"), 3)


if __name__ == "__main__":
    unittest.main()
