#!/usr/bin/env python3
"""Start the OpenMC Output Viewer:  python3 openmc_viewer.py [file.h5 ...]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openmc_viewer.__main__ import run  # noqa: E402

if __name__ == "__main__":
    run()
