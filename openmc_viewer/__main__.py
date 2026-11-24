import sys


def run():
    try:
        import tkinter  # noqa: F401
    except ImportError:
        sys.stderr.write(
            "The graphical toolkit 'tkinter' is missing.\n"
            "It is part of Python but Ubuntu ships it separately. Install it with:\n\n"
            "    sudo apt install python3-tk\n\n")
        sys.exit(1)
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print("usage: python3 openmc_viewer.py [FILE ...]\n\n"
              "Opens the OpenMC Output Viewer. Optional FILEs (.h5 or .xml) are "
              "opened at start.")
        return
    from .gui import main
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
