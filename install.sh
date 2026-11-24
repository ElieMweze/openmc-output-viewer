#!/bin/bash
# Adds "OpenMC Output Viewer" to the Ubuntu applications menu.
# Usage:  bash install.sh        (run from the project folder)
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 is not installed. Install it with:  sudo apt install python3"
    exit 1
fi
if ! python3 -c "import tkinter" >/dev/null 2>&1; then
    echo "The Python graphical toolkit (tkinter) is missing."
    echo "Install it with:  sudo apt install python3-tk"
    exit 1
fi

APPS="$HOME/.local/share/applications"
mkdir -p "$APPS"
cat > "$APPS/openmc-output-viewer.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=OpenMC Output Viewer
Comment=Read OpenMC output files (.h5 and .xml) as tables
Exec=python3 "$DIR/openmc_viewer.py" %F
Icon=x-office-spreadsheet
Terminal=false
Categories=Science;Education;
MimeType=application/x-hdf5;application/x-hdf;application/xml;
DESKTOP
chmod +x "$DIR/openmc_viewer.py"
echo "Done. Look for 'OpenMC Output Viewer' in the applications menu,"
echo "or start it with:  python3 $DIR/openmc_viewer.py"
