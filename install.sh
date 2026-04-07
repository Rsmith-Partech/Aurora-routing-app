#!/bin/bash
# Aurora Audio Router — installer / updater
# Run this script on the Raspberry Pi as your normal user (not sudo).
# It installs the app, sets up the desktop icon, and installs dependencies.
# Re-running this script will update the app to the latest version.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_FILE="$SCRIPT_DIR/audio_router.py"
DESKTOP_SRC="$SCRIPT_DIR/audio-router.desktop"

INSTALL_DIR="$HOME/audio-router"
DESKTOP_DIR="$HOME/Desktop"
APP_MENU_DIR="$HOME/.local/share/applications"

echo "=== Aurora Audio Router — Installer ==="
echo "User     : $(whoami)"
echo "Home     : $HOME"
echo "Install  : $INSTALL_DIR"
echo ""

# ---- Detect python3 ----
if command -v python3.12 &>/dev/null; then
    PYTHON="python3.12"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    echo "ERROR: python3 not found. Please install Python 3." >&2
    exit 1
fi
echo "Python   : $($PYTHON --version)"

# ---- Install system dependencies ----
echo ""
echo "Installing dependencies (requires sudo)..."
sudo apt-get update -qq
sudo apt-get install -y python3-tk python3-numpy portaudio19-dev 2>/dev/null || \
sudo apt-get install -y python3-tk python3-numpy 2>/dev/null || true

# ---- Install Python packages ----
echo ""
echo "Installing Python packages..."
if $PYTHON -m pip install sounddevice numpy --break-system-packages 2>/dev/null; then
    echo "Installed sounddevice and numpy via pip (--break-system-packages)"
elif $PYTHON -m pip install sounddevice numpy 2>/dev/null; then
    echo "Installed sounddevice and numpy via pip"
else
    echo "WARNING: pip install failed — trying pipx / system packages as fallback" >&2
    sudo apt-get install -y python3-sounddevice python3-numpy 2>/dev/null || true
fi

# ---- Create install directory ----
mkdir -p "$INSTALL_DIR"

# ---- Copy app file ----
cp "$APP_FILE" "$INSTALL_DIR/audio_router.py"
chmod +x "$INSTALL_DIR/audio_router.py"
echo "Installed: $INSTALL_DIR/audio_router.py"

# ---- Write desktop file with correct paths ----
mkdir -p "$DESKTOP_DIR" "$APP_MENU_DIR"

DESKTOP_CONTENT="[Desktop Entry]
Version=1.0
Type=Application
Name=Aurora Audio Router
Comment=Raspberry Pi audio cross-router (A↔B)
Exec=$PYTHON $INSTALL_DIR/audio_router.py
Icon=audio-card
Terminal=false
Categories=AudioVideo;Audio;Utility;
StartupNotify=true"

echo "$DESKTOP_CONTENT" > "$DESKTOP_DIR/audio-router.desktop"
echo "$DESKTOP_CONTENT" > "$APP_MENU_DIR/audio-router.desktop"
chmod +x "$DESKTOP_DIR/audio-router.desktop"
chmod +x "$APP_MENU_DIR/audio-router.desktop"

echo "Installed: $DESKTOP_DIR/audio-router.desktop"
echo "Installed: $APP_MENU_DIR/audio-router.desktop"

# ---- Trust desktop file (GNOME/Nautilus) ----
if command -v gio &>/dev/null; then
    gio set "$DESKTOP_DIR/audio-router.desktop" metadata::trusted true 2>/dev/null || true
fi

echo ""
echo "=== Installation complete ==="
echo ""
echo "To launch the app:"
echo "  $PYTHON $INSTALL_DIR/audio_router.py"
echo ""
echo "Or double-click the 'Aurora Audio Router' icon on your Desktop."
