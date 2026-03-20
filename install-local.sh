#!/usr/bin/env bash
# install-local.sh
# Installs Sift icons and desktop entry for the current user.
# Run this once after cloning — no root required.
# The app itself is still run directly with: python3 sift.py

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ICON_DIR="$HOME/.local/share/icons/hicolor"
APP_DIR="$HOME/.local/share/applications"

echo "Installing icons..."
mkdir -p "$ICON_DIR/scalable/apps"
mkdir -p "$ICON_DIR/symbolic/apps"
cp "$SCRIPT_DIR/sift.svg"          "$ICON_DIR/scalable/apps/io.github.IdleEndeavor.Sift.svg"
cp "$SCRIPT_DIR/sift-symbolic.svg" "$ICON_DIR/symbolic/apps/io.github.IdleEndeavor.Sift-symbolic.svg"
gtk-update-icon-cache "$ICON_DIR" 2>/dev/null || true

echo "Installing desktop entry..."
mkdir -p "$APP_DIR"

# Write a desktop file pointing at the actual script location
cat > "$APP_DIR/io.github.IdleEndeavor.Sift.desktop" <<EOF
[Desktop Entry]
Name=Sift
Comment=Tinder for Your Music Library
Exec=python3 $SCRIPT_DIR/sift.py
Icon=io.github.IdleEndeavor.Sift
Type=Application
Categories=Audio;Music;GTK;
Keywords=music;library;sort;tinder;
StartupWMClass=sift
EOF

update-desktop-database "$APP_DIR" 2>/dev/null || true

echo ""
echo "Done. Sift will now appear in your app launcher."
echo "To run from terminal: python3 $SCRIPT_DIR/sift.py"
