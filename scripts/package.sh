#!/usr/bin/env bash
# scripts/package.sh
# Builds a CLAP bundle layout from the build directory.
# Usage: ./scripts/package.sh <build_dir>

set -euo pipefail

BUILD_DIR="${1:-build/Release}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${PROJECT_ROOT}/dist"

# Detect platform
UNAME="$(uname -s)"
case "${UNAME}" in
    Linux*)     PLATFORM="Linux" ;;
    Darwin*)    PLATFORM="macOS" ;;
    CYGWIN*|MINGW*|MSYS*) PLATFORM="Windows" ;;
    *)          PLATFORM="Unknown" ;;
esac

echo "Packaging for ${PLATFORM} from ${BUILD_DIR}"

# Find the built plugin
if [[ "${PLATFORM}" == "Windows" ]]; then
    PLUGIN_PATH="${BUILD_DIR}/EmTSynth.clap"
    PLUGIN_NAME="EmTSynth"
    BUNDLE_DIR="${DIST_DIR}/Windows/${PLUGIN_NAME}.clap"
    ZIP_NAME="${PLUGIN_NAME}-Windows.zip"
elif [[ "${PLATFORM}" == "Darwin" ]]; then
    # macOS CLAP bundles are directories with Contents/MacOS/
    PLUGIN_PATH="${BUILD_DIR}/EmTSynth.clap"
    PLUGIN_NAME="EmTSynth"
    BUNDLE_DIR="${DIST_DIR}/macOS/${PLUGIN_NAME}.clap"
    ZIP_NAME="${PLUGIN_NAME}-macOS.zip"
else
    # Linux: .clap directory
    PLUGIN_PATH="${BUILD_DIR}/EmTSynth.clap"
    PLUGIN_NAME="EmTSynth"
    BUNDLE_DIR="${DIST_DIR}/Linux/${PLUGIN_NAME}.clap"
    ZIP_NAME="${PLUGIN_NAME}-Linux.zip"
fi

if [[ ! -e "${PLUGIN_PATH}" ]]; then
    echo "ERROR: Plugin not found at ${PLUGIN_PATH}" >&2
    exit 1
fi

# Clean and create bundle directory
rm -rf "${BUNDLE_DIR}"
mkdir -p "${BUNDLE_DIR}"

# Copy plugin bundle
if [[ "${PLATFORM}" == "Darwin" ]]; then
    # macOS: copy the entire .clap bundle (directory)
    cp -R "${PLUGIN_PATH}"/* "${BUNDLE_DIR}/"
else
    # Windows/Linux: .clap is a directory
    cp -R "${PLUGIN_PATH}"/* "${BUNDLE_DIR}/"
fi

# Create MANIFEST.txt
MANIFEST="${BUNDLE_DIR}/MANIFEST.txt"
{
    echo "EmT CLAP Plugin Bundle"
    echo "======================"
    echo ""
    echo "Plugin: ${PLUGIN_NAME}"
    echo "Platform: ${PLATFORM}"
    echo "Build: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
    echo "Git Commit: $(cd "${PROJECT_ROOT}" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")"
    echo "Git Branch: $(cd "${PROJECT_ROOT}" && git branch --show-current 2>/dev/null || echo "unknown")"
    echo ""
    echo "Files:"
    find "${BUNDLE_DIR}" -type f | sort | while read -r f; do
        size=$(stat -c%s "${f}" 2>/dev/null || stat -f%z "${f}" 2>/dev/null)
        rel="${f#${BUNDLE_DIR}/}"
        printf "  %-60s %10d bytes\n" "${rel}" "${size}"
    done
    echo ""
    total_size=$(du -sb "${BUNDLE_DIR}" 2>/dev/null | cut -f1 || du -sk "${BUNDLE_DIR}" 2>/dev/null | awk '{print $1*1024}')
    echo "Total size: ${total_size} bytes"
} > "${MANIFEST}"

# Create zip
mkdir -p "${DIST_DIR}/${PLATFORM}"
cd "${DIST_DIR}/${PLATFORM}"
zip -r "${ZIP_NAME}" "${PLUGIN_NAME}.clap" >/dev/null

echo "Created: ${DIST_DIR}/${PLATFORM}/${ZIP_NAME}"
echo "Manifest: ${MANIFEST}"
ls -lh "${DIST_DIR}/${PLATFORM}/${ZIP_NAME}"