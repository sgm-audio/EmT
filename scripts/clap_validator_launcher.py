#!/usr/bin/env python3
"""
clap_validator_launcher.py

Patches a copy of clap-validator.exe with a 16 MB stack reserve so that
its child processes (one per test) inherit enough stack for EmT's
SSM state. The plugin DLL's /STACK does not help—children get the
validator's stack, not the plugin's.

Usage (invoked by CMake's clap-validator test):
  python scripts/clap_validator_launcher.py <clap-validator.exe> <plugin.clap>

Exit code mirrors clap-validator.
"""

import os
import sys
import shutil
import subprocess
import tempfile
import stat
import platform
import sysconfig

def find_editbin():
    """Locate editbin.exe via vswhere or common VS paths."""
    # Try vswhere first
    vswhere = shutil.which("vswhere")
    if vswhere:
        try:
            out = subprocess.check_output(
                [vswhere, "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.VCTools",
                 "-property", "installationPath"],
                text=True, stderr=subprocess.DEVNULL
            ).strip()
            if out:
                cand = os.path.join(out, "VC", "Tools", "MSVC", "*", "bin", "Hostx64", "x64", "editbin.exe")
                for path in sorted(glob(cand), reverse=True):
                    if os.path.isfile(path):
                        return path
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    # Fallback: scan common locations
    for base in [
        r"C:\Program Files\Microsoft Visual Studio",
        r"C:\Program Files (x86)\Microsoft Visual Studio",
        os.environ.get("VCINSTALLDIR", ""),
    ]:
        if base:
            for root, dirs, files in os.walk(base):
                if "editbin.exe" in files:
                    return os.path.join(root, "editbin.exe")
    return None

# Windows-only glob helper
import glob as _glob
def glob(pattern):
    return _glob.glob(pattern)

def patch_stack(exe_path, stack_bytes=16_777_216):
    """Use editbin to set the stack reserve on a PE executable."""
    editbin = find_editbin()
    if not editbin:
        raise RuntimeError("editbin.exe not found; install VS Build Tools")
    subprocess.check_call([editbin, f"/STACK:{stack_bytes}", exe_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <clap-validator.exe> <plugin.clap>", file=sys.stderr)
        return 2

    validator_src = sys.argv[1]
    plugin_path = sys.argv[2]

    if not os.path.isfile(validator_src):
        print(f"clap-validator not found: {validator_src}", file=sys.stderr)
        return 2
    if not os.path.isdir(plugin_path):
        print(f"Plugin bundle not found: {plugin_path}", file=sys.stderr)
        return 2

    # Work in a temp dir so we don't mutate the installed validator
    with tempfile.TemporaryDirectory(prefix="clapv_") as tmp:
        validator_dst = os.path.join(tmp, "clap-validator.exe")
        shutil.copy2(validator_src, validator_dst)

        # Make writable (copy2 preserves read-only on Windows)
        os.chmod(validator_dst, stat.S_IWRITE | stat.S_IREAD)

        try:
            patch_stack(validator_dst)
        except Exception as e:
            print(f"Failed to patch stack: {e}", file=sys.stderr)
            return 1

        # Run the patched validator with the 'validate' subcommand
        # clap-validator 0.4.1+ requires the subcommand; bare path exits 2
        cmd = [validator_dst, "validate", plugin_path]
        try:
            result = subprocess.run(cmd, capture_output=False)
            return result.returncode
        except FileNotFoundError:
            print(f"Failed to execute {validator_dst}", file=sys.stderr)
            return 1

if __name__ == "__main__":
    sys.exit(main())