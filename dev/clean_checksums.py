#!/usr/bin/env python3
"""Remove non-essential vendored files and refresh cargo checksum metadata."""

import argparse
import hashlib
import json
import os
import shutil


def sha256_file(path):
    """Calculate SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_checksums(vendor_dir, quiet=False):
    """Strip vendored extras and update `.cargo-checksum.json` files."""
    if not quiet:
        print(f"Cleaning checksums in {vendor_dir}...")

    # 1. Remove non-essential directories
    strip_dirs = ["tests", "benches", "examples", "doc", "docs", ".github", ".config"]

    for root, dirs, files in os.walk(vendor_dir):
        # Remove hidden directories and strip_dirs
        del files
        for directory in list(dirs):
            if directory.startswith(".") or directory in strip_dirs:
                full_path = os.path.join(root, directory)
                shutil.rmtree(full_path, ignore_errors=True)
                dirs.remove(directory)

    # 2. Remove hidden files (except checksums)
    for root, dirs, files in os.walk(vendor_dir):
        del dirs
        for filename in files:
            if filename.startswith(".") and filename != ".cargo-checksum.json":
                full_path = os.path.join(root, filename)
                try:
                    os.remove(full_path)
                except OSError:
                    pass

    updated_count = 0
    for root, dirs, files in os.walk(vendor_dir):
        del dirs
        if ".cargo-checksum.json" in files:
            filepath = os.path.join(root, ".cargo-checksum.json")
            try:
                with open(filepath, "r", encoding="utf-8") as file_handle:
                    data = json.load(file_handle)

                if "files" in data:
                    original_files = data["files"]

                    # Remove keys that:
                    # 1. Were hidden files (handled above)
                    # 2. Don't exist on disk (were inside stripped dirs)
                    # Also: recalculate checksums for files that exist (handles CRLF->LF)
                    new_files = {}
                    for key, _checksum in original_files.items():
                        file_path = os.path.join(root, key)
                        exists = os.path.exists(file_path)
                        is_hidden = any(part.startswith(".") for part in key.split("/"))

                        if exists and not is_hidden:
                            # Recalculate checksum to handle line ending changes
                            new_hash = sha256_file(file_path)
                            new_files[key] = new_hash
                        # If it doesn't exist or is hidden, it is excluded from checksum

                    if original_files != new_files:
                        data["files"] = new_files
                        with open(filepath, "w", encoding="utf-8") as file_handle:
                            json.dump(data, file_handle)
                        updated_count += 1
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
                if not quiet:
                    print(f"  Error processing {filepath}: {error}")

    if not quiet:
        print(f"Done. Updated {updated_count} checksum files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean vendor checksums")
    parser.add_argument(
        "vendor_dir", nargs="?", default="src/vendor", help="Vendor directory"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()
    clean_checksums(args.vendor_dir, args.quiet)
