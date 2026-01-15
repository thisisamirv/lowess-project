#!/usr/bin/env python3
import argparse
import signal
import sys
import shutil
import subprocess
import os
import re

CARGO_TOML = "Cargo.toml"
BACKUP_TOML = "Cargo.toml.bak"

# Mapping of Make targets to Cargo workspace members (partial path matching)
TARGET_MAPPING = {
    "crates/lowess": ["crates/lowess"],
    "crates/fastLowess": ["crates/fastLowess"],
    "bindings/python": ["bindings/python"],
    "bindings/julia": ["bindings/julia"],
    "bindings/nodejs": ["bindings/nodejs"],
    "bindings/wasm": ["bindings/wasm"],
    "bindings/cpp": ["bindings/cpp"],
    # For R, it's not a member, so we essentially just keep core crates
    "bindings/r": [], 
}

ALWAYS_KEEP = {"examples", "tests"}

def restore_cargo_toml():
    """Restores Cargo.toml from backup if it exists."""
    if os.path.exists(BACKUP_TOML):
        # Atomic move/restore
        shutil.move(BACKUP_TOML, CARGO_TOML)
        # print("Restored Cargo.toml")

def signal_handler(sig, frame):
    """Handle interruption signals."""
    print("\nInterrupted! Restoring Cargo.toml...")
    restore_cargo_toml()
    sys.exit(130) # standard exit code for SIGINT

def isolate_members(keep_member_key):
    """
    Reads Cargo.toml, comments out members not in keep_list + ALWAYS_KEEP,
    and writes back to Cargo.toml.
    """
    if not os.path.exists(CARGO_TOML):
        print(f"Error: {CARGO_TOML} not found.")
        sys.exit(1)

    # Determine which members to keep
    keep_list = set(ALWAYS_KEEP)
    if keep_member_key:
         # direct path or key
         if keep_member_key in TARGET_MAPPING:
             keep_list.update(TARGET_MAPPING[keep_member_key])
         else:
             # Fallback: assume the key is the path itself
             keep_list.add(keep_member_key)
             # Also assume dependencies might be needed? 
             # For now, we trust the user/makefile to pass the correct target path.

    # print(f"Isolating workspace members. Keeping: {keep_list}")

    shutil.copy2(CARGO_TOML, BACKUP_TOML)

    with open(BACKUP_TOML, 'r') as f:
        lines = f.readlines()

    new_lines = []
    in_members = False
    
    for line in lines:
        stripped = line.strip()
        
        # Detect start of [workspace] members
        if stripped.startswith("members = ["):
            in_members = True
            new_lines.append(line)
            continue
        
        if in_members:
            if stripped == "]":
                in_members = False
                new_lines.append(line)
                continue
            
            # This is a member line (e.g. "crates/lowess",)
            # We extracting the string inside quotes
            match = re.search(r'"([^"]+)"', stripped)
            if match:
                member_path = match.group(1)
                if member_path not in keep_list:
                    # Comment it out if not already commented
                    if not stripped.startswith("#"):
                        # Preserve indentation
                        indent = line[:line.find('"')] if '"' in line else ""
                        new_lines.append(f"{indent}# \"{member_path}\",\n")
                    else:
                        new_lines.append(line)
                else:
                    # Ensure it's uncommented if it was commented (though we start from clean source ideally)
                    # But here we just keep it as is if it matches
                    new_lines.append(line)
            else:
                # Empty lines or comments inside members list
                new_lines.append(line)
        else:
            new_lines.append(line)

    with open(CARGO_TOML, 'w') as f:
        f.writelines(new_lines)


def main():
    parser = argparse.ArgumentParser(description="Isolate Cargo workspace members.")
    parser.add_argument("target_path", help="The workspace member path to keep active (e.g. bindings/python)")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after isolation")

    args = parser.parse_args()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        isolate_members(args.target_path)
        
        # Run the command
        if args.command:
            # args.command[0] might be '--' separator
            cmd = args.command[1:] if args.command[0] == '--' else args.command
            
            if not cmd:
                print("No command specified.")
                return

            # print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd)
            sys.exit(result.returncode)
            
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
    finally:
        restore_cargo_toml()

if __name__ == "__main__":
    main()
