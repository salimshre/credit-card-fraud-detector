#!/usr/bin/env python3
""" 
PREPARE FOR DEEPSEEK – Python script (with .gitignore support & config file)

Step 1: Generate FILE_STRUCTURE.md (project tree)
Step 2: Copy code files to tmp/ (flat)

Usage:  python prepare_for_deepseek.py
        (run from your project root; optionally place a prepare_config.json there)
"""

import json
import os
import shutil
import sys
import platform
import csv
import subprocess
import argparse
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

# Try to import pathspec for .gitignore support
try:
    import pathspec
    HAVE_PATHSPEC = True
except ImportError:
    HAVE_PATHSPEC = False

# ---------- Default configuration (mirrors your updated .bat) ----------
DEFAULT_CONFIG = {
    "code_extensions": [".php", ".js", ".css", ".html", ".py"],
    "skip_top_level": [
        "archive", ".git", ".github", ".agents", "tmp",
        "journal", "data", "deploy", "versions"
    ],
    "skip_any_depth": [
        "archive", "parsedown", "icons", "__pycache__",
        ".venv", "server-logs", "trash", "previous-version", "tmp"
    ],
    "tree_exclude": [".git", ".agents", "FILE_STRUCTURE.md"],
    "tree_summarize": ["__pycache__", ".pytest_cache", ".ruff_cache"],
    "max_tree_depth": 10,
    "tmp_dir": "tmp",
    "tree_output_file": "FILE_STRUCTURE.md",
    "use_gitignore": True,
    "open_csv": False,
#    "force_include": []
    
}

def human_size(n_bytes: int) -> str:
    """Return human readable file size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"

def load_config(args: argparse.Namespace = None) -> dict:
    """Load config from 'prepare_config.json' if it exists, else use defaults.
    Optional CLI args can override defaults (e.g., extra extensions, open_csv)."""
    config_path = Path("prepare_config.json")
    config = DEFAULT_CONFIG.copy()
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            config.update(user_config)
            print("   Loaded settings from prepare_config.json")
        except Exception as e:
            print(f"   Warning: Could not parse config file ({e}). Using defaults.")
    # CLI overrides
    if args:
        if args.extra_extensions:
            extra = [ext if ext.startswith('.') else f".{ext}" for ext in args.extra_extensions.split(',')]
            config["code_extensions"] = list(set(config.get("code_extensions", []) + extra))
        if args.open_csv:
            config["open_csv"] = True
    return config

def validate_config(config: dict) -> None:
    """Validate config keys and types using a simple schema. Raises ValueError on failure."""
    # Minimal schema – expand as needed
    required_keys = {
        "code_extensions": list,
        "skip_top_level": list,
        "skip_any_depth": list,
        "tree_exclude": list,
        "tree_summarize": list,
        "max_tree_depth": int,
        "tmp_dir": str,
        "tree_output_file": str,
        "use_gitignore": bool,
        "open_csv": bool,
        "force_include": list,
    }
    for key, expected_type in required_keys.items():
        if key not in config:
            raise ValueError(f"Missing config key: {key}")
        if not isinstance(config[key], expected_type):
            raise ValueError(f"Config key '{key}' expects {expected_type.__name__}, got {type(config[key]).__name__}")


# ---------- Tree generation ----------
def generate_tree(root: Path, config: dict) -> None:
    """Write the Markdown tree file."""
    output_path = root / config["tree_output_file"]
    exclude_set = set(config["tree_exclude"])
    summary_set = set(config["tree_summarize"])
    max_depth = config["max_tree_depth"]

    dir_count = 0
    file_count = 0
    lines = []

    def walk(directory: Path, prefix: str = "", depth: int = 0):
        nonlocal dir_count, file_count
        if depth > max_depth:
            return
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return
        for i, entry in enumerate(entries):
            if entry.name in exclude_set:
                continue
            is_last = (i == len(entries) - 1)
            branch = "└── " if is_last else "├── "
            indent_cont = "    " if is_last else "│   "

            if entry.is_dir() and entry.name in summary_set:
                lines.append(f"{prefix}{branch}{entry.name}/ (summary: compiled/pycache)")
                dir_count += 1
                continue

            line = f"{prefix}{branch}{entry.name}"
            if entry.is_dir():
                line += "/"
                lines.append(line)
                dir_count += 1
                walk(entry, prefix + indent_cont, depth + 1)
            else:
                size = human_size(entry.stat().st_size)
                line += f" ({size})"
                lines.append(line)
                file_count += 1

    walk(root)

    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    header = (
        f"# Repository file structure\n\n"
        f"Generated: {now}\n\n"
        f"A cleaner, hierarchical view of the repository. Directories end with '/'.\n"
        f"Cache folders ({', '.join(summary_set)}) are summarised.\n"
        f"Excluded: {', '.join(sorted(exclude_set))}.\n\n"
        f"## Summary\n"
        f"- Directories: {dir_count}\n"
        f"- Files: {file_count}\n\n"
        f"## Tree\n"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(lines))
        f.write("\n")

    print(f"   {output_path.name} generated successfully ({dir_count} dirs, {file_count} files).")

# ---------- .gitignore pattern matcher ----------
def get_gitignore_spec(root: Path) -> Optional[object]:
    """Return a pathspec object from .gitignore, or None if not available."""
    gitignore_path = root / ".gitignore"
    if not gitignore_path.exists() or not HAVE_PATHSPEC:
        return None
    try:
        with open(gitignore_path, "r", encoding="utf-8") as f:
            return pathspec.PathSpec.from_lines("gitwildmatch", f)
    except Exception:
        return None

# ---------- Flat copy with .gitignore and unique naming ----------
def should_skip_file(rel_path: Path, ext: str, config: dict, gitignore_spec) -> tuple[bool, str]:
    """Determine if a file should be skipped, and why.
    Forced inclusion paths in config["force_include"] bypass the skip checks."""
    # Forced include overrides – if any part of the relative path matches a forced include entry, do NOT skip.
    force_include = config.get("force_include", [])
    if force_include:
        for inc in force_include:
            # normalize to string without leading/trailing slashes
            inc_clean = inc.strip("/\\")
            if inc_clean and inc_clean in rel_path.parts:
                return False, ""  # never skip this path

    # 1. Extension not in whitelist
    if ext.lower() not in [e.lower() for e in config["code_extensions"]]:
        return True, f"not code ext ({ext})"

    parts = rel_path.parts

    # 2. Top-level folder skip
    if parts and parts[0] in config["skip_top_level"]:
        return True, f"in skipped folder: {parts[0]}"

    # 3. Any-depth folder skip
    if any(part in config["skip_any_depth"] for part in parts):
        return True, "in skipped subfolder"

    # 4. .gitignore pattern match (if enabled)
    if config["use_gitignore"] and gitignore_spec is not None:
        # pathspec expects a relative path with forward slashes
        rel_str = rel_path.as_posix()
        if gitignore_spec.match_file(rel_str):
            return True, "matched .gitignore"

    return False, ""

def make_unique_name(rel_path: Path, dest_dir: Path) -> str:
    """
    Return a unique filename inside dest_dir.
    - First try the original filename.
    - If that exists, use the full relative path (with -- as separator).
    - If that still exists, append a counter.
    """
    original = rel_path.name
    if not (dest_dir / original).exists():
        return original

    # Build prefixed name from full relative path (excluding the file itself)
    parent_parts = list(rel_path.parts[:-1])
    prefix = "--".join(parent_parts)  # e.g., public--assets--js
    prefixed = f"{prefix}--{original}"

    candidate = dest_dir / prefixed
    if not candidate.exists():
        return prefixed

    # Still clash – add a numeric suffix
    stem = Path(prefixed).stem
    suffix = Path(prefixed).suffix
    counter = 1
    while True:
        alt = f"{stem}_{counter}{suffix}"
        if not (dest_dir / alt).exists():
            return alt
        counter += 1

def write_csv_report(csv_path: Path, rows: list[dict]) -> None:
    """Write rows to a CSV file with headers."""
    import csv
    fieldnames = ["relative_path", "status", "reason", "size"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def copy_file_task(file_path: Path, rel: Path, dest_dir: Path, config: dict):
    """Helper for parallel copying. Returns a report row dict on success, or None on failure."""
    dest_name = make_unique_name(rel, dest_dir)
    dest_path = dest_dir / dest_name
    try:
        shutil.copy2(file_path, dest_path)
        if dest_name != rel.name:
            print(f"     {dest_name}  (from {rel.name})")
        else:
            print(f"     {dest_name}")
        return {"relative_path": str(rel), "status": "copied", "reason": "", "size": human_size(file_path.stat().st_size)}
    except Exception as e:
        print(f"     WARNING: Failed to copy {file_path}: {e}")
        return None

def copy_flat(root: Path, config: dict, args: argparse.Namespace = None) -> tuple[int, int, list[str]]:
    """Walk project, copy code files flat, return (copied, skipped, log_lines).
    Uses tqdm progress bar and parallel copying for speed."""
    dest_dir = root / config["tmp_dir"]

    if dest_dir.exists():
        print("   Removing old tmp folder...")
        shutil.rmtree(dest_dir)
    dest_dir.mkdir()

    gitignore_spec = get_gitignore_spec(root) if config["use_gitignore"] else None
    if config["use_gitignore"] and HAVE_PATHSPEC and gitignore_spec:
        print("   .gitignore patterns loaded.")
    elif config["use_gitignore"] and not HAVE_PATHSPEC:
        print("   Warning: pathspec not installed – .gitignore will be ignored.")
        print("   Install with: pip install pathspec")

    copied = 0
    report_rows: List[dict] = []
    skipped = 0
    log: List[str] = []

    tree_output = config["tree_output_file"]

    # Gather all files first for progress bar
    all_files = [p for p in root.rglob("*") if p.is_file()]
    with ThreadPoolExecutor() as executor:
        futures = []
        for file_path in tqdm(all_files, desc="Scanning files", unit="file"):
            rel = file_path.relative_to(root)
            # Skip tmp folder and tree output file
            if rel.parts and rel.parts[0] == config["tmp_dir"]:
                continue
            if rel.name == tree_output:
                continue
            ext = file_path.suffix
            skip, reason = should_skip_file(rel, ext, config, gitignore_spec)
            if skip:
                skipped += 1
                log.append(f"   {rel}  [{reason}]")
                report_rows.append({"relative_path": str(rel), "status": "skipped", "reason": reason, "size": ""})
                continue
            # Submit copy task
            futures.append(executor.submit(copy_file_task, file_path, rel, dest_dir, config))
        for future in as_completed(futures):
            result = future.result()
            if result:
                copied += 1
                report_rows.append(result)

    # Also copy the tree file into tmp/
    tree_file = root / tree_output
    if tree_file.exists():
        shutil.copy2(tree_file, dest_dir / tree_output)
        report_rows.append({"relative_path": str(tree_file), "status": "copied", "reason": "", "size": human_size(tree_file.stat().st_size)})
        copied += 1
        print(f"     {tree_output}  (project tree)")

    # Write CSV and HTML reports after processing all files
    write_csv_report(dest_dir / "files_report.csv", report_rows)
    # Generate simple HTML report
    try:
        generate_html_report(dest_dir / "files_report.csv", dest_dir / "files_report.html")
    except Exception as e:
        logging.warning(f"Failed to generate HTML report: {e}")

    return copied, skipped, log

def generate_html_report(csv_path: Path, html_path: Path) -> None:
    """Generate a minimal HTML report from the CSV file.
    The report shows a table with relative_path, status, reason, size.
    """
    import csv
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    # Simple HTML template
    html = """<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='UTF-8'>
<title>Prepare for DeepSeek – File Report</title>
<style>
  body {font-family: Arial, sans-serif; margin: 20px;}
  table {border-collapse: collapse; width: 100%;}
  th, td {border: 1px solid #ddd; padding: 8px;}
  th {background-color: #f2f2f2;}
  tr:nth-child(even) {background-color: #f9f9f9;}
</style>
</head>
<body>
<h1>File processing report</h1>
<table>
<thead>
<tr><th>Relative Path</th><th>Status</th><th>Reason</th><th>Size</th></tr>
</thead>
<tbody>
"""
    for r in rows:
        html += f"<tr><td>{r['relative_path']}</td><td>{r['status']}</td><td>{r['reason']}</td><td>{r['size']}</td></tr>\n"
    html += """</tbody>
</table>
</body>
</html>"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

def main():
    # Argument parser for CLI options
    parser = argparse.ArgumentParser(description="Prepare project for DeepSeek upload")
    parser.add_argument("--extra-extensions", type=str, default="", help="Comma‑separated list of extra file extensions to treat as code (e.g., py,txt)")
    parser.add_argument("--output-dir", type=str, default="", help="Custom directory for temporary files (overrides config tmp_dir)")
    parser.add_argument("--open-csv", action="store_true",
                        help="Open the generated CSV report after the run")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    # Set up logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    logging.basicConfig(level=numeric_level, format="[%(levelname)s] %(message)s")

    # Change to script directory
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)
    project_root = Path.cwd()

    print("=" * 55)
    print(" PREPARE FOR DEEPSEEK - All-in-one script (Python)")
    print(" Step 1: Generate FILE_STRUCTURE.md (project tree)")
    print(" Step 2: Copy code files to tmp/ (flat)")
    print("=" * 55)
    print()

    # Load configuration (from file or defaults) with CLI overrides
    config = load_config(args)
    # Apply output-dir override if provided
    if args.output_dir:
        # Ensure the directory exists
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        config["tmp_dir"] = str(output_path)
    else:
        # Use relative tmp_dir as before (will be joined with project_root later)
        pass

    # --- Step 1 ---
    print("[STEP 1] Generating FILE_STRUCTURE.md ...")
    print()
    generate_tree(project_root, config)
    print()
    print("[STEP 1] Done.")
    print()

    # --- Step 2 ---
    print("[STEP 2] Copying code files to tmp/ ...")
    print()
    print(f"   Scanning files from: {project_root}")
    print()
    print("   --- Copied files ---")

    copied, skipped, skip_log = copy_flat(project_root, config, args)

    print()
    print("   --- Skipped files ---")
    for line in skip_log:
        print(line)

    print()
    print("=" * 55)
    print(" ALL DONE!")
    print("-" * 55)
    print(f" FILE_STRUCTURE.md:  generated")
    print(f" Code files copied:  {copied}")
    print(f" Files skipped:      {skipped}")
    print("=" * 55)
    print()

    # Open tmp folder
    tmp_path = project_root / config["tmp_dir"]
    print("Opening tmp folder...")
    try:
        if platform.system() == "Windows":
            os.startfile(tmp_path)
        elif platform.system() == "Darwin":
            subprocess.run(["open", str(tmp_path)])
        else:
            subprocess.run(["xdg-open", str(tmp_path)])
    except Exception as e:
        print(f"Could not open folder automatically: {e}")
        print(f"Please open {tmp_path} manually.")

    input("Press Enter to exit...")



if __name__ == "__main__":
    main()
    