#!/usr/bin/env python3
"""Create a lightweight, Claude-friendly source ZIP for AML-Sentinel.
Run from the project root: python make_share_zip.py
"""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
ZIP_PATH = ROOT.parent / "AML-Sentinel-share.zip"
SKIP_DIRS = {
    ".venv", "venv", "env", "__pycache__", ".git", ".idea", ".vscode",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints",
    "node_modules", "data", "datasets", "dataset", "checkpoints", ".ivy2",
    "spark-warehouse", "metastore_db", "logs", "tmp", "temp",
}
SKIP_NAMES = {
    ".env", ".env.local", ".env.production", "credentials.json",
    "service-account.json", "id_rsa", "id_ed25519", "derby.log",
}
SKIP_SUFFIXES = {
    ".pyc", ".pyo", ".class", ".jar", ".parquet", ".orc", ".avro",
    ".db", ".sqlite", ".sqlite3", ".log", ".zip", ".7z", ".rar",
    ".mp4", ".mov", ".avi", ".pkl", ".pickle", ".joblib",
}
KEEP_OUTPUTS = {
    "evaluation_metrics.json", "streaming_summary.json", "network_summary.json",
    "network_pattern_summary.json", "network_top_accounts.csv",
    "network_pattern_accounts.csv", "network_pattern_edges.csv", "rule_quality.csv",
    "risk_score_distribution.csv", "threshold_comparison.csv",
}
SKIP_OUTPUT_NAMES = {
    "streaming_events.jsonl", "streaming_alerts.jsonl", "scored_transactions.csv",
    "transactions.csv",
}

def should_include(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part.lower() in {d.lower() for d in SKIP_DIRS} for part in rel.parts):
        return False
    if path.name.lower() in {n.lower() for n in SKIP_NAMES | SKIP_OUTPUT_NAMES}:
        return False
    if path.suffix.lower() in SKIP_SUFFIXES:
        return False
    if any(part.lower() in {"data", "datasets", "dataset"} for part in rel.parts[:-1]):
        return False
    if rel.parts and rel.parts[0].lower() == "outputs":
        return len(rel.parts) == 2 and path.name in KEEP_OUTPUTS
    if path.suffix.lower() in {".csv", ".tsv", ".jsonl", ".ndjson"} and path.name not in KEEP_OUTPUTS:
        return False
    return path.is_file()

def main() -> None:
    if not (ROOT / "dashboard").exists():
        raise SystemExit("Run this script from the AML-Sentinel project root; 'dashboard' folder not found.")
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    added = skipped = 0
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file() or path == ZIP_PATH or path.resolve() == Path(__file__).resolve():
                continue
            if should_include(path):
                zf.write(path, Path(ROOT.name) / path.relative_to(ROOT))
                added += 1
            else:
                skipped += 1
    print(f"Created: {ZIP_PATH}")
    print(f"Files included: {added}")
    print(f"Files skipped:  {skipped}")
    print(f"ZIP size:       {ZIP_PATH.stat().st_size / (1024 * 1024):.2f} MB")
    print("Review the ZIP before sharing to ensure no private files were included.")

if __name__ == "__main__":
    main()
