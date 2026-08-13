"""
test_all_samples.py — Test all sample files (TXT, PDF, PNG) through the full pipeline

Runs document loading, extraction, risk assessment, and multi-doc comparison
for all sample files in sample_data/. Includes detailed timing for each step.
"""

import os
import json
import time
import logging
from pathlib import Path
from openai import OpenAI
from document_loader import load_document
from extraction import extract_clinical_data
from reasoning import assess_risk, compare_documents

# Configure logging — file + console
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "clinical_app.log"

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
))

if not root_logger.handlers:
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)
logger.info(f"Log file: {LOG_FILE.resolve()}")

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_DEFAULT_KEY = "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"

SUPPORTED_EXTENSIONS = ("*.txt", "*.pdf", "*.png", "*.jpg", "*.jpeg")


def get_nvidia_client(api_key: str):
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key, timeout=120.0)


def test_all():
    api_key = os.environ.get("NVIDIA_API_KEY", NVIDIA_DEFAULT_KEY)
    client = get_nvidia_client(api_key)

    sample_dir = Path("sample_data")

    # Gather all supported file types
    sample_files = []
    for ext in SUPPORTED_EXTENSIONS:
        sample_files.extend(sample_dir.glob(ext))
    sample_files = sorted(set(sample_files), key=lambda p: p.name)

    if not sample_files:
        print("No sample files found.")
        return

    # Group by extension for clarity
    by_ext = {}
    for sf in sample_files:
        ext = sf.suffix.lower()
        by_ext.setdefault(ext, []).append(sf)

    print(f"\n{'='*70}")
    print(f"  CLINICAL DOCUMENT INTELLIGENCE — Full Pipeline Test")
    print(f"{'='*70}")
    print(f"\n  Found {len(sample_files)} sample files:")
    for ext, files in sorted(by_ext.items()):
        print(f"    {ext}: {len(files)} file(s)")
    print()

    results = []
    timing_summary = []
    pipeline_start = time.perf_counter()

    for idx, sf in enumerate(sample_files, 1):
        print(f"\n{'─'*60}")
        print(f"  [{idx}/{len(sample_files)}] Processing: {sf.name} ({sf.suffix})")
        print(f"{'─'*60}")

        doc_start = time.perf_counter()

        # ── Step 1: Load document
        print(f"  📄 Loading document...")
        t0 = time.perf_counter()
        text = load_document(sf.read_bytes(), sf.name)
        load_elapsed = time.perf_counter() - t0
        print(f"     ⏱️  Load: {load_elapsed:.2f}s")

        if text.startswith("[ERROR]"):
            print(f"     ❌ Error: {text}")
            timing_summary.append({
                "file": sf.name,
                "load_s": load_elapsed,
                "extract_s": None,
                "risk_s": None,
                "total_s": time.perf_counter() - doc_start,
                "status": "LOAD_ERROR",
            })
            continue

        print(f"     ✅ Extracted {len(text)} chars of text")

        # ── Step 2: Extract clinical data
        print(f"  🧠 Extracting structured data...")
        t0 = time.perf_counter()
        extraction = extract_clinical_data(text, client)
        extract_elapsed = time.perf_counter() - t0
        print(f"     ⏱️  Extraction: {extract_elapsed:.2f}s")

        if not extraction:
            print(f"     ❌ Extraction failed!")
            timing_summary.append({
                "file": sf.name,
                "load_s": load_elapsed,
                "extract_s": extract_elapsed,
                "risk_s": None,
                "total_s": time.perf_counter() - doc_start,
                "status": "EXTRACT_FAILED",
            })
            continue

        print(f"     ✅ Patient: {extraction.patient_name.value}")
        print(f"        Diagnoses: {len(extraction.diagnoses)}, "
              f"Medications: {len(extraction.medications)}, "
              f"Vitals: {len(extraction.vital_signs)}")

        # ── Step 3: Risk assessment
        print(f"  ⚖️  Assessing risk...")
        t0 = time.perf_counter()
        risk = assess_risk(extraction, client)
        risk_elapsed = time.perf_counter() - t0
        print(f"     ⏱️  Risk Assessment: {risk_elapsed:.2f}s")

        if not risk:
            print(f"     ❌ Risk assessment failed!")
            timing_summary.append({
                "file": sf.name,
                "load_s": load_elapsed,
                "extract_s": extract_elapsed,
                "risk_s": risk_elapsed,
                "total_s": time.perf_counter() - doc_start,
                "status": "RISK_FAILED",
            })
            continue

        print(f"     ✅ Risk Level: {risk.risk_level}")
        print(f"        Justification: {risk.justification}")

        doc_elapsed = time.perf_counter() - doc_start
        print(f"\n     📊 Total for {sf.name}: {doc_elapsed:.2f}s")

        timing_summary.append({
            "file": sf.name,
            "load_s": load_elapsed,
            "extract_s": extract_elapsed,
            "risk_s": risk_elapsed,
            "total_s": doc_elapsed,
            "status": "SUCCESS",
        })
        results.append(extraction)

    # ── Multi-doc comparison
    compare_elapsed = 0.0
    if len(results) >= 2:
        print(f"\n{'═'*60}")
        print(f"  📊 Multi-Document Comparison ({len(results)} documents)")
        print(f"{'═'*60}")
        t0 = time.perf_counter()
        comparison = compare_documents(results, client)
        compare_elapsed = time.perf_counter() - t0
        print(f"  ⏱️  Comparison: {compare_elapsed:.2f}s")
        if comparison:
            print(f"  ✅ Overall Risk: {comparison.overall_risk_level}")
            print(f"     Summary: {comparison.summary}")
            print(f"     Trends: {len(comparison.trending_observations)}")
            print(f"     Conflicts: {len(comparison.conflicts)}")
        else:
            print("  ❌ Comparison failed!")
    else:
        print(f"\n  ℹ️  Not enough successful extractions ({len(results)}) for comparison.")

    pipeline_elapsed = time.perf_counter() - pipeline_start

    # ── Final Timing Summary
    print(f"\n{'═'*70}")
    print(f"  ⏱️  TIMING SUMMARY")
    print(f"{'═'*70}")
    print(f"  {'File':<45} {'Load':>7} {'Extract':>9} {'Risk':>7} {'Total':>8}  Status")
    print(f"  {'─'*45} {'─'*7} {'─'*9} {'─'*7} {'─'*8}  {'─'*12}")

    for t in timing_summary:
        load_s = f"{t['load_s']:.2f}s"
        extract_s = f"{t['extract_s']:.2f}s" if t['extract_s'] is not None else "  —"
        risk_s = f"{t['risk_s']:.2f}s" if t['risk_s'] is not None else "  —"
        total_s = f"{t['total_s']:.2f}s"
        print(f"  {t['file']:<45} {load_s:>7} {extract_s:>9} {risk_s:>7} {total_s:>8}  {t['status']}")

    if compare_elapsed > 0:
        print(f"  {'Multi-doc Comparison':<45} {'—':>7} {'—':>9} {'—':>7} {f'{compare_elapsed:.2f}s':>8}")

    print(f"\n  {'TOTAL PIPELINE':>45} {'':>7} {'':>9} {'':>7} {f'{pipeline_elapsed:.2f}s':>8}")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    test_all()
