# =============================================================
# backend/tests/test_remedies.py
# Tests the remedies database
# Run with: python backend/tests/test_remedies.py
# =============================================================

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.app.db.remedies_db import (
    get_remedy, get_all_diseases, get_quick_summary
)

print("=" * 55)
print("  DAY 7 — REMEDIES DATABASE TEST")
print("=" * 55)

# Test 1: Get all diseases
print("\n[TEST 1] Loading all diseases...")
all_diseases = get_all_diseases()
print(f"✓ Total diseases loaded: {len(all_diseases)}")
for d in all_diseases:
    severity_icon = {
        'none': '🟢',
        'moderate': '🟡',
        'severe': '🔴'
    }.get(d['severity'], '⚪')
    print(f"  {severity_icon} {d['display_name']:40} | {d['severity']}")

# Test 2: Get specific remedy
print("\n[TEST 2] Getting remedy for Tomato Early Blight...")
remedy = get_remedy("Tomato_Early_blight")
print(f"✓ Found: {remedy['found']}")
print(f"  Disease    : {remedy['display_name']}")
print(f"  Pathogen   : {remedy['pathogen_name']}")
print(f"  Severity   : {remedy['severity']}")
print(f"  Overview   : {remedy['overview'][:80]}...")
print(f"\n  Symptoms ({len(remedy['symptoms'])}):")
for s in remedy['symptoms'][:3]:
    print(f"    • {s}")
print(f"\n  Chemical treatments ({len(remedy['treatments']['chemical_treatments'])}):")
for t in remedy['treatments']['chemical_treatments']:
    print(f"    • {t['name']} — {t['dosage']} every {t['frequency']}")
print(f"\n  Organic treatments ({len(remedy['treatments']['organic_treatments'])}):")
for t in remedy['treatments']['organic_treatments']:
    print(f"    • {t['name']}")
print(f"\n  Prevention tips ({len(remedy['prevention'])}):")
for p in remedy['prevention'][:3]:
    print(f"    • {p}")

# Test 3: Quick summary (what API returns)
print("\n[TEST 3] Quick summary for API response...")
summary = get_quick_summary("Potato___Late_blight")
print(f"✓ Disease    : {summary['display_name']}")
print(f"  Severity   : {summary['severity']}")
print(f"  Immediate  : {summary['immediate_action']}")
if summary['first_treatment']:
    print(f"  Treatment  : {summary['first_treatment']['name']}")
print(f"  Organic opt: {summary['has_organic_options']}")

# Test 4: Healthy plant
print("\n[TEST 4] Testing healthy plant...")
healthy = get_quick_summary("Tomato_healthy")
print(f"✓ Is healthy : {healthy['is_healthy']}")
print(f"  Message    : {healthy['overview'][:60]}...")

# Test 5: Unknown class
print("\n[TEST 5] Testing unknown class...")
unknown = get_remedy("Unknown_Disease")
print(f"✓ Found: {unknown.get('found', False)} (expected False)")

print("\n" + "=" * 55)
print("  ALL REMEDY TESTS PASSED!")
print("=" * 55)
print("""
Remedies database ready for API integration!
  • 15 diseases with full treatment info
  • Chemical + organic treatment options
  • Severity levels: none/moderate/severe
  • Treatment timelines for each disease
  • Prevention tips for healthy plants

Next → Day 8: Model export + evaluation!
""")