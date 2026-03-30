"""
test_api.py
============
Tests all API endpoints of the ARI Fusion backend
Run this while app.py is running in another terminal

Usage:
    python test_api.py
"""

import requests
import json
import os

BASE_URL = "http://localhost:5000"

# Use any real image from your dataset for CNN test
TEST_IMAGE = r"C:\Users\KIIT0001\college\minor project\github_setup\crop-analysis-disease-prediction\datasets\images\plant_disease\Tomato___Early_blight\0012b9d2-2130-4a06-a834-b1f3af34f57e___RS_Erly.B 8389.JPG"

print("=" * 60)
print("  ARI FUSION API — ENDPOINT TESTS")
print("=" * 60)

passed = 0
failed = 0

# ─────────────────────────────────────────────
# TEST 1: Health Check
# ─────────────────────────────────────────────
print("\n[TEST 1] GET /health")
try:
    res = requests.get(f"{BASE_URL}/health")
    if res.status_code == 200 and res.json()["status"] == "running":
        print("  ✅ PASSED — Server is running")
        passed += 1
    else:
        print(f"  ❌ FAILED — Status: {res.status_code}")
        failed += 1
except Exception as e:
    print(f"  ❌ FAILED — {e}")
    failed += 1


# ─────────────────────────────────────────────
# TEST 2: ML Only
# ─────────────────────────────────────────────
print("\n[TEST 2] POST /predict/ml")
try:
    payload = {
        "n": 90, "p": 42, "k": 43,
        "temperature": 20.8, "humidity": 82,
        "ph": 6.5, "rainfall": 202
    }
    res = requests.post(f"{BASE_URL}/predict/ml", json=payload)
    data = res.json()
    if res.status_code == 200 and data["success"]:
        print(f"  ✅ PASSED")
        print(f"     Recommended crop : {data['recommended_crop']}")
        print(f"     Confidence       : {data['confidence']*100:.1f}%")
        print(f"     Top 3 crops      : {[c['crop'] for c in data['top3_crops']]}")
        passed += 1
    else:
        print(f"  ❌ FAILED — {data}")
        failed += 1
except Exception as e:
    print(f"  ❌ FAILED — {e}")
    failed += 1


# ─────────────────────────────────────────────
# TEST 3: NLP Only
# ─────────────────────────────────────────────
print("\n[TEST 3] POST /predict/nlp")
try:
    payload = {
        "text": "leaves are turning yellow with brown spots and the plant looks very weak and wilting"
    }
    res  = requests.post(f"{BASE_URL}/predict/nlp", json=payload)
    data = res.json()
    if res.status_code == 200 and data["success"]:
        print(f"  ✅ PASSED")
        print(f"     Prediction       : {data['prediction']}")
        print(f"     Disease prob     : {data['disease_probability']*100:.1f}%")
        passed += 1
    else:
        print(f"  ❌ FAILED — {data}")
        failed += 1
except Exception as e:
    print(f"  ❌ FAILED — {e}")
    failed += 1


# ─────────────────────────────────────────────
# TEST 4: CNN Only
# ─────────────────────────────────────────────
print("\n[TEST 4] POST /predict/cnn")
try:
    if not os.path.exists(TEST_IMAGE):
        print(f"  ⚠️  SKIPPED — Test image not found at:\n     {TEST_IMAGE}")
        print(f"     Update TEST_IMAGE path at top of this file!")
    else:
        with open(TEST_IMAGE, "rb") as img:
            files = {"image": img}
            res   = requests.post(f"{BASE_URL}/predict/cnn", files=files)
        data = res.json()
        if res.status_code == 200 and data["success"]:
            print(f"  ✅ PASSED")
            print(f"     Predicted class  : {data['predicted_class']}")
            print(f"     Disease prob     : {data['disease_probability']*100:.1f}%")
            passed += 1
        else:
            print(f"  ❌ FAILED — {data}")
            failed += 1
except Exception as e:
    print(f"  ❌ FAILED — {e}")
    failed += 1


# ─────────────────────────────────────────────
# TEST 5: Full ARI Prediction
# ─────────────────────────────────────────────
print("\n[TEST 5] POST /predict (Full ARI)")
try:
    if not os.path.exists(TEST_IMAGE):
        print(f"  ⚠️  SKIPPED — Test image not found")
    else:
        form_data = {
            "n"          : "90",
            "p"          : "42",
            "k"          : "43",
            "temperature": "20.8",
            "humidity"   : "82",
            "ph"         : "6.5",
            "rainfall"   : "202",
            "text"       : "leaves are turning yellow with brown spots and the plant looks very weak"
        }
        with open(TEST_IMAGE, "rb") as img:
            files = {"image": img}
            res   = requests.post(f"{BASE_URL}/predict", data=form_data, files=files)

        data = res.json()
        if res.status_code == 200 and data["success"]:
            print(f"  ✅ PASSED")
            print(f"     Recommended crop : {data['ml']['recommended_crop']}")
            print(f"     NLP prediction   : {data['nlp']['prediction']}")
            print(f"     CNN class        : {data['cnn']['predicted_class']}")
            print(f"     ARI Score        : {data['fusion']['ARI']}")
            print(f"     Risk Level       : {data['fusion']['risk_level']}")
            print(f"     Advisory         : {data['fusion']['advisory']}")
            passed += 1
        else:
            print(f"  ❌ FAILED — {data}")
            failed += 1
except Exception as e:
    print(f"  ❌ FAILED — {e}")
    failed += 1


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  RESULTS: {passed} passed, {failed} failed out of 5 tests")
if failed == 0:
    print("  🎉 ALL ENDPOINTS WORKING!")
else:
    print("  ⚠️  Some endpoints need attention!")
print("=" * 60)