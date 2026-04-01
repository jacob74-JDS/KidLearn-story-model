import requests, json, time

BASE = "https://kidlearn-story-model.onrender.com"

print("1. Health check...")
r = requests.get(f"{BASE}/health", timeout=30)
print(f"   {r.json()}")

print("\n2. Submitting story job...")
t0 = time.time()
r = requests.post(f"{BASE}/generate", json={"category": "adventure", "childName": "Abebe"}, timeout=15)
data = r.json()
print(f"   {data}")

job_id = data.get("jobId")
if not job_id:
    print("   No jobId - server might use sync mode")
    if data.get("success"):
        print(json.dumps(data, indent=2))
    exit()

print(f"\n3. Polling for result (job: {job_id})...")
for attempt in range(1, 61):
    time.sleep(3)
    r2 = requests.get(f"{BASE}/result/{job_id}", timeout=10)
    d = r2.json()
    status = d.get("status")
    print(f"   Poll {attempt}: {status}")
    if status != "generating":
        break

elapsed = time.time() - t0
print(f"\nTotal time: {elapsed:.1f}s")
print(json.dumps(d, indent=2))
