import requests, json, time

BASE = "https://kidlearn-story-model.onrender.com"

print("1. Submitting story job...")
t0 = time.time()
r = requests.post(f"{BASE}/generate", json={"category": "animals", "childName": "Abebe"}, timeout=30)
data = r.json()
print(f"   Response in {time.time()-t0:.1f}s: {data}")

job_id = data.get("jobId")
if not job_id:
    print("   No jobId returned")
    exit()

print(f"\n2. Polling for result (job: {job_id})...")
for attempt in range(1, 90):
    time.sleep(3)
    try:
        r2 = requests.get(f"{BASE}/result/{job_id}", timeout=10)
        d = r2.json()
        status = d.get("status")
        elapsed = time.time() - t0
        print(f"   Poll {attempt} ({elapsed:.0f}s): {status}")
        if status != "generating":
            print(f"\nTotal time: {elapsed:.1f}s")
            print(json.dumps(d, indent=2))
            break
    except Exception as e:
        print(f"   Poll {attempt}: error - {e}")
else:
    print(f"\nTimed out after {time.time()-t0:.1f}s")
