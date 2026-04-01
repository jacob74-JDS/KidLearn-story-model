"""KidLearn Story Model API - optimized for Render free tier CPU."""
import os
import uuid
import time
import threading
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = os.environ.get("MODEL_ID", "JDS-74/kidlearn-story-model")
PORT = int(os.environ.get("PORT", 3001))

app = Flask(__name__)
jobs = {}

print(f"Loading model: {MODEL_ID}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.eval()
print(f"Model ready! ({sum(p.numel() for p in model.parameters()):,} params)", flush=True)


def run_generation(job_id, category, child_name, child_detail):
    t0 = time.time()
    try:
        prompt = f"Title: {child_name}'s {category.title()} Story\nPage 1:"
        inputs = tokenizer(prompt, return_tensors="pt")
        print(f"[{job_id}] Starting generation ({len(inputs.input_ids[0])} input tokens)...", flush=True)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=180,
                temperature=0.9,
                top_k=40,
                repetition_penalty=1.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        elapsed = time.time() - t0
        new_tokens = len(output[0]) - len(inputs.input_ids[0])
        print(f"[{job_id}] Generated {new_tokens} tokens in {elapsed:.1f}s", flush=True)

        full_text = tokenizer.decode(output[0], skip_special_tokens=True)
        pages = []
        title = f"{child_name}'s {category.title()} Story"
        moral = "Every day is a new adventure."

        for line in full_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("Title:"):
                title = line[6:].strip() or title
            elif line.startswith("Page") and ":" in line:
                page_text = line.split(":", 1)[1].strip()
                if page_text and len(page_text) > 10:
                    pages.append(page_text)
            elif line.startswith("Moral:"):
                moral = line[6:].strip() or moral

        if len(pages) < 3:
            sentences = full_text.replace("Title:", "").replace("Moral:", "")
            sentences = [s.strip() + "." for s in sentences.split(".") if len(s.strip()) > 15]
            pages = []
            for i in range(0, len(sentences), 2):
                chunk = " ".join(sentences[i:i+2])
                if chunk.strip():
                    pages.append(chunk)
            pages = pages[:6] if pages else [f"{child_name} went on a great {category} one sunny day."]

        result = {
            "title": title,
            "pages": pages[:6],
            "moral": moral,
            "category": category,
            "childName": child_name,
        }

        jobs[job_id] = {"status": "done", "story": result}
        print(f"[{job_id}] Done: {len(result['pages'])} pages, {elapsed:.1f}s total", flush=True)
    except Exception as e:
        print(f"[{job_id}] Error: {e}", flush=True)
        jobs[job_id] = {"status": "error", "error": str(e)}


@app.route("/generate", methods=["POST"])
def api_generate():
    body = request.get_json(silent=True) or {}
    category = body.get("category", "adventure")
    child_name = body.get("childName", "Child")
    child_detail = body.get("childDetail", "")

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "generating"}

    thread = threading.Thread(target=run_generation, args=(job_id, category, child_name, child_detail))
    thread.start()

    return jsonify({"jobId": job_id, "status": "generating"})


@app.route("/result/<job_id>", methods=["GET"])
def api_result(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404

    if job["status"] == "done":
        story = job.pop("story")
        del jobs[job_id]
        return jsonify({"success": True, "status": "done", "story": story})

    if job["status"] == "error":
        error = job.pop("error")
        del jobs[job_id]
        return jsonify({"success": False, "status": "error", "error": error}), 500

    return jsonify({"status": "generating"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_ID})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
