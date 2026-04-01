"""KidLearn Story Model API - async generation for Render free tier."""
import os
import uuid
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
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.eval()
print(f"Model ready! ({sum(p.numel() for p in model.parameters()):,} params)", flush=True)


def run_generation(job_id, category, child_name, child_detail):
    try:
        prompt = f"<|begin|>\nCategory: {category}\nChild: {child_name}\n\nPage 1:"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=350,
                temperature=0.8,
                top_p=0.92,
                top_k=50,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        full_text = tokenizer.decode(output[0], skip_special_tokens=False)
        clean = full_text.split("<|end|>")[0].replace("<|begin|>", "").strip()

        result = {"title": "", "pages": [], "moral": "", "category": category, "childName": child_name}
        for line in clean.split("\n"):
            line = line.strip()
            if line.startswith("Title:"):
                result["title"] = line[6:].strip()
            elif line.startswith("Page") and ":" in line:
                page_text = line.split(":", 1)[1].strip()
                if page_text:
                    result["pages"].append(page_text)
            elif line.startswith("Moral:"):
                result["moral"] = line[6:].strip()

        if not result["title"]:
            result["title"] = f"{child_name}'s {category.title()} Story"
        if not result["moral"]:
            result["moral"] = "Every day is a new adventure."

        jobs[job_id] = {"status": "done", "story": result}
    except Exception as e:
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
