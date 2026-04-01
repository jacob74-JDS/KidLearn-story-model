"""KidLearn Story Model API - deploy on Render or run locally."""
import os
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = os.environ.get("MODEL_ID", "JDS-74/kidlearn-story-model")
PORT = int(os.environ.get("PORT", 3001))

app = Flask(__name__)

print(f"Loading model: {MODEL_ID}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.eval()
print(f"Model ready! ({sum(p.numel() for p in model.parameters()):,} params)", flush=True)


def generate_story(category, child_name, child_detail=""):
    prompt = f"<|begin|>\nCategory: {category}\nChild: {child_name}\n\nPage 1:"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=450,
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

    return result


@app.route("/generate", methods=["POST"])
def api_generate():
    body = request.get_json(silent=True) or {}
    category = body.get("category", "adventure")
    child_name = body.get("childName", "Child")
    child_detail = body.get("childDetail", "")

    try:
        story = generate_story(category, child_name, child_detail)
        return jsonify({"success": True, "story": story})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_ID})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
