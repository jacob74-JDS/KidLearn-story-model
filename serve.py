"""Local inference server for the KidLearn story model."""
import json
import sys
import os
from http.server import HTTPServer, BaseHTTPRequestHandler

os.environ["PYTHONUNBUFFERED"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "JDS-74/kidlearn-story-model"
PORT = 3001

print(f"Loading model: {MODEL_ID}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print("Tokenizer loaded.", flush=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.eval()
print(f"Model loaded! ({sum(p.numel() for p in model.parameters()):,} params)", flush=True)


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
        elif line.startswith("Category:"):
            pass
        elif line.startswith("Child:"):
            pass
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


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/generate":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        category = body.get("category", "adventure")
        child_name = body.get("childName", "Child")
        child_detail = body.get("childDetail", "")

        try:
            story = generate_story(category, child_name, child_detail)
            response = json.dumps({"success": True, "story": story})
            self.send_response(200)
        except Exception as e:
            response = json.dumps({"success": False, "error": str(e)})
            self.send_response(500)

        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response.encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        print(f"[StoryModel] {args[0]}")


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Story model server running on http://localhost:{PORT}", flush=True)
    print("Ready to generate stories!", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()
