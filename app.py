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


import re
from collections import Counter

CATEGORY_PROMPTS = {
    "adventure": "went on a big adventure to find {detail}. {name} packed a small bag and walked down a path nobody had taken before. {name} really wanted to find {detail}.",
    "animals": "went to see the animals. {name} loved {detail} so much. {name} wanted to be friends with every {detail} in the world.",
    "fantasy": "discovered a magical {detail} in the garden. {name} was amazed by the {detail}. {name} opened a magical door and found a world full of wonder.",
    "moral": "was playing in the park with friends and talking about {detail}. {name} wanted to learn how to be kind and share with everyone.",
    "bedtime": "was getting ready for bed. {name} hugged a toy {detail} and pulled the blanket up tight. The moon shone through the window.",
    "alphabet": "was learning letters at school. {name} wrote the letter A for {detail}. {name} loved learning new things every day.",
}

CATEGORY_MORALS = {
    "adventure": "Being brave and curious leads to wonderful discoveries.",
    "animals": "Animals are our friends and we should always be kind to them.",
    "fantasy": "Magic happens when you believe in yourself.",
    "moral": "Being kind and honest makes the world a better place.",
    "bedtime": "Sweet dreams come to those with happy hearts.",
    "alphabet": "Learning new things every day is a great adventure.",
}


def build_prompt(category, child_name, child_detail):
    """Build a natural text prompt the model can continue from."""
    detail = child_detail if child_detail else "them"
    cat_prompt = CATEGORY_PROMPTS.get(category, "had a wonderful day.")
    cat_prompt = cat_prompt.replace("{name}", child_name).replace("{detail}", detail)

    if child_detail:
        title = f"{child_name} and the {child_detail.title()}"
    else:
        title = f"{child_name}'s {category.title()} Story"

    prompt = f"Once upon a time, there was a little child named {child_name}. One sunny day, {child_name} {cat_prompt} "

    return prompt, title


def find_most_common_name(text):
    """Find the most frequently used proper name in the text."""
    stopwords = {"The", "One", "Once", "There", "They", "She", "But", "And", "His",
                 "Her", "This", "That", "After", "Then", "When", "What", "All", "Some",
                 "Every", "For", "Not", "Day", "Now", "Little", "Big", "Mrs", "Mr",
                 "So", "He", "It", "My", "How", "Soon", "From", "Just", "As", "In",
                 "On", "At", "To", "Up", "Oh", "Yes", "No", "Was", "Is", "If"}
    names = re.findall(r'\b([A-Z][a-z]{2,10})\b', text)
    names = [n for n in names if n not in stopwords]
    if not names:
        return None
    counts = Counter(names)
    return counts.most_common(1)[0][0]


def replace_name_in_text(text, old_name, new_name):
    """Replace all occurrences of old_name with new_name, including possessives and mangles."""
    if not old_name or old_name == new_name:
        return text
    text = text.replace(f"{old_name}'s", f"{new_name}'s")
    text = text.replace(old_name, new_name)

    prefix = old_name[:3]
    words = text.split()
    result = []
    for w in words:
        stripped = w.strip(".,!?;:'\"()")
        if (stripped and stripped[0].isupper() and len(stripped) >= 3
                and stripped.startswith(prefix) and stripped != new_name
                and stripped not in {"The", "They", "Then", "That", "This", "There"}):
            result.append(w.replace(stripped, new_name))
        else:
            result.append(w)
    return " ".join(result)


def clean_text(text):
    """Fix encoding artifacts and remove training format leaks."""
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
    text = text.replace("\\n", " ").replace("\n", " ").replace("\r", " ")
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(Child:|Sign:|Category:|Title:|Moral:|Page\s*\d+:).*?(?=\.|$)', '', text)
    text = re.sub(r'<\|[^|]+\|>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def replace_all_foreign_names(text, child_name):
    """Replace every proper name in the text with the child's name."""
    stopwords = {"The", "One", "Once", "There", "They", "She", "But", "And", "His",
                 "Her", "This", "That", "After", "Then", "When", "What", "All", "Some",
                 "Every", "For", "Not", "Day", "Now", "Little", "Big", "Mrs", "Mr",
                 "So", "He", "It", "My", "How", "Soon", "From", "Just", "As", "In",
                 "On", "At", "To", "Up", "Oh", "Yes", "No", "Was", "Is", "If",
                 "Would", "Could", "Should", "Where", "Why", "Who", "Can", "Has",
                 "Had", "Did", "Does", "Will", "May", "The", "With", "Very", "Too",
                 "Sure", "Good", "Bad", "New", "Old", "Hello", "Wow", "Help", "Hi"}

    keep = {child_name, child_name + "'s"}
    words = text.split()
    result = []
    for w in words:
        stripped = w.strip(".,!?;:'\"()")
        if (stripped and stripped[0].isupper() and len(stripped) >= 3
                and stripped not in stopwords and stripped not in keep
                and not stripped.startswith(child_name)):
            result.append(w.replace(stripped, child_name))
        else:
            result.append(w)
    return " ".join(result)


def post_process_story(prompt_text, generated_text, child_name, category, child_detail):
    """Turn prompt + generated text into a personalized storybook."""
    generated_text = clean_text(generated_text)

    full_story = clean_text(prompt_text) + " " + generated_text
    full_story = replace_all_foreign_names(full_story, child_name)

    sentences = re.split(r'(?<=[.!?])\s+', full_story)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    pages = []
    buf = []
    for sent in sentences:
        buf.append(sent)
        if len(buf) >= 2:
            pages.append(" ".join(buf))
            buf = []
    if buf:
        if pages:
            pages[-1] += " " + " ".join(buf)
        else:
            pages.append(" ".join(buf))

    def is_good_page(p):
        if len(p) < 20 or len(p) > 350:
            return False
        junk_ratio = sum(1 for c in p if c in '\n\t{}[]|\\@#$%^&*~`') / max(len(p), 1)
        if junk_ratio > 0.05:
            return False
        junk_words = ["Page", "GOLD", "EUR", "USD", "http", "www", "@@"]
        if any(jw in p for jw in junk_words):
            return False
        return True

    pages = [clean_text(p) for p in pages]
    pages = [p for p in pages if is_good_page(p)][:6]

    if child_detail:
        title = f"{child_name} and the {child_detail.title()}"
    else:
        title = f"{child_name}'s {category.title()} Story"

    moral = CATEGORY_MORALS.get(category, "Every day is a new adventure.")

    if not pages:
        pages = [f"Once upon a time, {child_name} had a wonderful {category}."]

    return {"title": title, "pages": pages, "moral": moral, "category": category, "childName": child_name}


def run_generation(job_id, category, child_name, child_detail):
    t0 = time.time()
    try:
        prompt, default_title = build_prompt(category, child_name, child_detail)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_len = len(inputs.input_ids[0])
        print(f"[{job_id}] cat={category}, name={child_name}, detail={child_detail}", flush=True)
        print(f"[{job_id}] Prompt ({input_len} tokens): {repr(prompt[:150])}", flush=True)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.8,
                top_p=0.92,
                top_k=50,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        elapsed = time.time() - t0
        new_tokens = len(output[0]) - input_len

        full_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_only = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
        print(f"[{job_id}] Generated {new_tokens} tokens in {elapsed:.1f}s", flush=True)

        result = post_process_story(prompt, generated_only, child_name, category, child_detail)

        jobs[job_id] = {"status": "done", "story": result}
        print(f"[{job_id}] Done: \"{result['title']}\" ({len(result['pages'])} pages)", flush=True)
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
