import os
import re
import random
import time
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

TRAIN_STORIES = 3000
MAX_LENGTH = 384
BATCH_SIZE = 2
GRAD_ACCUM = 4
EPOCHS = 2
LR = 5e-5
MODEL_NAME = "roneneldan/TinyStories-33M"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "kidlearn-model-output")
FINAL_DIR = os.path.join(os.path.dirname(__file__), "kidlearn-story-model-final")

CATEGORIES = ["adventure", "animals", "fantasy", "moral", "bedtime", "alphabet"]
CATEGORY_KEYWORDS = {
    "animals": ["dog", "cat", "bird", "rabbit", "fish", "bear", "lion", "elephant", "monkey", "duck", "frog", "turtle", "bunny", "puppy", "kitten"],
    "fantasy": ["magic", "fairy", "dragon", "wizard", "princess", "prince", "castle", "unicorn", "spell", "enchanted", "wand", "kingdom"],
    "adventure": ["adventure", "explore", "journey", "discover", "treasure", "map", "forest", "mountain", "brave", "quest"],
    "bedtime": ["sleep", "dream", "night", "bed", "moon", "star", "tired", "yawn", "pillow", "blanket", "quiet", "soft"],
    "moral": ["learn", "lesson", "share", "kind", "help", "friend", "sorry", "thank", "honest", "brave", "try", "important"],
}


def detect_category(text):
    text_lower = text.lower()
    scores = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        scores[cat] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else random.choice(CATEGORIES)


def extract_name(text):
    stopwords = {"The", "One", "Once", "There", "They", "She", "But", "And", "His",
                 "Her", "This", "That", "After", "Then", "When", "What", "All", "Some",
                 "Every", "For", "Not", "Day", "Now", "Little", "Big"}
    common_names = re.findall(r'\b([A-Z][a-z]{2,8})\b', text[:300])
    names = [n for n in common_names if n not in stopwords]
    return names[0] if names else random.choice(["Lily", "Tom", "Sara", "Ben", "Mia", "Sam", "Emma", "Max"])


def extract_moral(text):
    sentences = re.split(r'[.!?]+', text)
    moral_kw = ["learned", "lesson", "important", "always", "never", "remember",
                 "realized", "understood", "knew that", "from that day"]
    for sent in reversed(sentences):
        sent = sent.strip()
        if any(kw in sent.lower() for kw in moral_kw) and 15 < len(sent) < 150:
            return sent.strip()
    for sent in reversed(sentences):
        sent = sent.strip()
        if 15 < len(sent) < 120:
            return sent
    return "Every day is a new adventure."


def split_into_pages(text, num_pages=6):
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    if len(paragraphs) == 1:
        paragraphs = re.split(r'(?<=[.!?])\s+', paragraphs[0])

    if len(paragraphs) <= num_pages:
        pages = list(paragraphs)
        while len(pages) < num_pages:
            longest_idx = max(range(len(pages)), key=lambda i: len(pages[i]))
            page = pages[longest_idx]
            split_pos = page.find('. ', max(0, len(page) // 2 - 40), len(page) // 2 + 40)
            if split_pos == -1:
                split_pos = page.find('. ')
            if split_pos != -1 and split_pos > 5:
                pages[longest_idx] = page[:split_pos + 1].strip()
                pages.insert(longest_idx + 1, page[split_pos + 2:].strip())
            else:
                break
    else:
        chunk_size = max(1, len(paragraphs) // num_pages)
        pages = []
        for i in range(num_pages):
            start = i * chunk_size
            end = start + chunk_size if i < num_pages - 1 else len(paragraphs)
            pages.append(' '.join(paragraphs[start:end]))

    pages = [p for p in pages if p.strip()]
    return pages[:num_pages]


def format_storybook(text):
    text = text.strip()
    if len(text) < 100 or len(text) > 3000:
        return None

    name = extract_name(text)
    category = detect_category(text)
    title_first = text.split('.')[0].strip()
    title = title_first if len(title_first) < 60 else f"{name}'s {category.title()} Story"
    pages = split_into_pages(text)
    moral = extract_moral(text)

    if len(pages) < 3:
        return None

    formatted = f"<|begin|>\nTitle: {title}\nCategory: {category}\nChild: {name}\n\n"
    for i, page in enumerate(pages, 1):
        formatted += f"Page {i}: {page}\n\n"
    formatted += f"Moral: {moral}\n<|end|>"
    return formatted


def main():
    print("=" * 60)
    print("KidLearn Story Model Training")
    print(f"Base model: {MODEL_NAME}")
    print(f"Training stories: {TRAIN_STORIES}")
    print(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, Grad Accum: {GRAD_ACCUM}")
    print(f"Device: CPU (8 cores)")
    print("=" * 60)

    # --- Step 1: Load dataset ---
    print("\n[1/6] Loading TinyStories dataset...")
    t0 = time.time()
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    print(f"  Loaded {len(dataset):,} stories in {time.time()-t0:.0f}s")

    # --- Step 2: Format stories ---
    print(f"\n[2/6] Formatting {TRAIN_STORIES} stories into storybook structure...")
    t0 = time.time()
    indices = random.sample(range(len(dataset)), min(TRAIN_STORIES * 2, len(dataset)))
    formatted = []
    for idx in indices:
        if len(formatted) >= TRAIN_STORIES:
            break
        result = format_storybook(dataset[idx]['text'])
        if result:
            formatted.append(result)
    print(f"  Formatted {len(formatted):,} storybooks in {time.time()-t0:.0f}s")
    print(f"  Sample:\n{formatted[0][:300]}...")

    # --- Step 3: Load model ---
    print(f"\n[3/6] Loading model: {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    tokenizer.add_special_tokens({"additional_special_tokens": ["<|begin|>", "<|end|>"]})
    model.resize_token_embeddings(len(tokenizer))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded in {time.time()-t0:.0f}s")
    print(f"  Parameters: {total_params:,}")
    print(f"  Vocab size: {len(tokenizer)}")

    # --- Step 4: Tokenize ---
    print(f"\n[4/6] Tokenizing dataset...")
    t0 = time.time()
    story_dataset = Dataset.from_dict({"text": formatted})

    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = story_dataset.map(tokenize_fn, batched=True, remove_columns=["text"], desc="Tokenizing")
    split = tokenized.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"  Done in {time.time()-t0:.0f}s: {len(train_ds)} train, {len(eval_ds)} eval")

    # --- Step 5: Train ---
    total_steps = (len(train_ds) // BATCH_SIZE // GRAD_ACCUM) * EPOCHS
    est_minutes = total_steps * 3.5 / 60
    print(f"\n[5/6] Starting training...")
    print(f"  Total steps: ~{total_steps}")
    print(f"  Estimated time: ~{est_minutes:.0f} minutes on CPU")
    print(f"  (Training logs will print every 50 steps)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=0,
        use_cpu=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0
    print(f"\n  Training complete!")
    print(f"  Loss: {train_result.training_loss:.4f}")
    print(f"  Time: {elapsed/60:.1f} minutes")

    # --- Step 6: Save ---
    print(f"\n[6/6] Saving final model...")
    os.makedirs(FINAL_DIR, exist_ok=True)
    trainer.save_model(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)

    total_size = sum(
        os.path.getsize(os.path.join(FINAL_DIR, f))
        for f in os.listdir(FINAL_DIR)
        if os.path.isfile(os.path.join(FINAL_DIR, f))
    )
    print(f"  Saved to: {FINAL_DIR}")
    print(f"  Model size: {total_size / 1e6:.1f} MB")

    # --- Test generation ---
    print(f"\n{'=' * 60}")
    print("Testing story generation...")
    print("=" * 60)

    model.eval()
    for test_cat, test_name in [("adventure", "Abebe"), ("animals", "Liya"), ("bedtime", "Dawit")]:
        prompt = f"<|begin|>\nCategory: {test_cat}\nChild: {test_name}\n\nPage 1:"
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
        generated = tokenizer.decode(output[0], skip_special_tokens=False)
        text = generated.split("<|end|>")[0].replace("<|begin|>", "").strip()
        print(f"\n--- {test_cat.upper()} for {test_name} ---")
        print(text[:500])

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE!")
    print(f"Model saved at: {FINAL_DIR}")
    print("Next step: Run upload_to_hf.py to push to HuggingFace Hub")
    print("=" * 60)


if __name__ == "__main__":
    main()
