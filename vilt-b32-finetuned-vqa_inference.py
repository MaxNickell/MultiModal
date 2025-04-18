import os
import json
import torch
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering

model_id = "dandelin/vilt-b32-finetuned-vqa"
image_dir = os.path.expanduser("~/git/VICE/out/stitched_images")
queries_path = os.path.expanduser("~/git/VICE/data/dev_queries.json")
output_path = "vilt-b32-finetuned-vqa_results.json"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = ViltProcessor.from_pretrained(model_id)
model = ViltForQuestionAnswering.from_pretrained(model_id)
model.to(device)
model.eval()

with open(queries_path, "r") as f:
    queries = json.load(f)

results = []

for identifier, question in queries.items():
    stitched_path = os.path.join(image_dir, f"{identifier}-stitched.png")

    if not os.path.exists(stitched_path):
        print(f"[!] Missing stitched image: {stitched_path}")
        continue

    image = Image.open(stitched_path).convert("RGB")
    inputs = processor(image, question, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer = model.config.id2label[idx]

    results.append((identifier, question, answer))

    print(f"🧠 {identifier}")
    print(f"Q: {question}")
    print(f"A: {answer}")
    print("-" * 50)


with open(output_path, "w") as f:
    json.dump(
        {i: {"question": q, "answer": a} for i, q, a in results},
        f,
        indent=2
    )

print(f"\n✅ Finished processing {len(results)} images. Results saved to: {output_path}")
