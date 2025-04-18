import os
import json
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch

model_id = "llava-hf/llava-1.5-13b-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

image_dir = os.path.expanduser("~/git/VICE/out/stitched_images")
queries_path = os.path.expanduser("~/git/VICE/data/dev_queries.json")
output_path = "llava-1.5-13b-hf_results.json"
max_new_tokens = 256

with open(queries_path, "r") as f:
    queries = json.load(f)

results = []

for identifier, query in queries.items():
    stitched_path = os.path.join(image_dir, f"{identifier}-stitched.png")

    if not os.path.exists(stitched_path):
        print(f"[!] Missing stitched image: {stitched_path}")
        continue

    image = Image.open(stitched_path).convert("RGB")
    prompt = f"<image>\nUSER: {query}\nASSISTANT:"

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    raw_response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if "ASSISTANT:" in raw_response:
        response = raw_response.split("ASSISTANT:", 1)[-1].strip()
    else:
        response = raw_response
    results.append((identifier, query, response))

    print(f"🧠 {identifier}")
    print(f"Q: {query}")
    print(f"A: {response}")
    print("-" * 50)

with open(output_path, "w") as f:
    json.dump(
        {identifier: {"question": q, "answer": a} for identifier, q, a in results},
        f,
        indent=2
    )

print(f"\n✅ Finished processing {len(results)} image pairs.")
