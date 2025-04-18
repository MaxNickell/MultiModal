import os
import json
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch

# Model config
model_id = "llava-hf/llava-interleave-qwen-7b-dpo-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Paths
image_dir = os.path.expanduser("~/git/VICE/out/images")
queries_path = os.path.expanduser("~/git/VICE/data/dev_queries.json")
output_path = "llava-interleave-qwen-7b-dpo-hf_results.json"
max_new_tokens = 256

# Load query file
with open(queries_path, "r") as f:
    queries = json.load(f)

results = []

for identifier, query in queries.items():
    img0_path = os.path.join(image_dir, f"{identifier}-img0.png")
    img1_path = os.path.join(image_dir, f"{identifier}-img1.png")

    if not os.path.exists(img0_path) or not os.path.exists(img1_path):
        print(f"[!] Missing one or both images for {identifier}")
        continue

    image0 = Image.open(img0_path).convert("RGB")
    image1 = Image.open(img1_path).convert("RGB")

    # 🔥 Prompt follows official formatting
    prompt = f"<|im_start|>user <image><image>\n{query}|im_end|><|im_start|>assistant"

    # Encode text + images
    inputs = processor(
        text=prompt,
        images=[image0, image1],
        return_tensors="pt"
    ).to(model.device)

    # Generate response
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    raw_response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if "|im_end|>assistant" in raw_response:
        response = raw_response.split("|im_end|>assistant", 1)[-1].strip()
    else:
        response = raw_response
    results.append((identifier, query, response))

    print(f"🧠 {identifier}")
    print(f"Q: {query}")
    print(f"A: {response}")
    print("-" * 50)

# Save to JSON
with open(output_path, "w") as f:
    json.dump(
        {identifier: {"question": q, "answer": a} for identifier, q, a in results},
        f,
        indent=2
    )

print(f"\n✅ Finished processing {len(results)} image pairs.")
print(f"📝 Results saved to: {output_path}")
