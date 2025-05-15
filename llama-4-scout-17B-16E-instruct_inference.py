# TOO BIG TO RUN ON LAB SERVERS
import os
import json
from PIL import Image
from transformers import AutoProcessor, Llama4ForConditionalGeneration, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    attn_implementation="flex_attention",
    device_map="auto",
    torch_dtype="auto"
)

image_dir = os.path.expanduser("~/git/VICE/out/images")
queries_path = os.path.expanduser("~/git/VICE/data/dev_queries.json")
output_path = "llama4-scout-17B-16E-instruct_results.json"
max_new_tokens = 256

with open(queries_path, "r") as f:
    queries = json.load(f)

results = []

for identifier, query in queries.items():
    img0_path = os.path.join(image_dir, f"{identifier}-img0.png")
    img1_path = os.path.join(image_dir, f"{identifier}-img1.png")
    if not os.path.exists(img0_path) or not os.path.exists(img1_path):
        print(f"[!] Missing images for {identifier}, skipping.")
        continue

    image0 = Image.open(img0_path).convert("RGB")
    image1 = Image.open(img1_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image0},
                {"type": "image", "image": image1},
                {"type": "text",  "text":  query},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation=messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    gen_tokens = outputs[:, inputs["input_ids"].shape[-1]:]
    response = processor.batch_decode(gen_tokens, skip_special_tokens=True)[0].strip()

    results.append((identifier, query, response))
    print(f"🧠 {identifier}\nQ: {query}\nA: {response}\n" + "-"*40)

with open(output_path, "w") as f:
    json.dump(
        {id: {"question": q, "answer": a} for id, q, a in results},
        f,
        indent=2,
    )

print(f"\n✅ Finished processing {len(results)} image pairs.")
