from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import os

processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llama3-llava-next-8b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

img_path1 = os.path.expanduser("~/git/VICE/out/images/dev-442-3-img0.png")
img_path2 = os.path.expanduser("~/git/VICE/out/images/dev-442-3-img1.png")
img1 = Image.open(img_path1).convert("RGB")
img2 = Image.open(img_path2).convert("RGB")


# Conversation 1
conversation_turn1 = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image."},
        ],
    },
]


prompt1 = processor.apply_chat_template(conversation_turn1, add_generation_prompt=True)
inputs1 = processor(images=[img1], text=prompt1, return_tensors="pt").to(model.device)

output1 = model.generate(**inputs1, max_new_tokens=100, do_sample=False)
response1_decoded = processor.decode(output1[0], skip_special_tokens=True)


assistant_marker = "assistant\n"
assistant_response_start = response1_decoded.find(assistant_marker)
if assistant_response_start != -1:
    assistant_response1 = response1_decoded[assistant_response_start + len(assistant_marker):].strip()
else:
    parts = response1_decoded.split("assistant")
    if len(parts) > 1:
        assistant_response1 = parts[-1].strip()
    else:
        assistant_response1 = " "

# Conversation 2
conversation_turn2 = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image."},
        ],
    },
]


prompt2 = processor.apply_chat_template(conversation_turn2, add_generation_prompt=True)
inputs2 = processor(images=[img2], text=prompt2, return_tensors="pt").to(model.device)

output2 = model.generate(**inputs2, max_new_tokens=100, do_sample=False)
response2_decoded = processor.decode(output2[0], skip_special_tokens=True)


assistant_marker = "assistant\n"
assistant_response_start = response2_decoded.find(assistant_marker)
if assistant_response_start != -1:
    assistant_response2 = response2_decoded[assistant_response_start + len(assistant_marker):].strip()
else:
    parts = response2_decoded.split("assistant")
    if len(parts) > 1:
        assistant_response2 = parts[-1].strip()
    else:
        assistant_response2 = " "

# Conversation 3
conversation_turn3 = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image."},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": assistant_response1}
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image."},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": assistant_response2}
        ]
    },
     {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is the difference between the two images?"},
        ],
    },
]

prompt3 = processor.apply_chat_template(conversation_turn3, add_generation_prompt=True)
inputs3 = processor(images=[img1, img2], text=prompt3, return_tensors="pt").to(model.device)

output3 = model.generate(**inputs3, max_new_tokens=100, do_sample=False)
response3_decoded = processor.decode(output3[0], skip_special_tokens=True)

assistant_response_start = response3_decoded.rfind(assistant_marker)
if assistant_response_start != -1:
    final_assistant_response = response3_decoded[assistant_response_start + len(assistant_marker):].strip()
else:
    parts = response3_decoded.split("assistant")
    if len(parts) > 1:
        final_assistant_response = parts[-1].strip()
    else:
        final_assistant_response = "Could not extract final response."

print(final_assistant_response)