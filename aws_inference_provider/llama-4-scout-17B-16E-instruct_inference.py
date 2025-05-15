import os, json, base64, mimetypes, pathlib, logging, time, io
from openai import OpenAI, OpenAIError
from PIL import Image


client = OpenAI(
   base_url="",
   api_key="",
)


MODEL_NAME = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
max_new_tokens = 128
streaming = False


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


dev_json = os.path.expanduser("~/git/VICE/data/dev.json")
img_dir = os.path.expanduser("~/git/VICE/out/images")
out_json = "llama4-scout-17B-16E-instruct_results.json"


MAX_EDGE = 768
JPEG_Q = 85


def img_to_data_uri(path: str) -> str | None:
   """
   Open local image, shrink longest edge to <=768 px, re-encode as JPEG 85,
   return data URI.  Keeps output well under 1 MB.
   """
   try:
       with Image.open(path) as im:
           im = im.convert("RGB")
           im.thumbnail((MAX_EDGE, MAX_EDGE))
           buf = io.BytesIO()
           im.save(buf, format="JPEG", quality=JPEG_Q, optimize=True)
           data = buf.getvalue()
       b64 = base64.b64encode(data).decode()
       return f"data:image/jpeg;base64,{b64}"
   except Exception as e:
       logging.warning(f"{path} – cannot process ({e})")
       return None


def find_pair(identifier: str) -> tuple[str | None, str | None]:
   "Return (data_uri0, data_uri1) or (None, None). Tries png then jpg."
   for stem in (identifier, "-".join(identifier.split("-")[:-1])):
       if not stem:
           continue
       for ext in (".png", ".jpg", ".jpeg"):
           p0 = os.path.join(img_dir, f"{stem}-img0{ext}")
           p1 = os.path.join(img_dir, f"{stem}-img1{ext}")
           if os.path.exists(p0) and os.path.exists(p1):
               return img_to_data_uri(p0), img_to_data_uri(p1)
   return None, None


with open(dev_json) as f:
   items = json.load(f)


results = {}
for item in items:
   identifier = item["identifier"]
   uri0, uri1 = find_pair(identifier)
   if not (uri0 and uri1):
       logging.warning(f"{identifier} – local images not found, skipping.")
       continue


   prompt = item["annotation"]["open_ended"][:512]
   messages = [{
       "role": "user",
       "content": [
           {"type": "image_url", "image_url": {"url": uri0}},
           {"type": "image_url", "image_url": {"url": uri1}},
           {"type": "text",      "text": prompt},
       ],
   }]


   for attempt in (1, 2, 3):
       try:
           chat = client.chat.completions.create(
               model=MODEL_NAME,
               messages=messages,
               max_tokens=max_new_tokens,
               stream=streaming,
           )
           if streaming:
               answer_chunks = []
               for chunk in chat:
                   token = chunk.choices[0].delta.content or ""
                   answer_chunks.append(token)
                   print(token, end="", flush=True)
               answer = "".join(answer_chunks).strip()
               print()
           else:
               answer = chat.choices[0].message.content.strip()


           results[identifier] = {"question": prompt, "answer": answer}
           logging.info(f"✓ {identifier}")
           break


       except OpenAIError as e:
           if attempt == 3:
               logging.warning(f"{identifier} skipped ({e.__class__.__name__}: {e})")
           else:
               logging.warning(f"{identifier} retry {attempt}/3 … ({e})")
               time.sleep(2 * attempt)
       except Exception as e:
           logging.warning(f"{identifier} skipped ({e})")
           break


with open(out_json, "w") as f:
   json.dump(results, f, indent=2)


print(f"\n✅ {len(results)} / {len(items)} items answered.")



