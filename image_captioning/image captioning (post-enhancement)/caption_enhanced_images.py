"""
caption_enhanced_images.py

Purpose:
This script loops through the four model folders (T5, BART, QWEN, DEEPSEEK),
reads each generated image and enhanced prompt, and generates a caption using GPT-4o-mini.
Each model's results are saved in a sorted CSV for evaluation and comparison.
"""

import os
import openai
import pandas as pd
from PIL import Image
import base64
from io import BytesIO
from dotenv import load_dotenv
from tqdm import tqdm
import time

# === STEP 1: Configuration ===
# Set environment, API key, folder paths, and filenames
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

base_folder = "C:/LocalRepo/captioner"
model_folders = ["T5", "BART", "QWEN", "DEEPSEEK"]
num_samples = 200
save_interval = 50
output_dir = os.path.join(base_folder, "model_outputs")
os.makedirs(output_dir, exist_ok=True)

# === STEP 2: Image captioning per model folder ===
# For each model folder, read image + enhanced prompt, caption it with GPT-4o-mini, and save
def encode_image(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Encoding failed: {image_path} — {e}")
        return None

def get_caption_from_api(b64_image, retries=3, delay=1.5):
    for attempt in range(retries):
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image briefly, including the subject(s) and visual details. Use one clear sentence."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                        ]
                    }
                ],
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(delay * (attempt + 1))
    return ""

def process_model_folder(model_name):
    folder_path = os.path.join(base_folder, model_name)
    png_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])[:num_samples]
    records = []

    for idx, fname in enumerate(tqdm(png_files, desc=f"Processing {model_name}", total=num_samples), start=1):
        img_id = fname.replace(".png", "")
        img_path = os.path.join(folder_path, fname)
        txt_path = os.path.join(folder_path, f"{img_id}.txt")

        if not os.path.exists(txt_path):
            print(f"[{img_id}] Missing .txt file")
            continue

        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
        except Exception as e:
            print(f"[{img_id}] Prompt read error: {e}")
            continue

        b64_image = encode_image(img_path)
        if not b64_image:
            print(f"[{img_id}] Image encoding failed")
            continue

        caption = get_caption_from_api(b64_image)
        if caption == "":
            print(f"[{img_id}] Failed after retries.")
            continue

        records.append({
            "id": int(img_id),
            "prompt": prompt,
            "image_filename": fname,
            "caption": caption
        })

        if idx % save_interval == 0 or idx == len(png_files):
            temp_df = pd.DataFrame(records)
            temp_path = os.path.join(output_dir, f"{model_name}_captions_temp.csv")
            temp_df.to_csv(temp_path, index=False)
            print(f"Auto-saved {len(records)} records to: {temp_path}")

        time.sleep(1.5)

    return pd.DataFrame(records)

# === STEP 3: Loop through all 4 model folders ===
# For each model, caption images, sort by ID, and save to final CSV
for model in model_folders:
    df_model = process_model_folder(model)
    df_model = df_model.sort_values(by="id").reset_index(drop=True)
    final_path = os.path.join(output_dir, f"{model}_captions_sorted.csv")
    df_model.to_csv(final_path, index=False)
    print(f"\nFinal sorted save: {len(df_model)} records to {final_path}")
