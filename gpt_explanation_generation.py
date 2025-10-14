from openai import OpenAI
from datasets import Dataset
import os
import base64
from tqdm import tqdm
import json
import re
from PIL import Image
import pillow_avif
import argparse
import json
import random
import io
from PIL import Image
import pillow_avif

api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

def pick_rd_sample(situations, rationales, images, shot_num):
    valid_samples = False
    while not valid_samples:
        random_indices = random.sample(range(len(situations)), shot_num)
        examples_Situations = [situations[i] for i in random_indices]
        examples_Rationale = [rationales[i] for i in random_indices]
        examples_images = [images[i] for i in random_indices]
        
        # Check if randomly chosen images are valid
        valid_samples = all(
            img[0] is not None and img[1] is not None and len(img[1]) > 0 and any(ext in img[1][0].lower() for ext in ['png', 'jpg', 'jpeg', 'gif', 'webp'])
            for img in examples_images
        )
        if not valid_samples:
            print("Randomly chosen images contain None, re-picking samples.")
    
    return examples_Situations, examples_Rationale, examples_images

def img_dir_to_base64(img_dir):
    with open(img_dir, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image_dir(image_dir, img_list):
    path = image_dir
    if not os.path.exists(path):
        path = path.strip('.')
        if not os.path.exists(path):
            return None
    img_dir = path+'/'+img_list[0]
    return img_dir

def estimate_rationale_zero_shot(situation,image):
    base_prompt = "You are tasked with rationalizing a unexpected outcome where each entry consists of the following components:\nImage: A single image that contains a scene.\nRationale: A plausible reasoning explaining why the situation might happend from image.\nSituation: Unexpected outcome based on the scene of the image.\nGuidelines for Output:\nThe Rationale must provide plauble reason why the outcome happend from the scene. \nUse clear and concise language.\nFormat the output for each entry as follows, enclosed in curly brackets {} to make it easy to parse:\n{Situation: \"<situation text>\"} {Rationale: \"<rationale text>\"}\n\nNow your task:\nBased on the provided Image and Situation, generate the corresponding Rationale following the structure and format above."

    if "avif" in image:
        #load aivf image and convert to png 
        img = Image.open(image)
        img = img.convert("RGB")
        img.save(image.replace(".avif", ".png"))
        image = image.replace(".avif", ".png")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": base_prompt
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_dir_to_base64(image)}",
                        "detail": "low"
                    }
                    },
                    {
                    "type": "text",
                    "text": f"\nSituation: {{{situation}}}\n Rationale:"
                    }
                ]
                }
            ],
            response_format={
                "type": "text"
            },
            temperature=1,
            max_completion_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
    except Exception as e:
        print(f"Error processing image {image}: {e}")
        raise ValueError(f"Error processing image {image}: {e}")
    return response.choices[0].message.content

def estimate_rationale_n_shot(situations, rationales, images, shot_num):
    assert len(situations) == len(images)
    base_prompt = "You are tasked with rationalizing a unexpected outcome where each entry consists of the following components:\nImage: A single image that contains a scene.\nRationale: A plausible reasoning explaining why the situation might happend from image.\nSituation: Unexpected outcome based on the scene of the image.\nGuidelines for Output:\nThe Rationale must provide plauble reason why the outcome happend from the scene. \nUse clear and concise language.\nFormat the output for each entry as follows, enclosed in curly brackets {} to make it easy to parse:\n{Rationale: \"<rationale text>\"} {Situation: \"<situation text>\"}\n\nNow your task:\nBased on the provided Image and Situation, generate the corresponding Rationale following the structure and format above."
    
    messages = [{"role": "user", "content": [{"type": "text", "text": base_prompt}]}]

    for i in range(len(images)):
        if "avif" in images[i]:
            #load aivf image and convert to png 
            img = Image.open(images[i])
            img = img.convert("RGB")
            img.save(images[i].replace(".avif", ".png"))
            images[i] = images[i].replace(".avif", ".png")

    for i in range(shot_num):
        messages[0]["content"].extend([
            {"type": "text", "text": f"\n\n[case{i+1}]"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_dir_to_base64(images[i])}","detail": "low"}},
            {"type": "text", "text": f"\n{{{situations[i]}}} {{{rationales[i]}}}"}
        ])
    
    messages[0]["content"].extend([
        {"type": "text", "text": f"\n\n[case{shot_num+1}]"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_dir_to_base64(images[shot_num])}","detail": "low"}},
        {"type": "text", "text": f"\n{{{situations[shot_num]}}} "}
    ])
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        response_format={"type": "text"},
        temperature=1,
        max_completion_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content

def chk_if_test_set(item, data_type):
    if data_type == "mun-vis":
        test_set = test_set1
    elif data_type == "mu-lang":
        test_set = test_set2
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    for test_item in test_set:
        if item["object"] == test_item["object"] and item["situation"] == test_item["situation"] and item["img_dir"] == test_item["img_dir"] and item["caption"] == test_item["caption"]:
            return True
    return False

def main(data_type, mode, debug=False):
    dataset = Dataset.from_file(f"./{data_type}/data-00000-of-00001.arrow")
    # dataset = dataset.select(range(756, len(dataset)))
    updated_dataset = []
    situations = [item['situation'] for item in dataset]
    rationales = [item['rationale'] for item in dataset]
    images = [(item["img_dir"], item["img"]) for item in dataset]

    for item in tqdm(dataset):
        if item["img"] is None or len(item["img"]) == 0:
            continue
        assert item["img"] is not None
        image_dir = process_image_dir(item["img_dir"], item["img"])
        if image_dir is None:
            continue
        if not chk_if_test_set(item, data_type):
            continue
        situation = item["situation"]
        shots_examples = []
        if mode == "zero_shot":
            rationale1 = estimate_rationale_zero_shot(situation,image_dir)
        elif "shot" in mode:
            shot_num = int(mode.split('_')[0])
            examples_Situations, examples_Rationale, examples_images = pick_rd_sample(situations, rationales, images, shot_num)
            examples_image_dirs = [process_image_dir(examples_images[i][0], examples_images[i][1]) for i in range(shot_num) if (examples_images[i][0] is not None and examples_images[i][1] is not None and len(examples_images[i][1]) > 0)]
            examples_Situations.append(situation)
            examples_image_dirs.append(image_dir)
            shots_examples = [
                [examples_Situations[i], examples_Rationale[i], examples_image_dirs[i]]
                for i in range(shot_num)
            ]
            
            rationale1 = estimate_rationale_n_shot(examples_Situations, examples_Rationale, examples_image_dirs, shot_num)
        
        try:
            rationale = rationale1.split("{Rationale: ")[1].split("}")[0].strip('"')
        except:
            print(f"Failed to parse situation from response: {rationale}")
            rationale = rationale.strip()

        updated_item = dict(item)
        updated_item["generated_rationale"] = rationale
        updated_item["shots_examples"] = shots_examples
        updated_dataset.append(updated_item)
        if debug:
            import pdb; pdb.set_trace()
    new_dataset = Dataset.from_list(updated_dataset)
    new_dataset.save_to_disk(f"./{data_type}_gpt4o_{mode}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(data_type=args.data_type, mode=args.mode, debug=args.debug)