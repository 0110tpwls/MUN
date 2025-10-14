from openai import OpenAI
import json
import numpy as np
import argparse
from tqdm import tqdm
import itertools
import os
import base64
from PIL import Image
from io import BytesIO
import openai

api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

def encode_image_for_openai(image_path, size=(512, 512), format='PNG'):
    img = Image.open(image_path).resize(size).convert('RGB')
    buffer = BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def compare_two_models(image_dir, situation, model_a_response, model_b_response): 
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "I want you to create a leaderboard of different large-language models. To do so, I will give you the instructions (prompts) given to the models and the responses of two models. Please rank the models based on which responses would be preferred by humans. All inputs and outputs should be Python dictionaries.\\nHere is the prompt:\n\n[Start of Prompt]\nYou are tasked with rationalizing an unexpected outcome where each entry consists of the following components:\nImage: A single image that contains a scene.\nUnlikely Outcome: Unexpected outcome based on the scene of the image.\nRationale: A plausible reasoning explaining why the situation might happen from the image.\n\nGuidelines for Output: \nThe Rationale must provide a plausible reason why the outcome happened from the scene. \nUse clear and concise language.\n\nNow your task:\nBased on the provided Image and Unlikely Outcome, generate the corresponding Rationale following the structure and format above.\nImage:"
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encode_image_for_openai(image_dir)}"
            }
            },
            {
            "type": "text",
            "text": f"""Unlikely Outcome: "{situation}"
Rationale:
[End of Prompt]

Here are the outputs for the models:
{{
"model": "model_A",
"answer": "{model_a_response}"
}}
{{
"model": "model_B",
"answer": "{model_b_response}"
}}

Now, please rank the models by the quality of their answers so that the model with rank 1 has the best output. Then return a list of the model names and ranks, i.e., produce the following output:"""
            }
        ]
        }
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
        "name": "rank_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
            "model_a": {
                "type": "object",
                "properties": {
                "rank": {
                    "type": "string",
                    "enum": [
                    "rank 1",
                    "rank 2"
                    ],
                    "description": "Rank assigned by Model A."
                },
                "score": {
                    "type": "number",
                    "description": "Confidence score of Model A's ranking."
                }
                },
                "required": [
                "rank",
                "score"
                ],
                "additionalProperties": False
            },
            "model_b": {
                "type": "object",
                "properties": {
                "rank": {
                    "type": "string",
                    "enum": [
                    "rank 1",
                    "rank 2"
                    ],
                    "description": "Rank assigned by Model B."
                },
                "score": {
                    "type": "number",
                    "description": "Confidence score of Model B's ranking."
                }
                },
                "required": [
                "rank",
                "score"
                ],
                "additionalProperties": False
            }
            },
            "required": [
            "model_a",
            "model_b"
            ],
            "additionalProperties": False
        }
        }
    },
    temperature=0,
    max_completion_tokens=64,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    store=False
    )

    # print(response)
    output= response.choices[0].message.content
    output = json.loads(output)
    assert type(output) == dict
    return output

def make_pair(output_dir, data_type, model_name, shot_mode, shot_num=0):
    with open(os.path.join(output_dir, f"{data_type}_dataset_{model_name}_{shot_mode}.json"), "r") as f:
        mun_dataset = json.load(f)

    win_count = 0
    lose_count = 0
    results = []
    average_confidence_model = []
    average_confidence_hllm = []

    for item in tqdm(mun_dataset):
        if item["human_rationale"] == "" or "generated_rationale" not in item or str(shot_num) not in item["generated_rationale"]:
            continue

        hllm_rationale = item['hllm_rationale']
        model_rationale = item['generated_rationale'][str(shot_num)]
        
        rank_output = compare_two_models(item['image'], item['situation'], model_a_response=hllm_rationale, model_b_response=model_rationale)
        if rank_output['model_a']['rank'] == 'rank 2':
            win_count += 1
        else:
            lose_count += 1
        average_confidence_model.append(rank_output['model_b']['score'])
        average_confidence_hllm.append(rank_output['model_a']['score'])

        result = {
            "dataset_index": item["index"],
            "model_b_rationale": model_rationale,
            "model_a_rationale": hllm_rationale,
            "rank_output": rank_output,
            "who_won": model_name if rank_output['model_b']['rank'] == 'rank 1' else "hllm"
        }
        results.append(result)

        # import pdb; pdb.set_trace()
    
    if win_count + lose_count == 0:
        print(f"No results for {data_type}_{model_name}_{shot_mode}_{shot_num}")
        return None
    
    common_data = {
        "data_type": data_type,
        "model_name": model_name,
        "shot_mode": shot_mode,
        "shot_num": shot_num,
        "model_b": model_name,
        "model_a": "hllm",
        "win_count": win_count,
        "lose_count": lose_count,
        "win_rate": win_count / (win_count + lose_count),
        "average_confidence_model": np.mean(average_confidence_model),
        "average_confidence_hllm": np.mean(average_confidence_hllm),
    }
    
    output_short = common_data.copy()
    output_long = {**common_data, "results": results}
    return output_short, output_long

def run_all_combinations(model_output_dir):
    data_types = ["mun_vis", "mun_lang"]
    sample_modes = ["random", "retrieval", "zero"] #
    model_names = ["qwen2vl", "phi3_5v", "internvl2_5", "deepseekvl2", "gemma3", "llavaonevision", "qwen2_5vl", "phi4mm"] #, 
    shot_nums = [1,3,5]
    
    for data_type, sample_mode, model_name, shot_num in itertools.product(data_types, sample_modes, model_names, shot_nums):
        if sample_mode == "zero" and shot_num != 1:
            continue
        elif sample_mode == "zero":
            shot_num = 0
        
        output_file = os.path.join(model_output_dir, f"{data_type}_{model_name}_{sample_mode}_{shot_num}_eval_result.json")
        # Check if the output file already exists
        if os.path.exists(output_file):
            print(f"Skipping {output_file} as it already exists.")
            continue
        
        score, output = make_pair(model_output_dir, data_type, model_name, sample_mode, shot_num)
        if output is None:
            continue
        with open(output_file, "w") as f:
            json.dump(output, f, indent=4)
        
        # Print the output except the results
        print(score)

# if __name__ == "__main__":
#     args = argparse.ArgumentParser()
#     args.add_argument("--model_output_dir", type=str)
#     args = args.parse_args()
#     run_all_combinations(args.model_output_dir)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_output_dir", type=str, required=True)
    args.add_argument("--data_type", type=str, choices=["mun_vis", "mun_lang"], default="mun_vis")
    args.add_argument("--sample_mode", type=str, choices=["random", "retrieval", "zero"], default="retrieval")
    args.add_argument("--model_name", type=str, choices=["qwen2vl", "phi3_5v", "internvl2_5", "gemma3", "llavaonevision", "qwen2_5vl", "phi4mm"], default="phi3_5v")
    args.add_argument("--mode", type=str,choices=["1_shot", "3_shot", "5_shot"], default="1_shot")
    args = args.parse_args()
    mode = args.mode.split("_")[0]
    if args.sample_mode == "zero":
        mode = 0
    output = make_pair(args.data_type, args.sample_mode, args.model_name, mode)
    with open(os.path.join(args.model_output_dir, f"{args.data_type}_{args.model_name}_{args.sample_mode}_{args.mode}_eval_result.json"), "w") as f:
        json.dump(output, f, indent=4)
    #print the output except the results
    output.pop("results")
    print(output)


