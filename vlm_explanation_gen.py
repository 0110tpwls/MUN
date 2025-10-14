from models import Qwen2VL, Phi3_5V, InternVL2_5, Gemma3, LlavaOneVision, DeepSeekVL2, Qwen2_5VL, Phi4mm
from retriever import EnsembleRetriever
from datasets import Dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
import os
import re
from PIL import Image
import pillow_avif
import argparse
import json
import random
import torch
import torch.multiprocessing as mp
import logging
import pdb
# Configure logging for vllm
logging.basicConfig()
vllm_logger = logging.getLogger("vllm")
vllm_logger.setLevel(logging.WARNING)

random.seed(42)
dataset_dir = ""

base_prompt = "You are tasked with rationalizing a unexpected outcome where each entry consists of the following components:\nImage: A single image that contains a scene.\nRationale: A plausible reasoning explaining why the situation might happend from image.\nSituation: Unexpected outcome based on the scene of the image.\nGuidelines for Output:\nThe Rationale must provide plauble reason why the outcome happend from the scene. \nUse clear and concise language.\nFormat the output for each entry as follows, enclosed in curly brackets {} to make it easy to parse:\n{Rationale: \"<rationale text>\"} {Situation: \"<situation text>\"}\n\nNow your task:\nBased on the provided Image and Situation, generate the corresponding Rationale following the structure and format above."

def pick_rd_sample(dataset, shot_num):
    ret_index = []
    random_keys = random.sample(list(dataset.keys()), shot_num)
    for key in random_keys:
        ret_index.append(dataset[key]["index"])
    return ret_index

def pick_retrieval_sample(retriever, query_image, query_text, shot_num):
    results = retriever.retrieve_top_k_with_similarity(query_image, query_text, k=shot_num)
    # print(results)
    ret_index = [metadata["index"] for score, img_path, text, metadata in results]
    return ret_index

def main(model, data_type, max_shot_num=5, shot_sampling_strategy='random', output_dir='./ricl', debug=False):
    with open(os.path.join(dataset_dir, f"{data_type}_dataset_full_human_rationale.json"), "r") as f:
        dataset = json.load(f)
    dataset = {str(entry["index"]): entry for entry in dataset}
    
    for idx, item in tqdm(dataset.items()):
        image_dir = os.path.join(dataset_dir, data_type, item["image"])
        if not os.path.exists(image_dir):
            print(f"Image not found: {image_dir}")
            continue
        else:
            dataset[idx]["image"] = image_dir

    #split dataset into embedding-set and test-set
    embedding_dataset = {}
    for idx, item in tqdm(dataset.items()):
        if item["human_rationale"] == "":
            embedding_dataset[idx] = item
    test_dataset = {k: v for k, v in dataset.items() if k not in embedding_dataset}


    if shot_sampling_strategy == "retrieval" and max_shot_num > 0:
        print(f"Initializing retriever for embedding dataset")
        retriever = EnsembleRetriever(ensemble_ratio=0.5, device='cpu')
        for idx, item in tqdm(embedding_dataset.items()):
            retriever.embed_pair(item["image"], item["situation"], metadata={"index": item["index"]})
        print(f"Retriever initialized for dataset")
    
    for idx, item in tqdm(test_dataset.items()):
        image_dir = item["image"]
        situation = item["situation"]
        shot_idxs = []
        if shot_sampling_strategy == "zero":
            model_rationale = model.single_image_inference(situation, image_dir)
            try:
                rationale = model_rationale.split("{Rationale: ")[1].split("}")[0].strip('"')
            except:
                # print(f"Failed to parse situation from response: {rationale} at index {item['index']}")
                rationale = f"[RAW]{model_rationale}"
            
            shot_rationales = {0: rationale}
            index = str(item["index"])
            # print(index, type(index))
            # add rationale to item
            dataset[index]["generated_rationale"] = shot_rationales
            dataset[index]["shots_examples"] = shot_idxs
        else:
            if shot_sampling_strategy == "random":
                shot_idxs = pick_rd_sample(embedding_dataset, max_shot_num)
            elif shot_sampling_strategy == "retrieval":
                shot_idxs = pick_retrieval_sample(retriever, image_dir, situation, max_shot_num)
            else:
                raise ValueError(f"Invalid sample mode: {shot_sampling_strategy}")
            
            examples_Situations = []
            examples_Rationale = []
            examples_images = []
            # print(shot_idxs, type(shot_idxs[0]))
            for shot_idx in shot_idxs:
                shot_idx = str(shot_idx)
                examples_Situations.append(dataset[shot_idx]["situation"])
                examples_Rationale.append(dataset[shot_idx]["gpt_rationale"])
                examples_images.append(dataset[shot_idx]["image"])

            
            shot_rationales = {}
            for shot in [1,3,5]:
                shot_situations = examples_Situations[:shot]
                shot_rationale = examples_Rationale[:shot]
                shot_images = examples_images[:shot]

                shot_situations.append(situation)
                shot_images.append(image_dir)

                model_rationale = model.multi_shot_inference(shot_situations, shot_rationale, shot_images)
                
                try:
                    rationale = model_rationale.split("{Rationale: ")[1].split("}")[0].strip('"')
                except:
                    # print(f"Failed to parse situation from response: {rationale} at index {item['index']}")
                    rationale = f"[RAW]{model_rationale}"
                
                shot_rationales[shot] = rationale

            if debug:
                import pdb; pdb.set_trace()
            
            index = str(item["index"])
            # print(index, type(index))
            # add rationale to item
            dataset[index]["generated_rationale"] = shot_rationales
            dataset[index]["shots_examples"] = shot_idxs

    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, f"{data_type}_dataset_{model.name}_{shot_sampling_strategy}.json")
    output_dataset = list(dataset.values())
    output_dataset.sort(key=lambda x: x["index"])
    with open(output_dir, "w") as f:
        json.dump(output_dataset, f, indent=4)
            
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--model", type=str, default="Qwen2VL")
    parser.add_argument("--data_type", type=str, default="mun_vis") # mun_vis, mun_lang
    parser.add_argument("--shot_mode", type=int, default="5") # "0", "1", "3", "5"
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--shot_sampling_strategy", type=str, default="retrieval") # random, retrieval, zero
    args = parser.parse_args()
    
    if args.model == "Qwen2VL":
        model = Qwen2VL()
    elif args.model == "Phi3_5V":
        model = Phi3_5V()
    elif args.model == "InternVL2_5":
        model = InternVL2_5()
    elif args.model == "Gemma3":
        model = Gemma3()
    elif args.model == "LlavaOneVision":
        model = LlavaOneVision()
    elif args.model == "Qwen2_5VL":
        model = Qwen2_5VL()
    elif args.model == "Phi4mm":
        model = Phi4mm()
    else:
        raise ValueError(f"Invalid model: {args.model}")
    
    dataset_dir = args.dataset_dir

    main(model, args.data_type, int(args.shot_mode), args.shot_sampling_strategy, args.output_dir, args.debug)