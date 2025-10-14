from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image
import requests
import torch
from PIL import Image
from transformers import AutoTokenizer
import logging

class Phi3_5V:
    def __init__(self):
        self.model_name = "microsoft/Phi-3.5-vision-instruct"
        self.name = 'phi3_5v'
        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=True, # Required to load Phi-3.5-vision
            max_model_len=8192,  # Otherwise, it may not fit in smaller GPUs
            limit_mm_per_prompt={"image": 6},  # The maximum number to accept
            # enforce_eager=True,
            max_num_seqs=2,
            mm_processor_kwargs={"num_crops": 4},
            dtype="bfloat16",
            tensor_parallel_size=4,
            gpu_memory_utilization=0.98,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=2048,
            stop_token_ids=None
        )
        self.base_prompt = "You are tasked with rationalizing a unexpected outcome where each entry consists of the following components:\nImage: A single image that contains a scene.\nRationale: A plausible reasoning explaining why the situation might happend from image.\nSituation: Unexpected outcome based on the scene of the image.\nGuidelines for Output:\nThe Rationale must provide plauble reason why the outcome happend from the scene. \nUse clear and concise language.\nFormat the output for each entry as follows, enclosed in curly brackets {} to make it easy to parse:\n{Rationale: \"<rationale text>\"} {Situation: \"<situation text>\"}\n\nNow your task:\nBased on the provided Image and Situation, generate the corresponding Rationale following the structure and format above."


    def single_image_inference(self, prompt, image_dir):
        pil_image = Image.open(image_dir).resize((512,512)).convert('RGB')
        prompt = self.base_prompt + f"{{Situation: \"{prompt}\"}}"
        prompt = f"<|user|><|image_1|>\n{prompt}\n"
        images = [pil_image]
        outputs = self.llm.generate({
            "prompt": prompt,
            "multi_modal_data": {
                "image": images
            },
        }, sampling_params= self.sampling_params, use_tqdm= False)
        return outputs[0].outputs[0].text

    def multi_shot_inference(self, Situations: list, Rationale: list, image_dirs: list):
        prompt = f"<|user|>\n{self.base_prompt}\n"
        pil_images = [Image.open(image_dir).resize((512,512)).convert('RGB') for image_dir in image_dirs]
        for i in range(len(pil_images)):
            prompt += f"<|image_{i+1}|>\n{Situations[i]}\n"
            if i < len(Rationale):
                prompt += f"{Rationale[i]}\n\n"
        prompt += "<|end|>\n<|assistant|>\n"
        
        images = pil_images
        outputs = self.llm.generate({
            "prompt": prompt,
            "multi_modal_data": {
                "image": images
            },
        }, sampling_params=self.sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text