from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import requests
import torch
import os
from PIL import Image
from huggingface_hub import snapshot_download
import logging

class Phi4mm:
    def __init__(self):
        self.model_name = "microsoft/Phi-4-multimodal-instruct"
        self.name = 'phi4mm'
        model_path = snapshot_download("microsoft/Phi-4-multimodal-instruct")
        vision_lora_path = os.path.join(model_path, "vision-lora")
        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=True, # Required to load Phi-3.5-vision
            max_model_len=131072 ,  # Otherwise, it may not fit in smaller GPUs
            limit_mm_per_prompt={"image": 6},  # The maximum number to accept
            # enforce_eager=True,
            max_num_seqs=2,
            enable_lora=True,
            max_lora_rank=320,
            max_loras=2,
            dtype="auto",
            tensor_parallel_size=1, #current only support 1
            gpu_memory_utilization=0.98,
        )
        self.llm.llm_engine.add_lora(lora_request=LoRARequest("vision", 1, vision_lora_path))
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=2048,
            stop_token_ids=None
        )
        self.base_prompt = "You are tasked with rationalizing a unexpected outcome where each entry consists of the following components:\nImage: A single image that contains a scene.\nRationale: A plausible reasoning explaining why the situation might happend from image.\nSituation: Unexpected outcome based on the scene of the image.\nGuidelines for Output:\nThe Rationale must provide plauble reason why the outcome happend from the scene. \nUse clear and concise language.\nFormat the output for each entry as follows, enclosed in curly brackets {} to make it easy to parse:\n{Rationale: \"<rationale text>\"} {Situation: \"<situation text>\"}\n\nNow your task:\nBased on the provided Image and Situation, generate the corresponding Rationale following the structure and format above."


    def single_image_inference(self, prompt, image_dir):
        pil_image = Image.open(image_dir).resize((512,512)).convert('RGB')
        prompt = self.base_prompt + f"{{Situation: \"{prompt}\"}}"
        prompt = f"<|system|>You are a helpful assistant.<|end|><|user|><|image_1|>\n{prompt}\n<|end|><|assistant|>"
        images = [pil_image]
        outputs = self.llm.generate({
            "prompt": prompt,
            "multi_modal_data": {
                "image": images
            },
        }, sampling_params= self.sampling_params, use_tqdm= False)
        return outputs[0].outputs[0].text

    def multi_shot_inference(self, Situations: list, Rationale: list, image_dirs: list):
        assert len(Situations) == len(image_dirs) == len(Rationale)+1
        pil_images = [Image.open(image_dir).resize((512,512)).convert('RGB') for image_dir in image_dirs]
        prompt = f"<|user|>\n{self.base_prompt}\n"
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