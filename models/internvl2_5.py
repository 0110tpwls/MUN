from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image
import requests
import torch
from PIL import Image
from transformers import AutoTokenizer

class InternVL2_5:
    def __init__(self):
        self.model_name = "OpenGVLab/InternVL2_5-8B"
        self.name = "internvl2_5"
        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=True,  
            max_model_len=16384, 
            limit_mm_per_prompt={"image": 6},  
            enforce_eager=True,
            mm_processor_kwargs={"max_dynamic_patch": 4},
            dtype="bfloat16",
            tensor_parallel_size=4,
            gpu_memory_utilization=0.98,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        self.stop_token_ids = [self.tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1024,
            stop_token_ids=self.stop_token_ids
        )
        self.base_prompt = "You are tasked with rationalizing a unexpected outcome where each entry consists of the following components:\nImage: A single image that contains a scene.\nRationale: A plausible reasoning explaining why the situation might happend from image.\nSituation: Unexpected outcome based on the scene of the image.\nGuidelines for Output:\nThe Rationale must provide plauble reason why the outcome happend from the scene. \nUse clear and concise language.\nFormat the output for each entry as follows, enclosed in curly brackets {} to make it easy to parse:\n{Rationale: \"<rationale text>\"} {Situation: \"<situation text>\"}\n\nNow your task:\nBased on the provided Image and Situation, generate the corresponding Rationale following the structure and format above."
        

    def single_image_inference(self, situation, image_dir):
        pil_image = Image.open(image_dir).resize((512,512)).convert('RGB')
        prompt = self.base_prompt + f"{{Situation: \"{situation}\"}}"
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": f"<image>\n{prompt}"}], 
            tokenize=False, 
            add_generation_prompt=True
        )
        images = [pil_image]
        outputs = self.llm.generate({
            "prompt": prompt,
            "multi_modal_data": {
                "image": images
            },
        }, sampling_params= self.sampling_params, use_tqdm= False)
        return outputs[0].outputs[0].text
    
    def multi_shot_inference(self, Situations: list, Rationale: list, image_dirs: list):
        prompt = f"{self.base_prompt}\n"
        for i in range(len(image_dirs)):
            prompt += f"Image-{i+1}: <image>\n{Situations[i]}\n"
            if i < len(Rationale):
                prompt += f"{Rationale[i]}\n\n"
        
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], 
            tokenize=False, 
            add_generation_prompt=True
        )
        pil_images = [Image.open(image_dir).resize((512,512)).convert('RGB') for image_dir in image_dirs]
        outputs = self.llm.generate({
            "prompt": prompt,
            "multi_modal_data": {
                "image": pil_images
            },
        }, sampling_params= self.sampling_params, use_tqdm= False)
        return outputs[0].outputs[0].text