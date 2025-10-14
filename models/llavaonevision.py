from vllm import LLM, SamplingParams
import requests
import torch
from PIL import Image
from transformers import AutoProcessor

class LlavaOneVision: # VLLM_USE_V1=0
    def __init__(self):
        self.model_name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
        self.name = "llavaonevision"
        self.llm = LLM(
            model=self.model_name,
            # enforce_eager=True,
            max_model_len=32768,
            limit_mm_per_prompt={"image": 6},
            dtype="bfloat16",
            tensor_parallel_size=4, # 28 heads
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
        prompt =  f"<|im_start|>user <image>\n{prompt}<|im_end|> <|im_start|>assistant\n"
        images = [pil_image]
        outputs = self.llm.generate({
            "prompt": prompt,
            "multi_modal_data": {
                "image": images
            },
        }, sampling_params= self.sampling_params, use_tqdm= False)
        return outputs[0].outputs[0].text
    
    def multi_image_inference(self, pil_image1, pil_image2, pil_image3):
        # prompt = f"<|im_start|>user <image>\n{prompt1} <image>\n{prompt2} {prompt3}<|im_end|> <|im_start|>assistant\n"
        prompt = f"<|im_start|>user Here is the first image: <image>\n Here is the second image: <image> Here is the third image: <image> which of the two images is the one that contains the banana? <|im_end|> <|im_start|>assistant\n"
        # prompt = (
        #     "Here is the first image: <image>\n"
        #     "Can you describe it?\n"
        #     "Here is the second image: <image>\n"
        #     "What do you think about this one?"
        # )

        images = [pil_image1, pil_image2, pil_image3]
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
        prompt = f"<|im_start|>user {self.base_prompt}\n"
        for situation, rationale, image in zip(Situations, Rationale, pil_images):
            prompt += f"<image> {{Situation: \"{situation}\"}} {{Rationale: \"{rationale}\"}}\n"
        
        prompt += f"<image> {{Situation: \"{Situations[-1]}\"}} <|im_end|> <|im_start|>assistant\n"
        outputs = self.llm.generate({
            "prompt": prompt,
            "multi_modal_data": {
                "image": pil_images
            },
        }, sampling_params= self.sampling_params, use_tqdm= False)
        return outputs[0].outputs[0].text