from vllm import LLM, SamplingParams
import requests
import torch
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

class Qwen2VL: # VLLM_USE_V1=0
    def __init__(self):
        self.model_name = "Qwen/Qwen2-VL-7B-Instruct"
        self.name = "qwen2vl"
        self.llm = LLM(
            model=self.model_name,
            # max_model_len=32768,  # Otherwise, it may not fit in smaller GPUs
            limit_mm_per_prompt={"image": 6},  # The maximum number to accept
            # enforce_eager=True,
            max_num_seqs=5,
            dtype="bfloat16",
            tensor_parallel_size=4,
            gpu_memory_utilization=0.98,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.sampling_params = SamplingParams(
            temperature=0.0,
            repetition_penalty=1.05,
            max_tokens=1024,
            stop_token_ids=None,
        )
        self.base_prompt = "You are tasked with rationalizing a unexpected outcome where each entry consists of the following components:\nImage: A single image that contains a scene.\nRationale: A plausible reasoning explaining why the situation might happend from image.\nSituation: Unexpected outcome based on the scene of the image.\nGuidelines for Output:\nThe Rationale must provide plauble reason why the outcome happend from the scene. \nUse clear and concise language.\nFormat the output for each entry as follows, enclosed in curly brackets {} to make it easy to parse:\n{Rationale: \"<rationale text>\"} {Situation: \"<situation text>\"}\n\nNow your task:\nBased on the provided Image and Situation, generate the corresponding Rationale following the structure and format above."


    def single_image_inference(self, prompt, image_dir):
        pil_image = Image.open(image_dir).resize((512,512)).convert('RGB')
        prompt = self.base_prompt + f"{{Situation: \"{prompt}\"}}"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "image", "image": pil_image},{"type": "text", "text": prompt}]}
        ]
        prompt = self.processor.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True)

        image_data, _ = process_vision_info(messages)
        outputs = self.llm.generate({
            "prompt": prompt,
            "multi_modal_data": {
                "image": image_data
            },
        }, sampling_params= self.sampling_params, use_tqdm= False)
        return outputs[0].outputs[0].text
    
    def multi_shot_inference(self, Situations: list, Rationale: list, image_dirs: list):
        assert len(Situations) == len(image_dirs) == len(Rationale)+1
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "text", "text": self.base_prompt}]}
        ]
        pil_images = [Image.open(image_dir).resize((512,512)).convert('RGB') for image_dir in image_dirs]
        for situation, rationale, image in zip(Situations, Rationale, pil_images):
            messages[1]["content"].extend([
                {"type": "image", "image": image, "min_pixels": 64*28*28, "max_pixels": 1280*28*28},
                {"type": "text", "text": f"{{Situation: \"{situation}\"}} {{Rationale: \"{rationale}\"}}\n"}
            ])

        # Add the last image and situation without rationale
        if len(Situations) > len(Rationale):
            messages[1]["content"].append({"type": "image", "image": pil_images[-1]})
            messages[1]["content"].append({"type": "text", "text": f"{{Situation: \"{Situations[-1]}\"}}"})

        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_data, _ = process_vision_info(messages)
        outputs = self.llm.generate({
            "prompt": prompt,
            "multi_modal_data": {
                "image": image_data
            },
        }, sampling_params= self.sampling_params, use_tqdm= False)
        return outputs[0].outputs[0].text