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
from copy import deepcopy
import re

api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

skill_description_rubric = {
    "logical_thinking": {
        "logical_robustness": {
            "definition": "Does the model ensure general applicability and avoid logical contradictions in its reasoning steps for an instruction that requires step-by-step logical process? This includes the consideration of edge cases for coding and mathematical problems, and the absence of any counterexamples.",
            "application": "When asked to explain how to bake a cake, a logically robust response should include consistent steps in the correct order without any contradictions.",
            "score_criteria": {
                "Score 1": "The logic of the model’s response is completely incoherent.",
                "Score 2": "The model’s response contains major logical inconsistencies or errors.",
                "Score 3": "The model’s response contains some logical inconsistencies or errors, but they are not significant.",
                "Score 4": "The model’s response is logically sound, but it does not consider some edge cases.",
                "Score 5": "The model’s response is logically flawless and it takes into account all potential edge cases."
            }
        },
        "logical_correctness": {
            "definition": "Is the final answer provided by the response logically accurate and correct for an instruction that has a deterministic answer?",
            "application": "When asked what the sum of 2 and 3 is, the logically correct answer would be 5.",
            "score_criteria": {
                "Score 1": "The model’s final answer is completely incorrect and lacks sound reasoning.",
                "Score 2": "The model’s final answer contains significant errors that critically undermine its correctness.",
                "Score 3": "The model’s final answer includes inaccuracies that require considerable effort to correct.",
                "Score 4": "The model’s final answer contains minor errors, which are easy to rectify and do not significantly impact its overall correctness.",
                "Score 5": "The model’s final answer is completely accurate and sound."
            }
        },
        "logical_efficiency": {
            "definition": "Is the response logically efficient? The logic behind the response should have no redundant step, remaining simple and efficient. For tasks involving coding, the proposed solution should also consider time complexity.",
            "application": "If asked to sort a list of numbers, a model should provide a concise, step-by-step explanation without restating the obvious or using an overly complex algorithm.",
            "score_criteria": {
                "Score 1": "The logic behind the response is significantly inefficient and redundant, necessitating a complete reorganization of logic for clarity and efficiency.",
                "Score 2": "The logic of the response lacks efficiency and conciseness, requiring a substantial reorganization for better optimization.",
                "Score 3": "The logic of the response is not efficient enough, necessitating major edits for improved optimization.",
                "Score 4": "The logic of the response is largely efficient, but it still has some redundant steps. It could be handled from minor edits for better optimization.",
                "Score 5": "The logic of the response is optimally efficient, requiring no further optimization."
            }
        }
    },
    "background_knowledge": {
        "factuality": {
            "definition": "Did the model extract pertinent and accurate background knowledge without any misinformation when factual knowledge retrieval is needed? Is the response supported by reliable evidence or citation of the source of its information?",
            "application": "When asked about the boiling point of water at sea level, a factually correct response would be 100 degrees Celsius (212 Fahrenheit).",
            "score_criteria": {
                "Score 1": "The model did not extract pertinent background knowledge and provided inaccurate or misleading information. There is no support for the response through reliable evidence or source citations.",
                "Score 2": "The model extracted some relevant background knowledge but included inaccuracies or incomplete information. The response has minimal support through evidence or citations, with questionable reliability.",
                "Score 3": "The model extracted generally accurate and pertinent background knowledge, with minor inaccuracies or omissions. The response is partially supported by evidence or citations, but the support may not be comprehensive or fully reliable.",
                "Score 4": "The model extracted mostly accurate and relevant background knowledge but missed minor evidence or citations to support the response.",
                "Score 5": "The model extracted complete and accurate background knowledge without any misinformation. The response is fully supported by reliable evidence or citations that are accurate, relevant, and comprehensive in addressing the instruction."
            }
        },
        "commonsense_understanding": {
            "definition": "Is the model accurately interpreting world concepts for instructions that require a simulation of the expected result or necessitate commonsense or spatial reasoning?",
            "application": "The model should know that ice melts when exposed to heat, even if it is not explicitly mentioned.",
            "score_criteria": {
                "Score 1": "The model completely misinterprets world concepts or misunderstands commonsense knowledge.",
                "Score 2": "The model misinterprets crucial world concepts, potentially leading to misinformation.",
                "Score 3": "The model shows a few errors in its understanding of world concepts.",
                "Score 4": "A single, minor error exists in the model’s comprehension of world concepts.",
                "Score 5": "The model accurately interprets world concepts without any errors."
            }
        }
    },
    "problem_handling": {
        "comprehension": {
            "definition": "Does the response fulfill the requirements of the instruction by providing relevant information especially when the instruction is complex and includes multiple requirements? This includes responding in accordance with the explicit and implicit purpose of given instruction.",
            "application": "If asked to evaluate the pros and cons of a particular policy, a model demonstrating strong Comprehension would discuss the potential benefits and drawbacks of the policy.",
            "score_criteria": {
                "Score 1": "The response is completely unrelated to the instruction, or the model entirely misunderstands the instruction.",
                "Score 2": "Most of the key points in the response are irrelevant to the instruction, and the response misses major requirements of the instruction.",
                "Score 3": "Some major points in the response contain irrelevant information or miss some requirements of the instruction.",
                "Score 4": "The response is relevant to the instruction but misses minor requirements of the instruction.",
                "Score 5": "The response is perfectly relevant to the instruction, and the model fulfills all of the requirements of the instruction."
            }
        },
        "insightfulness": {
            "definition": "Is the response creative, original or novel, including new perspectives or interpretations of existing information?",
            "application": "When discussing potential trends in fashion, an insightful response could suggest a unique style or combination based on past trends and current preferences.",
            "score_criteria": {
                "Score 1": "The response is overly simplistic, lacking any originality or novelty.",
                "Score 2": "The ideas or perspectives within the response are commonplace, demonstrating a lack of originality or novelty.",
                "Score 3": "Some may perceive the response as original and novel, but others may find it ordinary or uninspiring.",
                "Score 4": "The response includes some innovative perspectives or ideas that require thoughtful consideration, yet they aren’t particularly surprising.",
                "Score 5": "The response is infused with surprisingly creative perspectives or ideas that are challenging to conceive, showcasing significant originality and novelty."
            }
        },
        "completeness": {
            "definition": "Does the response provide a sufficient explanation? Comprehensiveness and thoroughness of the response should be considered, which depends on the breadth of topics covered and the level of detail provided within each topic.",
            "application": "When asked to describe how photosynthesis works, a complete response should explain the process, including the roles of sunlight, water, and carbon dioxide in producing glucose and oxygen.",
            "score_criteria": {
                "Score 1": "The response doesn’t include any specifics or examples to support the statements made.",
                "Score 2": "The response does not provide sufficient details or supportive examples, requiring a major effort to make the response more complete.",
                "Score 3": "It is a decent response, but the breadth and depth of the response are rather limited. The details and examples used to substantiate the response may be insufficient.",
                "Score 4": "The response provides detailed explanations, but there is room for enhancement. The response could be further improved by including more details and supportive examples.",
                "Score 5": "The response fully provides comprehensive explanations. It delves deep into the topic, providing as much detail as possible, and it offers several examples to back up its points."
            }
        }
    }
}


template = """We would like to request your feedback on the performance of the response of the assistant to the user instruction displayed below. In the feedback, I want you to rate the quality of the response in these {skill_num} categories according to each Score rubric: 

{skill_description_rubric} 

[Instruction] 
You are tasked with rationalizing an unexpected outcome where each entry consists of the following components:\nImage: A single image that contains a scene.\nUnlikely Outcome: Unexpected outcome based on the scene of the image.\nRationale: A plausible reasoning explaining why the situation might happen from the image.\n\nGuidelines for Output: \nThe Rationale must provide a plausible reason why the outcome happened from the scene. \nUse clear and concise language.\n\nNow your task:\nBased on the provided Image and Unlikely Outcome, generate the corresponding Rationale following the structure and format above.\nImage:{image_caption}\nUnlikely Outcome: "{situation}"\nRationale:

[Ground truth Answer] 
{hllm_rationale}

[Assistant’s Response] 
{model_rationale}
[The End of Assistant’s Response] 

Please give feedback on the assistant’s responses. Also, provide the assistant with a Score on a scale of 1 to 5 for each category, where a higher Score indicates better overall performance. Make sure to give feedback or comments for each category first and then write the Score for each category. Only write the feedback corresponding to the Score rubric for each category. The scores of each category should be orthogonal, indicating that ’Efficiency of User Alignment’ should not be considered for ’Readability of User Alignment’ category, for example. 

Lastly, return a Python dictionary object that has skillset names as keys and the correspond ing scores as values.
"""

# def encode_image_for_openai(image_path, size=(512, 512), format='PNG'):
#     img = Image.open(image_path).resize(size).convert('RGB')
#     buffer = BytesIO()
#     img.save(buffer, format=format)
#     buffer.seek(0)
#     return base64.b64encode(buffer.read()).decode('utf-8')

def cal_flask_score(image_caption, situation, hllm_rationale, model_rationale, skill_name): 
    # skill_description_rubric_str = json.dumps(skill_description_rubric[skill_name])
    skill_description_rubric_str = ""
    for key in skill_description_rubric[skill_name].keys():
        skill_description_rubric_str += f"{key}\n"
        for sub_key in skill_description_rubric[skill_name][key].keys():
            # print(skill_name, key, sub_key)
            skill_description_rubric_str += f"{sub_key}: {skill_description_rubric[skill_name][key][sub_key]}\n"
    # print(skill_description_rubric_str)

    assert type(skill_description_rubric_str) == str
    prompt = template.format(skill_num=len(skill_description_rubric[skill_name]),skill_description_rubric=skill_description_rubric_str, image_caption=image_caption, situation=situation, hllm_rationale=hllm_rationale, model_rationale=model_rationale)
    response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": prompt
        }
      ]
    }
    ],
    response_format={
        "type": "text"
    },
    temperature=0,
    max_completion_tokens=2048,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    store=False
    )

    # print(response)
    raw_output= response.choices[0].message.content
    # output = re.search(r'\{.*?\}', raw_output, re.DOTALL)
    #parse by pythono dict
    try:
        output = raw_output.split("```python")[1].split("```")[0]
        # print(output)
        
        output = json.loads(output)
    except:
        copy = deepcopy(skill_description_rubric[skill_name])
        output = {}
        for key in copy.keys():
            output[key] = 0
        print(f"Failed to parse output: {raw_output}")
        import pdb; pdb.set_trace()
    assert type(output) == dict
    return output, raw_output

def make_pair(output_dir, data_type, model_name, shot_mode, shot_num=0):
    with open(os.path.join(output_dir, f"{data_type}_dataset_{model_name}_{shot_mode}.json"), "r") as f:
        mun_dataset = json.load(f)

    win_count = 0
    lose_count = 0
    results = []
    average_logical_thinking = []
    average_background_knowledge = []
    average_problem_handling = []

    for item in tqdm(mun_dataset):
        if item["human_rationale"] == "" or "generated_rationale" not in item or str(shot_num) not in item["generated_rationale"]:
            continue

        hllm_rationale = item['hllm_rationale']
        model_rationale = item['generated_rationale'][str(shot_num)]
        
        logical_thinking_output, logical_thinking_raw_output = cal_flask_score(item['caption'], item['situation'], hllm_rationale, model_rationale, "logical_thinking")
        background_knowledge_output, background_knowledge_raw_output = cal_flask_score(item['caption'], item['situation'], hllm_rationale, model_rationale, "background_knowledge")
        problem_handling_output, problem_handling_raw_output = cal_flask_score(item['caption'], item['situation'], hllm_rationale, model_rationale, "problem_handling")

        average_logical_thinking.append(logical_thinking_output)
        average_background_knowledge.append(background_knowledge_output)
        average_problem_handling.append(problem_handling_output)
        #gather all the output
        output = {
            "logical_thinking": logical_thinking_output,
            "background_knowledge": background_knowledge_output,
            "problem_handling": problem_handling_output
        }
        raw_output = {
            "logical_thinking": logical_thinking_raw_output,
            "background_knowledge": background_knowledge_raw_output,
            "problem_handling": problem_handling_raw_output
        }

        result = {
            "dataset_index": item["index"],
            "hllm_rationale": hllm_rationale,
            "model_rationale": model_rationale,
            "flask_output": output,
            "flask_raw_output": raw_output,
        }
        results.append(result)

        # import pdb; pdb.set_trace()
    
    #calculate the average score for each category per key
    def cal_average_score_per_key(list_of_dict):
        keys = list_of_dict[0].keys()
        mean_dict = {key: np.mean([d[key] for d in list_of_dict]) for key in keys}
        return mean_dict

    average_score = {
        "logical_thinking": cal_average_score_per_key(average_logical_thinking),
        "background_knowledge": cal_average_score_per_key(average_background_knowledge),
        "problem_handling": cal_average_score_per_key(average_problem_handling)
    }
    
    common_data = {
        "data_type": data_type,
        "model_name": model_name,
        "shot_mode": shot_mode,
        "shot_num": shot_num,
        "average_score": average_score
    }
    
    output_short = common_data.copy()
    output_long = {**common_data, "results": results}
    return output_short, output_long

def run_all_combinations(model_output_dir):
    data_types = ["mun_vis", "mun_lang"]
    sample_modes = ["random", "retrieval", "zero"] #
    model_names = ["qwen2vl", "phi3_5v", "internvl2_5", "gemma3", "llavaonevision", "qwen2_5vl", "phi4mm"] #, 
    shot_nums = [1,3,5]
    
    for data_type, sample_mode, model_name, shot_num in itertools.product(data_types, sample_modes, model_names, shot_nums):
        if sample_mode == "zero" and shot_num != 1:
            continue
        elif sample_mode == "zero":
            shot_num = 0
        
        output_file = os.path.join(model_output_dir, f"{data_type}_{model_name}_{sample_mode}_{shot_num}_flask_result.json")
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
#     args.add_argument("--model_output_dir", type=str, required=True)
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


