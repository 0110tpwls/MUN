from openai import OpenAI
from datasets import Dataset
import os, glob
from tqdm import tqdm
import json
import re
api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)
def generate_situation(caption, rationale):
    response = client.chat.completions.create(
        model="gpt-4o-11-20",
        messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "You are tasked with generating a dataset where each entry consists of the following components:\nCaption: A short description of an object or scene in an image.\nRationale: A plausible reasoning explaining why the object or scene might lead to an issue.\nSituation: A potential outcome based on the caption and rationale, without explicitly mentioning the cause.\nGuidelines for Output:\nThe Situation must describe the outcome without directly linking it to the rationale.\nUse clear and concise language.\nFormat the output for each entry as follows, enclosed in curly brackets {} to make it easy to parse:\n{Caption: \"<caption text>\"} {Rationale: \"<rationale text>\"} {Situation: \"<situation text>\"}\n\nExamples:\nExample 1:\n{Caption: \"A coffee maker ready to brew the perfect cup.\"} {Rationale: \"While the coffee maker looks functional, its internals are corroded, leading to potential contamination of the brewed coffee.\"} {Situation: \"A customer experienced stomach discomfort after drinking coffee brewed from the machine.\"}\nExample 2:\n{Caption: \"A sleek sports car parked in the driveway.\"} {Rationale: \"The sports car is problematic because it has an undiagnosed mechanical issue, making it dangerous to drive.\"} {Situation: \"The driver encountered a sudden loss of control while driving, leading to a minor collision.\"}\nExample 3:\n{Caption: \"A colorful toy ready for playtime.\"} {Rationale: \"This is problematic because the toy is a recall item due to safety hazards that could pose a choking risk.\"} {Situation: \"A child briefly choked while playing with the toy, requiring quick intervention.\"}\nExample 4:\n{Caption: \"A desktop computer ready for work.\"} {Rationale: \"The computer appears functional but is severely infected with malware that could compromise sensitive information.\"} {Situation: \"The user faced unauthorized access to their private accounts after using the computer for online transactions.\"}\nNow your task:\nBased on the provided Caption and Rationale, generate the corresponding Situation following the structure and format above.\n"+f"{{Caption: \"{caption}\"}}{{Rationale: \"{rationale}\"}}"
        }
            ]
        },
        ],
        response_format={
            "type": "text"
        },
        temperature=1,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content

def generate_situation_2(caption, rationale):
    response = client.chat.completions.create(
        model="gpt-4o-11-20",
        messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "You are tasked with generating a dataset where each entry consists of the following components:\nCaption: A short description of an object or scene in an image.\nRationale: A plausible reasoning explaining why the object or scene might lead to an issue.\nSituation: A potential outcome based on the caption and rationale, without explicitly mentioning the cause.\nGuidelines for Output:\nThe Situation must describe the outcome without directly linking it to the rationale.\nUse clear and concise language.\nFormat the output for each entry as follows, enclosed in curly brackets {} to make it easy to parse:\n{Caption: \"<caption text>\"} {Rationale: \"<rationale text>\"} {Situation: \"<situation text>\"}\n\nExamples:\nExample 1:\n{Caption: \"'red liquid in steak packaging'} {Rationale: \"The red liquid found in steak packaging is often mistaken for blood. It is actually a mixture of water and a protein called myoglobin that naturally occurs in muscle tissue. This liquid is perfectly normal and does not indicate that the meat is unsafe.\"} {Situation: \"Person cooked and enjoyed the steak without health issues.\"}\nExample 2:\n{Caption: \"settling of liquid in yogurt'} {Rationale: \"When you open a container of yogurt, you might observe a layer of clear liquid on top, which some may believe signifies spoilage. This liquid is simply whey separating from the yogurt solids, a natural process that doesn't affect the yogurt's quality. Stirring the whey back into the yogurt will restore its creamy texture.\"} {Situation: \"Person enjoyed the yogurt as part of their breakfast.\"}\nExample 3:\n{Caption: \"green patina on copper cookware'} {Rationale: \"Copper cookware may develop a greenish layer called patina. Some people mistake this for harmful corrosion, but patina is natural and can actually protect the copper from further oxidation. The cookware is still usable after proper cleaning.\"} {Situation: \"Person used copper cookware to prepare a delicious meal.\"}\nExample 4:\n{Caption: \"yellowing leaves on indoor plants''} {Rationale: \"Indoor plant leaves may start to turn yellow as a natural part of their growth cycle or due to minor stress factors like overwatering. A few yellow leaves do not necessarily indicate that the plant is dying.\"} {Situation: \"Person continued to care for the plant, and it grew healthy new leaves over time.\"}\nExample 5:\n{Caption: \"skin peeling after a sunburn\"} {Rationale: \"After a sunburn, the skin may start to peel. This peeling is part of the natural healing process where the body sheds damaged skin cells. While it might look alarming, it is a normal response to skin damage from ultraviolet light exposure and not a cause for concern.\"} {Situation: \"Person applied moisturizer and supported the skin's healing process comfortably.\"}\n\nNow your task:\nBased on the provided Caption and Rationale, generate the corresponding Situation following the structure and format above.\n"+f"{{Caption: \"{caption}\"}}{{Rationale: \"{rationale}\"}}"
        }
            ]
        },
        ],
        response_format={
            "type": "text"
        },
        temperature=1,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content

def generate_hllm(caption,situation ,rationale):

    response = client.chat.completions.create(
    model="gpt-4o-11-20",
    messages=[
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": f"Your task is to refine the provided explanation to make it more specific to the given context (described as the caption of the image) and to strengthen the connection between the context and the outcome. Ensure that the explanation maintains the original intention of the human annotation and does not introduce new information or change the meaning.\n\nContext: {caption}\nOutcome: {situation}\nHuman-annotated Explanation: {rationale}\n\nProvide your response in the following format. You need to add brackets!:\n{{Improved Explanation: Your improved explanation}}"
            }
        ]
        }
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
        "name": "response_format",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
            "Improved Explanation": {
                "type": "string",
                "description": "Your improved explanation"
            }
            },
            "required": [
            "Improved Explanation"
            ],
            "additionalProperties": False
        }
        }
    },
    temperature=0,
    max_completion_tokens=2048,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    store=False
    )
    return response.choices[0].message.content


def main(data_file):
    dataset = Dataset.from_file(data_file)
    updated_dataset = []
    for item in tqdm(dataset):
        caption = item["caption"]
        rationale = item["rationale"]
        situation = generate_situation(caption, rationale)
        
        # Extract situation text from response
        try:
            situation = situation.split("{Situation: ")[1].split("}")[0].strip('"')
        except:
            print(f"Failed to parse situation from response: {situation}")
            situation = ""
            
        # Add situation to item
        updated_item = dict(item)
        updated_item["situation"] = situation
        updated_dataset.append(updated_item)
    
    new_dataset = Dataset.from_list(updated_dataset)
    # Save updated dataset as Arrow file
    new_dataset.save_to_disk(data_file+"_with_siutation")

def main2(data_file):
    dataset = Dataset.from_file(data_file)
    updated_dataset = []
    for item in tqdm(dataset):
        caption = item["caption"]
        rationale = item["rationale"]
        situation = generate_situation_2(caption, rationale)
        
        # Extract situation text from response
        try:
            situation = situation.split("{Situation: ")[1].split("}")[0].strip('"')
        except:
            print(f"Failed to parse situation from response: {situation}")
            situation = ""
            
        # Add situation to item
        updated_item = dict(item)
        updated_item["situation"] = situation
        updated_dataset.append(updated_item)
    
    new_dataset = Dataset.from_list(updated_dataset)
    # Save updated dataset as Arrow file
    new_dataset.save_to_disk(data_file+"_with_siutation")


def gen_hllm(data_file):
    with open(data_file, "r") as f:
        data = json.load(f)
    new_data = []
    for item in tqdm(data):
        caption = item["caption"]
        situation = item["situation"]
        rationale = item["annotations_rationale_text"]
        improved_rationale = generate_hllm(caption, situation, rationale)
        new_item = dict(item)

        try:
            improved_rationale = improved_rationale.split("{Improved Explanation: ")[1].split("}")[0].strip('"')
        except:
            print(f"Failed to parse improved rationale from response: {improved_rationale}")
            improved_rationale = ""
        new_item["annotations_rationale_text_improved"] = improved_rationale
        new_data.append(new_item)
        # import pdb; pdb.set_trace()
    
    with open(data_file.replace(".json, _improved.json"), "w") as f:
        json.dump(new_data, f)

def gen_hllm_2(data_file):
    with open(data_file, "r") as f:
        data = json.load(f)
    new_data = []
    for item in tqdm(data):
        caption = item["caption"]
        situation = item["situation"]
        human_rationale = item["human_rationale"]
        if human_rationale == "":
            hllm_rationale = ""
        elif item['hllm_rationale'] != "":
            hllm_rationale = item['hllm_rationale']
        else:
            hllm_rationale = generate_hllm(caption, situation, human_rationale)
            try:
                hllm_rationale = json.loads(hllm_rationale)
                hllm_rationale = hllm_rationale['Improved Explanation']
            except:
                print(f"Failed to parse hllm rationale from response: {hllm_rationale}")

        new_item = dict(item)
        new_item["hllm_rationale"] = hllm_rationale
        new_data.append(new_item)
    with open(data_file.replace("full_human_rationale.json", "full_hllm_rationale.json"), "w") as f:
        json.dump(new_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    args = parser.parse_args()
    gen_hllm_2(args.data_file)

