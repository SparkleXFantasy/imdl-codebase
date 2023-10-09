import mimetypes
import os
from io import BytesIO
from typing import Union
import cv2
import requests
import torch
import transformers
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

sys.path.append('/home/aya/workspace/workshop/utils')
from otter_ai import OtterForConditionalGeneration
from abspack.dataset.dataset import FolderDataset, PureDataset

# Disable warnings
requests.packages.urllib3.disable_warnings()

# ------------------- Utility Functions -------------------


def get_content_type(file_path):
    content_type, _ = mimetypes.guess_type(file_path)
    return content_type


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

# ------------------- Image Handling Functions -------------------


def get_image(url: str) -> Union[Image.Image, list]:
    if not url.strip():  # Blank input, return a blank Image
        return Image.new("RGB", (224, 224))  # Assuming 224x224 is the default size for the model. Adjust if needed.
    elif "://" not in url:  # Local file
        content_type = get_content_type(url)
    else:  # Remote URL
        content_type = requests.head(url, stream=True, verify=False).headers.get("Content-Type")

    if "image" in content_type:
        if "://" not in url:  # Local file
            return Image.open(url)
        else:  # Remote URL
            return Image.open(requests.get(url, stream=True, verify=False).raw)
    else:
        raise ValueError("Invalid content type. Expected image.")


# ------------------- OTTER Prompt and Response Functions -------------------


def get_formatted_prompt_hist(prompts: list, answers: list) -> str:
    formatted_prompt = "<image>"
    try:
        assert(len(prompts) == len(answers) + 1)
    except:
        print(f'Unmatched length of prompts and answers. Expected len(prompts) == len(answers) + 1. Got {len(prompts)}, {len(answers)}')
    for i in range(len(prompts) - 1):
        formatted_prompt += f'User: {prompts[i]} GPT:<answer> {answers[i]}<|endofchunk|>'
    formatted_prompt += f'User: {prompts[-1]} GPT:<answer>'
    return formatted_prompt
    
    
def get_formatted_prompt(prompt: str) -> str:
    return f"<image>User: {prompt} GPT:<answer>"


def get_responses(image, prompts: str, model=None, image_processor=None) -> str:
    input_data = image

    if isinstance(input_data, Image.Image):
        if input_data.size == (224, 224) and not any(input_data.getdata()):  # Check if image is blank 224x224 image
            vision_x = torch.zeros(1, 1, 1, 3, 224, 224, dtype=next(model.parameters()).dtype)
        else:
            vision_x = image_processor.preprocess([input_data], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
    else:
        vision_x = input_data
    
    model_dtype = next(model.parameters()).dtype
    vision_x = vision_x.to(dtype=model_dtype)
    
    answers = []
    
    for i in range(len(prompts)):
        
        lang_x = model.text_tokenizer(
            [
                get_formatted_prompt_hist(prompts[:i + 1], answers),
            ],
            return_tensors="pt",
        )
        
        
        lang_x_input_ids = lang_x["input_ids"]
        lang_x_attention_mask = lang_x["attention_mask"]

        generated_text = model.generate(
            vision_x=vision_x.to(model.device),
            lang_x=lang_x_input_ids.to(model.device),
            attention_mask=lang_x_attention_mask.to(model.device),
            max_new_tokens=512,
            num_beams=3,
            no_repeat_ngram_size=3,
        )
        output = model.text_tokenizer.decode(generated_text[0])
        parsed_output = (
            output
            .split("<answer>")[-1]
            .lstrip()
            .rstrip()
            .split("<|endofchunk|>")[0]
            .lstrip()
            .rstrip()
            .lstrip('"')
            .rstrip('"')
        )
        answers.append(parsed_output)
    return answers


# ------------------- Main Function -------------------

if __name__ == "__main__":
    load_bit = "bf16"
    precision = {}
    if load_bit == "bf16":
        precision["torch_dtype"] = torch.bfloat16
    elif load_bit == "fp16":
        precision["torch_dtype"] = torch.float16
    elif load_bit == "fp32":
        precision["torch_dtype"] = torch.float32
    model = OtterForConditionalGeneration.from_pretrained("/home/aya/workspace/hub/weights/gpt/OTTER-Image-MPT7B", device_map='cuda', **precision)
    model.text_tokenizer.padding_side = "left"
    tokenizer = model.text_tokenizer
    image_processor = transformers.CLIPImageProcessor()
    model.eval()
    
    trans = Compose([
        Resize((224, 224)), 
        ToTensor()
    ])
    dataset = PureDataset('/home/aya/workspace/data/imdl/train/det/fake/D_latent/D_latent_bed', transform=trans)
    l = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    real_pred = []
    fake_pred = []

    for id, x in tqdm(enumerate(l)):
        if id > 0:
            break
        data = x
        data = data.unsqueeze(1).unsqueeze(0)
        prompts_input = [
            'Is the image a real image or a synthesized image?',
            'What is in the image?'
        ]
        
        responses = get_responses(data, prompts_input, model=model)
        for response in responses:
            print(f"Response: {response}")

        # if cls.item() == 0:    
        #     if 'is a real' in response:
        #         real_pred.append(1)
        #     elif 'is a synthesize' in response:
        #         real_pred.append(0)
        #     else:
        #         with open('biggan_output.txt', 'a') as f:
        #             f.write(str(cls.item()) + ': ' + response + '\n')
        # else:
        #     if 'is a synthesize' in response:
        #         fake_pred.append(1)
        #     elif 'is a real' in response:
        #         real_pred.append(0)
        #     else:
        #         with open('output.txt', 'a') as f:
        #             f.write(str(cls.item()) + ': ' + response + '\n')
        
        
        # print(f'real: {sum(real_pred)} / {len(real_pred)}')
        # print(f'fake: {sum(fake_pred)} / {len(fake_pred)}')

    # while True:
    #     image_path = input("Enter the path to your image (or type 'quit' to exit): ")
    #     if image_path.lower() == "quit":
    #         break

    #     image = get_image(image_path)

    #     prompts_input = input("Enter the prompts (or type 'quit' to exit): ")

    #     print(f"\nPrompt: {prompts_input}")
        
