import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
from PIL import Image
from typing import Tuple, List
import base64
import requests
import os
import openai
import re
import json

os.environ['OPENAI_API_KEY']= "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"
openai.api_key = os.environ["OPENAI_API_KEY"]


def fetch_clip() -> Tuple[CLIPModel, CLIPProcessor]:
    """
    CLIP model과 processor

    Returns:
        Tuple[CLIPModel, CLIPProcessor]: A tuple containing the CLIP model and processor.
    """
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    return model, processor

def extract_img_features(PIL_image: Image.Image, processor: CLIPProcessor, model: CLIPModel) -> torch.Tensor:
    """
    CLIP 모델을 활용하여 image embedding vector 추출

    Args:
        PIL_image (Image.Image): The image to process.
        processor (CLIPProcessor): The CLIP processor.
        model (CLIPModel): The CLIP model.

    Returns:
        torch.Tensor: The extracted image features.
    """
    inputs = processor(images=PIL_image, return_tensors="pt") # pytorch format
    outputs = model.get_image_features(**inputs)
    return outputs.detach()

def search_image(query_feature: torch.Tensor, features: List[torch.Tensor], topk: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    주어진 vector들과 비교하여, query_feature와 유사한 vector의 index와 유사도를 제공함

    Args:
        query_feature (torch.Tensor): query image의 embedding vector
        features (List[torch.Tensor]): 이미지들의 embedding vector
        topk (int, optional): top-k

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 유사한 이미지들의 index & cosine-distance
    """
    similarities = cosine_similarity(query_feature, torch.vstack(features)).flatten()
    # sort in descending order
    sorted_indices_desc = similarities.argsort()[::-1]
    topk_indices = sorted_indices_desc[0:topk]
    # cosine similarities for the top-k indices
    topk_similarities = similarities[topk_indices]

    return topk_indices, topk_similarities

def draw_images(images: List[Image.Image], texts: List[str]=None):
    """
    최대 5개까지 이미지 show

    Args:
        images (List[Image.Image]): 이미지 list
        texts (List[str], optional): 이미지 아래에 display 할 텍스트. Defaults to None.
    """
    k = len(images)
    if texts is None:
       texts = ['']*k
    # Set up the figure and axes for a 1x5 grid
    fig, axs = plt.subplots(1, k, figsize=(20, 4))

    for i in range(k):
        axs[i].imshow(images[i])
        axs[i].axis('off')
        axs[i].text(0.5, -0.1, texts[i], va='bottom', ha='center', fontsize=10, transform=axs[i].transAxes)

    plt.show()


def tensor2np(tensor):
    if tensor.is_cuda:
      numpy_array = tensor.cpu().numpy()
    else:
      numpy_array = tensor.numpy()

    return numpy_array

def detect_objects(img_path, model):
    img = Image.open(img_path)
    results = model(img, size=1280, augment=True)

    pred_dict = dict()
    predictions =results.pred[0]

    pred_dict['boxes'] = tensor2np(predictions[:, :4]) # x1, y1, x2, y2
    pred_dict['scores'] = tensor2np(predictions[:, 4])
    pred_dict['categories'] = tensor2np(predictions[:, 5])

    categories = results.names
    pred_dict['labels'] = [categories[i] for i in pred_dict['categories']]

    return results, pred_dict

def crop_bbox(pil_image, bbox):
    x_min, y_min, x_max, y_max = bbox
    crop_box = (x_min, y_min, x_max, y_max)

    cropped_image = pil_image.crop(crop_box)

    return cropped_image

def normalize_image(pil_image, target_size=(224, 224)):
    # resizing
    resized_image = pil_image.resize(target_size, Image.LANCZOS)

    # normalization
    np_image = np.array(resized_image).astype('float32')
    np_image /= 255.0  # pixel values to [0, 1]
    normalized_image = Image.fromarray((np_image * 255).astype('uint8'))
    return normalized_image


def filter_furniture(detections):
    furniture_class = [56, 57, 59, 60] # detections[0].names
    furniture_names = ['chair', 'couch', 'bed', 'dining table']
    furniture_detected = {}

    filter = [True if (i in furniture_names) and (s>0.5) else False for i, s in zip(detections[1]['labels'], detections[1]['scores'])]
    furniture_detected['boxes'] = detections[1]['boxes'][filter]
    furniture_detected['scores'] = detections[1]['scores'][filter]
    furniture_detected['categories'] = detections[1]['categories'][filter]
    furniture_detected['lables'] = [item for item, bool in zip(detections[1]['labels'], filter) if bool==True]
    
    return furniture_detected


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  

def describe_image(image_paths):
  """
  이미지에 대한 description을, GPT-4v를 활용하여 
  """

  text_prompt = """Please analyze the living room image provided.  
Include 'Color Scheme', 'Lighting', 'Spatial Layout', and 'Architectural Features' with descriptions based on the room's characteristics.
The output should be formatted in a JSON-like dictionary structure. Each image should be done separately.

Example output :

```json
  {
    "Color Scheme": <Description about color scheme>,
    "Lighting": <Description about lighting>,
    "Spatial Layout": <Description about spatial layouts >,
    "Architectural Features": <Descrption about architectural features>
  }
```
"""

  headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {openai.api_key}"
      }
  imgs = [encode_image(i) for i in image_paths]

  payload = {
          "model": "gpt-4-vision-preview",
          "messages": [{"role": "user",
                      "content": []
                      },
                      ],
          "max_tokens": 1000
          }
  
  img_contents = [{"type": "text", "text": text_prompt}]
  for img in imgs:
    input_template = {
      "type": "image_url",
      "image_url": {
        "url": f"data:image/jpeg;base64,{img}"
      }
    }
    img_contents.append(input_template)

  payload['messages'][0]['content'] = img_contents

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
  output = response.json()['choices'][0]['message']['content']
  return output

def parse_response(text):
    """
    GPT-4V로부터 얻은 image description을 json 형태로 변환
    """
    matches = re.findall(r'(\{[\s\S]*?\})', text)
    matches = [json.loads(m) for m in matches]
    return matches

def reformat_img_description(input_content, model='gpt-4-turbo-preview'):
    """
    parse_response를 활용하여 json의 형태로 변환이 안되는 경우, gpt-4-turbo를 활용하여 정형화된 포멧으로 변환
    """
    output_formatting_prompt = """Using the provided text, find the smallest format of json there is and store them in a list as separate elements.
The ouput list should have two json objects found from the provided text.

Desired output :
{'list': [{'Image 1': {'Color Scheme': <Color Scheme>,
    'Lighting': <Lighting>,
    'Spatial Layout': <Spatial Layout>,
    'Architectural Features': <Architectural Features>}},
  {'Image 2': {'Color Scheme': <Color Scheme>,
    'Lighting': <Lighting>,
    'Spatial Layout': <Spatial Layout>,
    'Architectural Features': <Architectural Features>}}]}

Provided text : """

    client = openai.OpenAI()

    response = client.chat.completions.create(
        model=model,
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": 'You are a smart and intelligent program that understands information and provides output in JSON format'},
            {"role": "user", "content":output_formatting_prompt + input_content}
        ]
        )
    return response