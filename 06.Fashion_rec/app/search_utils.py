import numpy as np
from PIL import Image, ImageFilter
import pandas as pd
from llama_index.llms.openai import OpenAI
from llama_index.program.openai import OpenAIPydanticProgram
from pydantic import BaseModel
from typing import List, Literal
import torch
import io
import base64
import requests
import json
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core import SimpleDirectoryReader
import os

from image_utils import crop_bbox, extract_img_features
from yolo_utils import rescale_bboxes, plot_results, box_cxcywh_to_xyxy

############################################# Image search #############################################

def clothes_detector(image, feature_extractor, model, thresh=0.5):
    # all categories
    cats = ['shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants', 'shorts', 
            'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat', 'headband, head covering, hair accessory', 
            'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 
            'umbrella', 'hood', 'collar', 'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 
            'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel']
    # category we are interested in
    category_of_interest = ['pants', 'shirt, blouse', 'jacket', 'top, t-shirt, sweatshirt', 'dress', 'shoe', 'glasses', 
                        'skirt', 'bag, wallet', 'belt', 'headband, head covering, hair accessory', 'sock', 'hat', 
                        'watch', 'glove', 'tights, stockings', 'sweater', 'tie', 'shorts', 'scarf', 'coat', 'vest', 
                        'umbrella', 'cardigan', 'cape', 'jumpsuit', 'leg warmer']
    # yolo detection
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    
    # extract detected labels and boundingboxes
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > thresh

    prob = probas[keep]

    indices = [np.argmax(idx.detach().numpy()) for idx in prob]
    detected_cats = [cats[idx] for idx in indices]
    boxes = outputs.pred_boxes[0, keep].cpu()

    bboxes_scaled = rescale_bboxes(boxes, image.size).tolist()
    
    # keep boxes that we are interested in
    keep_indices = list()
    keep_bboxes = list()

    for idx, box in zip(detected_cats, bboxes_scaled):
        if idx in category_of_interest:
            keep_indices.append(idx)
            keep_bboxes.append(box)
    # overlapping한 구간이 있는 bbox들을 통합
    merged_bbox, merged_labels = merge_boxes(keep_bboxes, keep_indices)

    # cropping
    categories = pd.read_csv("categories.csv")
    cropped_images = dict()

    for label, box in zip(merged_labels, merged_bbox):
        cropped = resize_img(crop_bbox(image, box), categories.loc[categories['name']==label, 'supercategory'].values[0])
        cropped_images[label] = cropped

    return cropped_images


def image_search(index, sparse_vector, cropped_images, model, processor, top_k=10):
    results = dict()

    for label, image in cropped_images.items():
        img_emb = extract_img_features(image, processor, model).tolist()

        result = index.query(
            vector=img_emb[0],
            top_k=top_k,
            sparse_vector=sparse_vector,
            filter={"category": {"$eq": label}},
            include_metadata=True
        )

        results[label] = result
    return results

def resize_img(image, category):
    standard_size = {"lowerbody":[420, 540],
        "upperbody":[500, 700],
        "wholebody":[480, 880],
        "legs and feet":[100, 150],
        "head":[150, 100],
        "others":[200, 350],
        "waist":[200, 100],
        "arms and hands":[75, 75],
        "neck":[120, 200]}
    
    w, h = image.size
    img_size = w*h

    new_width, new_height = standard_size[category]
    new_size = new_width * new_height

    if img_size >= new_size:
        # For downsizing
        downsized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return downsized_image
    else:
        # For upsizing
        upsized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        upsized_image = upsized_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        return upsized_image


def iou(boxA, boxB):
    # Calculate the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def merge_boxes(boxes, labels):
    merged_boxes = []
    merged_labels = []
    used = set()

    for i in range(len(boxes)):
        if i in used:
            continue
        current_box = boxes[i]
        for j in range(i + 1, len(boxes)):
            if j in used or labels[i] != labels[j]:
                continue
            if iou(current_box, boxes[j]) > 0.5:  # Assuming a positive IoU indicates overlap
                # For xyxy format, we merge by finding the min and max coordinates
                current_box = [
                    min(current_box[0], boxes[j][0]), 
                    min(current_box[1], boxes[j][1]), 
                    max(current_box[2], boxes[j][2]), 
                    max(current_box[3], boxes[j][3])
                ]
                used.add(j)
        merged_boxes.append(current_box)
        merged_labels.append(labels[i])
        used.add(i)

    return np.array(merged_boxes), merged_labels


############################################# Text search #############################################

def fashion_query_transformer(text_input):

    llm = OpenAI(model="gpt-4-turbo-preview")

    #### text가 패션과 관련된 항목인지 여부를 판단
    first_gateway_output = pass_first_gateway(text_input, llm)
    print(first_gateway_output)

    if (first_gateway_output['clothes_topic']):
        # print("Passed the first gateway. Moving on to the second gateway...")
        if (not first_gateway_output['fashion_item']):
            # print("However, specific item is not found. Searching the whole database.")
            gateway_output = pass_third_gateway(text_input, llm)
        else:
            done=False
            while not done:
                try:
                    gateway_output = pass_second_gateway(text_input, llm)
                    done=True
                except:
                    continue
    else:
        return None
    return gateway_output


def text_search(index, items_dict, model, tokenizer, splade_model, splade_tokenizer, top_k=10):
    search_results = dict()
    for item in items_dict['items']:
        text_emb = get_single_text_embedding(item['refined_text'], model, tokenizer)
        sparse_vector = gen_sparse_vector(item['refined_text'], splade_model, splade_tokenizer)
        
        if 'clothes_type' in list(item.keys()):
            search_result = index.query(
                            vector=text_emb[0],
                            sparse_vector=sparse_vector,
                            top_k=top_k,
                            filter={"category": {"$eq": item['clothes_type']}},
                            include_metadata=True
                        )
            search_results[item['clothes_type']] = search_result
        else:
            search_result = index.query(
                            vector=text_emb[0],
                            sparse_vector=sparse_vector,
                            top_k=top_k,
                            include_metadata=True
                        )
            search_results['all'] = search_result
    return search_results


def pass_first_gateway(input_text, llm, verbose=False):
    first_gateway_prompt = """Input text : {text_input}
    Using the input text, do the following
    - clothes_topic : Determine whether the subject it is related to fashion or clothes. The output should be a python boolean.
    - Determine whether it mentions a specific fashion items such as boots or shirt, umbrella etc. The output should be a python boolean.
    """
    
    class first_gateway(BaseModel):
        "Data model to determine whether the text is related to clothes."
        clothes_topic: bool
        fashion_item: bool

    program = OpenAIPydanticProgram.from_defaults(
        output_cls=first_gateway, prompt_template_str=first_gateway_prompt, llm=llm,verbose=verbose
    )

    output = program(
        text_input=input_text
    )

    return output.dict()


def pass_second_gateway(text_input, llm, verbose=False):
    second_gateway_prompt = """Input text : {text_input}.
    Using the input text, do the following.

    First, divide the items listed in the sentence, ensuring that descriptive words for each item are kept together during the separation.
    Second, for each item listed, do the following :
        - Categorize the clothes type mentioned from the input.
            - From the options below, choose the clothes type mentioned. : 
                'pants', 'shirt, blouse', 'jacket', 'top, t-shirt, sweatshirt',
                'dress', 'shoe', 'glasses', 'skirt', 'bag, wallet', 'belt',
                'headband, head covering, hair accessory', 'sock', 'hat', 'watch',
                'glove', 'tights, stockings', 'sweater', 'tie', 'shorts', 'scarf',
                'coat', 'vest', 'umbrella', 'cardigan', 'cape', 'jumpsuit',
                'leg warmer'
            - a suit is part of jacket
            - If none of the above is mentioned, say "None"
        - Refine the text into a comma-separated string of attributes
            -  as an example, the text 'casual, urban-inspired jacket with bold graphics and loose-fitting designs'
            would be converted to 'casual, urban-inspired, jacket, bold graphics, loose-fit'.
            - another example, the text 'color Pink, - silhouette Straight, - silhouette_fit Loose'
            would be converted to 'color pink, silhouette Straight, silhouette_fit Loose'.
            - do not hesitate to repeat the modifiers for each item.
    The output should be in Englsih.
    """

    class second_gateway_list(BaseModel):
        "Data model to categorize the clothing type, and refine text into a specific format."
        clothes_type: Literal['pants', 'shirt, blouse', 'jacket', 'top, t-shirt, sweatshirt',
                            'dress', 'shoe', 'glasses', 'skirt', 'bag, wallet', 'belt',
                            'headband, head covering, hair accessory', 'sock', 'hat', 'watch',
                            'glove', 'tights, stockings', 'sweater', 'tie', 'shorts', 'scarf',
                            'coat', 'vest', 'umbrella', 'cardigan', 'cape', 'jumpsuit',
                            'leg warmer']
        refined_text: str

    class second_gateway(BaseModel):
        "Data model to list items."
        items: List[second_gateway_list]

    program = OpenAIPydanticProgram.from_defaults(
        output_cls=second_gateway, prompt_template_str=second_gateway_prompt, llm=llm, verbose=verbose
    )

    output = program(
        text_input=text_input
    )

    return output.dict()

def pass_third_gateway(text_input, llm, verbose=False):
    
    third_gateway_prompt = """Input text : {text_input}.
    Using the input text, do the following.
        - Refine the text into a comma-separated string of attributes
            -  as an example, the text 'casual, urban-inspired jacket with bold graphics and loose-fitting designs'
            would be converted to 'casual, urban-inspired, jacket, bold graphics, loose-fit'
            - do not hesitate to repeat the modifiers for each item.
    """

    class third_gateway_list(BaseModel):
        "Data model to reformat an input text."
        refined_text: str

    class third_gateway(BaseModel):
        "Data model to list items."
        items: List[third_gateway_list]

    program = OpenAIPydanticProgram.from_defaults(
        output_cls=third_gateway, prompt_template_str=third_gateway_prompt, llm=llm, verbose=verbose
    )

    output = program(
        text_input=text_input
    )
    return output.dict()

def get_single_text_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors = "pt", padding=True)
    text_embeddings = model.get_text_features(**inputs)
    # convert the embeddings to numpy array
    embedding_as_np = text_embeddings.cpu().detach().numpy()
    return embedding_as_np.tolist()



############################################# sparse vector #############################################

def gen_sparse_vector(text, splade_model, splade_tokenizer):
    tokens = splade_tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        sparse_emb = splade_model(
            d_kwargs=tokens.to('cpu')
        )['d_rep'].squeeze()
    
    indices = sparse_emb.nonzero().squeeze().cpu().tolist()
    values = sparse_emb[indices].cpu().tolist()

    sparse_values = {
        "indices": indices,
        "values": values
    }

    return sparse_values


############################################# hybrid search #############################################


# GPT를 이용해서 이미지를 읽어와서 description을 생성한다

def describe_clothes(image, label, openai_key):
  buffer = io.BytesIO()
  # Save the image to the buffer in JPEG format
  image.save(buffer, format="JPEG")
  buffer.seek(0)
  image_data = buffer.read()

  base64_image = base64.b64encode(image_data).decode('utf-8')
  image_desc_prompt = """Focus on {} inside the image.
        Identify the attributes of the item.
        The attributes you should answer are : 
        - clothes_type
        - color
        - silhouette
        - silhouette_fit
        - waisteline
        - sleeve_type
        - collar_type
        - length
        - gender
        - textile_pattern
        
        Ignore the attributes you cannot answer.
        Keep the answer simple and clear, having max three words per attribute.
  """.format(label)

  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai_key}"
  }

  payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": image_desc_prompt
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            }
          }
        ]
      }
    ],
    "max_tokens": 300
  }

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

  return response.json()['choices'][0]['message']['content']


def calculate_dot_products(embedding, df, column_name):
    dot_products = df[column_name].apply(lambda x: np.dot(embedding, x))
    return dot_products

def get_top_indices(db, input_data, category, clip_processor, clip_model, clip_tokenizer, top_k, type='image'):
    if type=='image':
        # input_data should be a single cropped image
        emb = extract_img_features(input_data, clip_processor, clip_model)
        # Calculate dot products
        dot_products = calculate_dot_products(emb.cpu().numpy()[0], db, 'values')
    elif type=='text':
        # input_data should be a single string of text
        emb = get_single_text_embedding(input_data, clip_model, clip_tokenizer)
        # Calculate dot products
        dot_products = calculate_dot_products(np.array(emb)[0], db, 'values')

    # Find the indices of the top 5 most similar embeddings
    top_indices = dot_products.nlargest(top_k).index

    # Retrieve the top k most similar embeddings
    top_similar_ids = db.loc[top_indices, 'vdb_id'].tolist()

    return {category:top_similar_ids}

def additional_search(local_db, cropped_images, search_results, clip_processor, clip_model, clip_tokenizer, top_k):
    """
    구체적인 아이템 없이, 전반적인 분위기를 설명하는 텍스트인 경우, 이미지로부터 얻을 레이블을 중심으로 서치
    """
    ids = list()
    for category, value in search_results.items():
        id = [i['id'] for i in value['matches']]
        ids.extend(id)

    final_results = list()

    # 전체 아이템 중, 1차 retrieve 된 것들만 가져옴
    db = local_db.loc[local_db['vdb_id'].isin(ids)]

    for label, v in search_results.items(): # 텍스트로부터
        tmp = db.loc[db['name']==label]

        # 텍스트에도 있고, 이미지에도 레이블이 있는 경우
        if label in cropped_images.keys():
            r = get_top_indices(tmp, cropped_images[label], label, clip_processor, clip_model, clip_tokenizer, top_k, type='image')
            final_results.append(r)
        # 텍스트에만 있는 경우, 그냥 top_k를 가져온다
        else:
            final_results.append({ label : [i['id'] for i in v['matches']][:top_k]} )

    refined_result = dict()

    for search_result in final_results:
        category = list(search_result.keys())[0]
        paths = list(search_result.values())[0]

        full_paths = [os.path.join("imaterialist-fashion-2020-fgvc7", "cropped_images", i+".jpg") for i in paths]
        refined_result[category] = full_paths

        
    return refined_result

