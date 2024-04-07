from search_utils import fashion_query_transformer, clothes_detector, text_search, \
                         gen_sparse_vector, describe_clothes, additional_search
from yolo_utils import fix_channels
from torchvision.transforms import ToTensor
from search_utils import get_top_indices
import os
from openai import OpenAI
import io
import base64
import requests
import warnings
warnings.filterwarnings("ignore")


def redirect_image_path(paths):
    new_paths = dict()
    for k,v in paths.items():
        new_paths[k] = [f"../{p}" for p in v]
    return new_paths

def text_input_only(input, pc_index, clip_model, clip_tokenizer, splade_model, splade_tokenizer, top_k=10):
    """Returns None if the input text is inappropriate."""
    text_query = fashion_query_transformer(input)

    print("### Result from the text_input gateway : \n{}".format(text_query))
    
    if text_query:
        # search
        print("Searching ...")
        # text search
        result = text_search(pc_index, text_query, clip_model, clip_tokenizer, splade_model, splade_tokenizer, top_k=top_k)

        # 이미지들의 path 가져오기
        paths = dict()
        for k,v in result.items():
            paths[k] = [i['metadata']['img_path'] for i in v['matches']]
    else:
        print("패션과 무관한 텍스트 입니다. 다시 입력하세요.")
        return ["패션과 무관한 텍스트 입니다. 다시 입력하세요", None]

    return text_query, redirect_image_path(paths)


def image_input_only(image, index, yolo_feature_extractor, yolo_model, clip_model, clip_tokenizer, clip_processor, splade_model, splade_tokenizer, local_db, api_key, top_k):
    
    image = fix_channels(ToTensor()(image))

    # object detections
    print("Detecting items from the image.")
    cropped_images = clothes_detector(image, yolo_feature_extractor, yolo_model)

    if len(cropped_images.keys())==0:
        print("Nothing detected from the image")
        return None

    # describe the labels I have found
    descriptions = dict()

    print("Start creating descriptions for each item.")
    for i, img in cropped_images.items():
        print(i)
        desc = describe_clothes(img, i, api_key)
        descriptions[i] = desc
    print("Transform the descriptions into structured query.")
    text_query = fashion_query_transformer(str(descriptions))
    
    print("Retrieved 100 images based on text")
    results = text_search(index, text_query, clip_model, clip_tokenizer, splade_model, splade_tokenizer, top_k=100)
    print("Conducting additional search using the images")
    results2 = additional_search(local_db, cropped_images, results, clip_processor, clip_model, clip_tokenizer, top_k)

    return [list(cropped_images.keys()), descriptions, text_query], redirect_image_path(results2)


def hybrid_input(input, image, index, yolo_feature_extractor, yolo_model, clip_model, clip_tokenizer, clip_processor, local_db, api_key, top_k):
    
    # 가장 먼저 text를 심사하면서, fashion과 연관된 text인지, 또는 구체적인 아이템이 없는지 등을 체크
    print("Starting hybrid search...")
    text_query = fashion_query_transformer(input)

    # fashion 관련 쿼리
    if text_query:
        print("Text query: ", text_query)

        if 'clothes_type' not in text_query['items'][0].keys():
            # 구체적인 아이템이 없는 쿼리
            print("doing image search")
            # image search
            log, image_search_result = image_input_only(image, index, yolo_feature_extractor, yolo_model, clip_model, clip_tokenizer, clip_processor, local_db, api_key, top_k=20)
            log.extend(text_query)
            print("Image search result : ", image_search_result.keys())

            new_results = list()

            for k,v in image_search_result.items():
                # file_name을 다시 가져온다
                ids = [os.path.splitext(os.path.basename(i))[0] for i in v]
                tmp = local_db.loc[local_db['vdb_id'].isin(ids)]

                # similarity search using text input
                if top_k>10:
                    # 10개로 강제
                    top_k=10
                r = get_top_indices(tmp, text_query['items'][0]['refined_text'], k, clip_processor, clip_model, clip_tokenizer, top_k=top_k, type='text')
                new_results.append(r)
        
            # 통일된 아웃풋 형태로 변환
            refined_result = dict()

            for search_result in new_results:
                category = list(search_result.keys())[0]
                paths = list(search_result.values())[0]

                full_paths = [os.path.join("imaterialist-fashion-2020-fgvc7", "cropped_images", i+".jpg") for i in paths]
                refined_result[category] = full_paths

            return log, redirect_image_path(refined_result)
        
        else:
            print("구체적인 아이템을 언급하는 대신, 이미지를 기반으로 원하는 전반적인 분위기를 말씀해주세요.")
            return None
    
    else:
        print("")
        return None


############### recommender ###############

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def create_modifications(text_desc, image, openai_key):
    text_query = fashion_query_transformer(text_desc)

    if text_query:
        print("input Text query : ", text_query)
        # GPT를 이용해서 이미지를 읽어와서 description을 생성한다

        # text_desc = "convert it into a more street fashion styled clothings"


        image_desc_prompt = """
        Analyze the user provided input and ensure the description accurately reflects the 'user preference'.

        Incorporating user input, the modified fashion description emphasizes their unique color palette, 
        showcases an updated fashion style with innovative textiles, introduces intricate patterns for visual interest, 
        and highlights a distinctive shape that redefines their overall silhouette.
        The description should consider attributes such as textile, sleeve, color etc too.
        It should consider both the fashion in the image, and the 'user input'.

        Remember, the total length of your response, including all characters and spaces, must stay within the 500-letter constraint. 
        Aim for clarity and brevity in your answer.

        #user input : {}
        """.format(text_desc)

        # # Path to your image
        # image_path = "test_images/test_image5.jpg"

        # Getting the base64 string
        # base64_image = encode_image(image_path)

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
                    "url": f"data:image/jpeg;base64,{image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 800
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        print("Created clothes descriptions based on user image and text")
        print("Now creating image ...")

        # Dall-E intpu
        
        img_desc = response.json()['choices'][0]['message']['content']
        text_desc = "more formal, suitable for a wedding event. a green coat"

        text_prompt = """Create a visual representation of the following fashion description. 
        Focus on capturing the essence of the outfit in a realistic manner without overcomplication.
        The background should be subtle and not detract from the outfit itself, 
        Choose a minimalistic background that complements the style of the attire:

        Fashion Description:
        {}
        """.format(img_desc)
        

        client = OpenAI()

        response = client.images.generate(
        model="dall-e-3",
        prompt=text_prompt,
        size="1024x1024",
        quality="standard",
        n=1,
        style='vivid'
        )

        image_url = response.data[0].url
        print("Image created!")
        # url is live for 60 seconds after generation
        return img_desc, image_url

    else:
        return None