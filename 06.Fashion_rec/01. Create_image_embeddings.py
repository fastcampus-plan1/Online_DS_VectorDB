from PIL import Image
import os
import json
from tqdm import tqdm
import numpy as np
from image_utils import extract_img_features
from transformers import CLIPProcessor, CLIPModel


if __name__ == "__main__":
    # load CLIP
    model_name = "patrickjohncyh/fashion-clip"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    print("-"*60)
    print("CLIP 모델 로딩 완료, embedding 추출 시작")

    # crop된 이미지들의 path 불러오기
    cropped_path = "imaterialist-fashion-2020-fgvc7/cropped_images"
    images = list(os.walk(cropped_path))[0][2]
    images = [i for i in images if '.jpg' in i]

    # 기존에 데이터가 있었다면, 삭제
    open('img_embeddings_fashion_fine_tuned.json', 'w').close()

    # local에 한 줄씩 저장을 함
    for i in tqdm(images):
        img = Image.open(os.path.join(cropped_path, i))
        # clip을 활용하여 embedding 추출
        img_emb = extract_img_features(img, processor, model)
        # json 파일에 한 줄씩 작성 (ram에 임베딩들을 올리고 있는 것을 방지)
        with open('img_embeddings_fashion_fine_tuned.json','a') as file:
            key = i.split(".")[0]
            d = {key : np.array(img_emb)[0].tolist()}
            json_string = json.dumps(d, ensure_ascii=False)
            file.write(json_string + '\n')
