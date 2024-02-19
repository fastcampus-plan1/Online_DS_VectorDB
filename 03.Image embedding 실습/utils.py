import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
from PIL import Image
from typing import Tuple, List

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
    # 가장 유사도 높은 이미지는 제외 (query 이미지와 같은 이미지이기 때문)
    topk_indices = sorted_indices_desc[1:topk+1]
    # cosine similarities for the top-k indices
    topk_similarities = similarities[topk_indices]

    return topk_indices, topk_similarities

def draw_images(images: List[Image.Image], texts: List[str] = ['', '', '', '', '']):
    """
    최대 5개까지 이미지 show

    Args:
        images (List[Image.Image]): 이미지 list
        texts (List[str], optional): 이미지 아래에 display 할 텍스트. Defaults to ['', '', '', '', ''].
    """
    # Set up the figure and axes for a 1x5 grid
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))

    for i in range(5):
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

def filter_furniture(detections):
    furniture_class = [56, 57, 59, 60] # detections[0].names
    furniture_names = ['chair', 'couch', 'bed', 'dining table']
    furniture_detected = {}

    filter = [True if i in furniture_names else False for i in detections[1]['labels']]
    furniture_detected['boxes'] = detections[1]['boxes'][filter]
    furniture_detected['scores'] = detections[1]['scores'][filter]
    furniture_detected['categories'] = detections[1]['categories'][filter]
    furniture_detected['lables'] = [item for item, bool in zip(detections[1]['labels'], filter) if bool==True]
    
    return furniture_detected