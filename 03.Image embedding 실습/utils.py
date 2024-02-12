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