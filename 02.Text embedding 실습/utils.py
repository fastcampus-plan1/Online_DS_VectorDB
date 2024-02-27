import numpy as np
from numpy.linalg import norm
from typing import List, Tuple
from openai import OpenAI
import openai

def cosine_similarity(vector_a, vector_b):
    """Calculate the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = norm(vector_a)
    norm_b = norm(vector_b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def create_embeddings(txt_list: List[str], model='text-embedding-3-small') -> List[np.ndarray]:
    """
    주어진 텍스트 목록에 대한 embedding vector를 생성함

    Args:
        txt_list (List[str]): embedding을 생성할 텍스트 목록
        model (str, optional): embedding 모델

    Returns:
        List[np.ndarray]: 각 embedding vector
    """

    client = OpenAI()

    response = client.embeddings.create(
    input=txt_list,
    model=model)
    responses = [r.embedding for r in response.data]

    return responses

def search_similar_vector(query_feature: np.array, features: List[np.array], topk: int = 10) -> Tuple[np.array, np.array]:
    """
    주어진 vector들과 비교하여, query_feature와 유사한 vector의 index와 유사도를 제공함

    Args:
        query_feature (np.array): input embedding vector
        features (List[np.array]): embedding vector들의 list
        topk (int, optional): top-k

    Returns:
        Tuple[np.array, np.array]: 유사한 embedding vector들의 index & cosine-distance
    """
    similarities = cosine_similarity([query_feature], np.vstack(features)).flatten()
    # sort in descending order
    sorted_indices_desc = similarities.argsort()[::-1]
    topk_indices = sorted_indices_desc[0:topk]
    # cosine similarities for the top-k indices
    topk_similarities = similarities[sorted_indices_desc]

    return topk_indices, topk_similarities


def normal_chat_completion(input_prompt: str, model: str = 'gpt-4-turbo-preview') -> dict:
    """
    Openai chat completion을 활용하여 JSON output 생성

    Args:
        input_prompt (str): The input prompt to the chat model.
        model (str, optional): Model name. Defaults to 'gpt-4-turbo-preview'.

    Returns:
        dict: The chat completion response formatted as a JSON object.
    """
    client = openai.OpenAI()

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": 'You are a smart and intelligent program that understands information and provides output in JSON format'},
            {"role": "user", "content": input_prompt}
        ]
    )
    return response
