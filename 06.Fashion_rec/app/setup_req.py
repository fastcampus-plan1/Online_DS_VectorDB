from pinecone import Pinecone
from image_utils import fetch_clip
from splade.splade.models.transformer_rep import Splade
from transformers import AutoTokenizer
from transformers import YolosFeatureExtractor, YolosForObjectDetection
import pandas as pd
import json

import warnings
warnings.filterwarnings("ignore")

# setup

def setup():
    #### pinecone ####
    pc = Pinecone(api_key="74e30e50-02fa-4e55-9bff-affa6a3817a0")

    index = pc.Index("fastcampus")
    index.describe_index_stats()

    #### CLIP model ####
    clip_model, clip_processor, clip_tokenizer = fetch_clip(model_name="patrickjohncyh/fashion-clip")


    #### SPLADE model ####
    splade_model_id = 'naver/splade-cocondenser-ensembledistil'

    splade_model = Splade(splade_model_id, agg='max')
    splade_model.to('cpu')
    splade_model.eval()

    splade_tokenizer = AutoTokenizer.from_pretrained(splade_model_id)

    #### local DB ####
    local_db = pd.read_csv("local_db.csv")
    local_db['values'] = local_db['values'].apply(json.loads)

    #### YOLO model ####
    MODEL_NAME = "valentinafeve/yolos-fashionpedia"

    yolo_feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-small')
    yolo_model = YolosForObjectDetection.from_pretrained(MODEL_NAME)

    return pc, index, clip_model, clip_processor, clip_tokenizer, splade_model, splade_tokenizer, local_db, yolo_feature_extractor, yolo_model