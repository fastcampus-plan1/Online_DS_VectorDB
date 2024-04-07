import streamlit as st
import openai
import os
from setup_req import setup
import requests
from PIL import Image
import io
from io import BytesIO
import base64
from search_method_wrapper import text_input_only, image_input_only, hybrid_input, create_modifications

# Setup. 필요한 모델과 localDB를 한 번에 upload
@st.cache_resource(show_spinner=True)
def cached_setup():
    print("Reading necessary files.")
    return setup()

pc, index, clip_model, clip_processor, \
    clip_tokenizer, splade_model, splade_tokenizer, \
        local_db, yolo_feature_extractor, yolo_model = cached_setup()

# Initialize openai api
os.environ['OPENAI_API_KEY']= "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"
openai.api_key = os.environ["OPENAI_API_KEY"]

# Display an image at the top of the app, fully extended width-wise
st.image("app_image.jpg", use_column_width=True)

# App title
st.title("Welcome to StyleFinder!")

# Function to add logs to the markdown field
def add_log(logs, new_log):
    logs.append(new_log)
    return logs

# Placeholder for displaying logs in the main section
log_placeholder1 = st.empty()
log_placeholder2 = st.empty()

# Initialize image_paths here to make it accessible throughout the app
image_paths = []

# Define the sidebar tabs
tab1, tab2 = st.sidebar.tabs(["상품 서치", "패션 추천"])

# Main placeholders for images and logs
image_display_placeholder = st.empty()  # Add this placeholder to display images in the main section
response = None

# Tab 1: 상품 서치 - StyleFinder
with tab1:
    st.header("상품 서치")
    # Inputs
    text_input1 = st.text_area("Enter the description or name of the fashion item you're searching for:", key="search_text", height=300)
    image_input1 = st.file_uploader("Or upload an image of the fashion item:", type=["jpg", "png"], key="search_image")
    top_k = st.text_input("추천 개수", key="top_k", value=5)
    top_k = int(top_k)
    
    logs1 = []
    if st.button("찾아줘!", key="search_btn"):
        # text and image input
        if text_input1 and image_input1:
            print("-"*30)
            logs1 = add_log(logs1, f"Text and image received.\nText: {text_input1}")
            image = Image.open(image_input1)
            
            # hybrid search
            result = hybrid_input(text_input1, image, index, yolo_feature_extractor, yolo_model, clip_model, clip_tokenizer, clip_processor, local_db, openai.api_key, top_k)
            if result:
                log, image_paths = result

                logs1 = add_log(logs1, "Detected_items:{}".format(log[0]))
                logs1 = add_log(logs1, "image_Descriptions:{}".format(log[1]))
                logs1 = add_log(logs1, "search_text:{}".format(log[2]))
                logs1 = add_log(logs1, "원하는 분위기 :{}".format(log[3]))
            else:
                logs1 = add_log(logs1, "Nothing found")

            log_messages1 = "\n".join(f"- {log}" for log in logs1)
            log_placeholder1.markdown(log_messages1)
            
        elif text_input1 or image_input1:
            # text input only
            if text_input1:
                print("-"*30)
                logs1 = add_log(logs1, f"Text: {text_input1}")

                result = text_input_only(text_input1, index, clip_model, clip_tokenizer, 
                                                   splade_model, splade_tokenizer, top_k=int(top_k))
                
                if result:

                    log, image_paths = result
                
                    logs1 = add_log(logs1, log)
                else:
                    logs1 = add_log(logs1, "The text is not related to clothes/fashion.")
            # image input only
            if image_input1:
                print("-"*30)
                image = Image.open(image_input1)

                result = image_input_only(image, index, yolo_feature_extractor,
                                                     yolo_model, clip_model, clip_tokenizer, clip_processor, local_db, openai.api_key, top_k)
                if result:
                    log, image_paths = result

                    logs1 = add_log(logs1, "Image received.")
                    logs1 = add_log(logs1, "Detected_items:{}".format(log[0]))
                    logs1 = add_log(logs1, "image_Descriptions:{}".format(log[1]))
                    logs1 = add_log(logs1, "search_text:{}".format(log[2]))
                else:
                    logs1 = add_log(logs1, "No fashion item detected.")

            log_messages1 = "\n".join(f"- {log}" for log in logs1)
            log_placeholder1.markdown(log_messages1)
        else:

            log_placeholder1.markdown("- Please provide at least one input.")


# Tab 2: 패션 추천 - StyleFinder
with tab2:
    st.header("패션 추천")
    # Inputs
    text_input2 = st.text_area("Tell us what you're in the mood for or upload a style reference image:", key="recommend_text", height=300)
    image_input2 = st.file_uploader("Upload an image for style reference:", type=["jpg", "png"], key="recommend_image")
    
    if st.button("추천해줘!", key="recommend_btn"):
        if text_input2 and image_input2:
            print("-"*30)

            # Convert the uploaded file to a PIL Image
            image = Image.open(image_input2)

            # Convert the PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format)
            img_bytes = img_byte_arr.getvalue()

            # Convert the bytes to a base64 encoded string
            base64_image = base64.b64encode(img_bytes).decode('utf-8')

            result = create_modifications(text_input2, base64_image, openai.api_key)

            if result:
                log, image_url = result
                # Fetch the content of the image
                response = requests.get(image_url)
                
            else:
                log_placeholder2.markdown("- Please provide fashion-related text")    
            
        else:
            log_placeholder2.markdown("- Please provide text and image input.")

if image_input1 is not None:
    st.markdown("Uploaded image:")
    image = Image.open(image_input1)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

if image_input2 is not None:
    st.markdown("Uploaded image:")
    image = Image.open(image_input2)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

if response:
    if response.status_code == 200:
        # Open the image
        image = Image.open(BytesIO(response.content))
        
        # Display the image
        log_placeholder1.markdown(log)
        st.image(image, caption='Image from URL')
    else:
        st.error('Failed to fetch image. Please check the URL.')


# Display images in the main section after tabs
if image_paths:  # Check if there are any images to display
    for k,v in image_paths.items():
        st.write(k)
        cols = st.columns(len(v))  # Create columns based on the number of images
        for col, img_path in zip(cols, v):
            with col:
                st.image(img_path)  # Display each image in its column
