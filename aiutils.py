from PIL import Image, ImageDraw
import streamlit as st
import google.generativeai as genai
import json
import os
import re

MODEL_ID = "gemini-2.0-flash-exp" 
api_key = os.getenv("GEMINI_API_KEY")
model_id = MODEL_ID
genai.configure(api_key=api_key)
model = genai.GenerativeModel(MODEL_ID)

# For this task, we will not use streaming
ENABLE_STREAMING = False

def generate_response(image_path, prompt):
    """Generates a response from an AI model

    Args:
    prompt: The prompt to send to the AI model.

    Returns:
    response from the AI model.
    """
    try:
        # Read image data as bytes
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Create the PDF input for the model
        image_part = {
            "mime_type": "image/jpeg",
            "data": image_bytes
        }

        # Send file and prompt to Gemini API
        response = model.generate_content(
            [
                prompt, 
                image_part
            ]
        )

        return response.text

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None 

def plot_bounding_boxes(im, noun_phrases_and_positions):
    """
    Plots bounding boxes on an image with markers for each noun phrase, using PIL, normalized coordinates, and different colors.

    Args:
        im: The PIL Image object to draw on.
        noun_phrases_and_positions: A list of tuples containing the noun phrases
         and their positions in normalized [y1, x1, y2, x2] format.
    """
    im = Image.open(im)
    img = im.copy()
    width, height = img.size
    draw = ImageDraw.Draw(img)

    colors = [
        'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown',
        'gray', 'beige', 'turquoise', 'cyan', 'magenta', 'lime', 'navy', 
        'maroon', 'teal', 'olive', 'coral', 'lavender', 'violet', 'gold', 'silver'
    ]

    for i, (noun_phrase, (y1, x1, y2, x2)) in enumerate(noun_phrases_and_positions):
        color = colors[i % len(colors)]

        abs_x1 = int(x1 * width)
        abs_y1 = int(y1 * height)
        abs_x2 = int(x2 * width)
        abs_y2 = int(y2 * height)

        draw.rectangle(
            ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
        )

        text_position = (abs_x1 + 8, abs_y1 + 6)
        draw.text(text_position, noun_phrase, fill=color)

    return img

def format_input(data):
    """
    Converts the input list of tuples into the format required by the plot_bounding_boxes function.

    Args:
        data: A list of tuples in the format [(noun_phrase, [x1, y1, x2, y2]), ...].

    Returns:
        A list of tuples in the format [(noun_phrase, [y1, x1, y2, x2]), ...],
        where coordinates are normalized (0–1).
    """
    formatted_data = []

    for noun_phrase, coordinates in data:
        x1, y1, x2, y2 = coordinates
        normalized_coordinates = [x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000]
        formatted_data.append((noun_phrase, normalized_coordinates))

    return formatted_data

def parse_list_boxes_with_label(text):
    text = text.split("```\n")[0]
    try:
        result =  json.loads(text.strip("```").strip("python").strip("json").replace("'", '"').replace('\n', '').replace(',}', '}'))
        return result
    except Exception as e:
        st.write(f"An error occurred: {e}")
        return None, {}
    
def generate_prompt(object_list):
    objects_string = ", ".join(object_list)
    return f"Return bounding boxes for each object: {objects_string} in the following format as \
        a list.\n {{'{object_list[0]}_0': [ymin, xmin, ymax, xmax], ...}}. If there are more than \
            one instance of an object, add them as 'object_0', 'object_1', etc. \
            Output only a valid JSON. Do not add any other information."

def add_boxes_to_image(image_file, prompt):
    
    try:
        if image_file is not None:
            response = generate_response(image_file, prompt)

            boxes = parse_list_boxes_with_label(response)
            noun_phrases_and_positions = list(boxes.items())
            noun_phrases_and_positions = format_input(noun_phrases_and_positions)

            object_counts = {}
            for noun_phrase, _ in noun_phrases_and_positions:
                object_name = noun_phrase.split("_")[0]
                object_counts[object_name] = object_counts.get(object_name, 0) + 1

            output_image = plot_bounding_boxes(image_file, noun_phrases_and_positions)
            return output_image, object_counts
        
        else:
            st.warning("Please upload an image.")
            return None, {}
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None, {}
    
def extract_list(input_string):
    """
    Extracts a list from a given string.

    Args:
        input_string (str): The input string containing the list.

    Returns:
        list: The extracted list.
    """
    # Use regular expression to find the list in the string
    match = re.search(r'\[(.*?)\]', input_string)

    # If a match is found, extract the list and split it into items
    if match:
        list_string = match.group(1)
        list_items = [item.strip().strip('"').strip("'") for item in list_string.split(",")]
        return list_items
    else:
        return None