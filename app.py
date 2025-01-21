
import os
import tempfile
import streamlit as st
from aiutils import add_boxes_to_image, generate_prompt, generate_response

def main():
    st.title("Detect and Count Objects in an Image using Gemini AI")

    uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    selected_objects = None
    initial_objects = None
    image_path = None
    manual_input = False

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        # Create a temporary file with a unique name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_image.getvalue())
            image_path = temp_file.name

        # Add checkbox for manual input
        manual_input_checkbox = st.checkbox("Manually input objects to detect")
        if manual_input_checkbox:
            manual_input = True
            selected_objects = st.text_input("Enter objects to detect (comma-separated):")
        else:
            # If initial objects are not set (new image uploaded), ask AI for objects
            select_prompt = """Enumerate all the objects found in the image. Some objects might
            appear multiple times. Only mention each object once. Output as a comma 
            separated list. Do not add any other information."""
            selected_objects = generate_response(image_path, select_prompt)
            st.write(f"Objects found: {selected_objects}")

        if st.button("Analyze Image"):
            with st.spinner("Processing..."):
                if selected_objects:
                    list_objects = selected_objects.split(",")
                    prompt = generate_prompt(list_objects)                    

                    response, object_counts = add_boxes_to_image(image_path, prompt)

                    if response is not None:
                        st.image(response, caption="Image Analysis:", use_container_width=True)

                        st.subheader("Detected Objects:")
                        for obj, count in object_counts.items():
                            st.write(f"{obj}: {count}")
                    else:
                        st.write("Error: Could not display image.")
                else:
                    st.warning("No objects were selected.")

        # Clean up the temporary file after analysis
        os.remove(image_path)
    else:
        st.warning("Please upload an image.")

if __name__ == "__main__":
    main()