import streamlit as st
import cv2
import numpy as np

from obj_dc import load_yolo_model, load_image, detect_objects, classify_room, draw_objects

def main():
    st.title("Object Detection and Room Classification App")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        yolo, classes, output_layers = load_yolo_model()
        img = load_image(uploaded_file)
        class_ids, confidences, boxes = detect_objects(yolo, img, output_layers)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if any(indexes):
            class_labels = [classes[class_id] for i, class_id in enumerate(class_ids) if i in indexes]
            room_type = classify_room(class_labels)

            img_result = draw_objects(img.copy(), class_ids, boxes, indexes, classes)

            st.image(img_result, caption=f"Result: The detected objects suggest that the room is a {room_type}.", use_column_width=True)
        else:
            st.warning("No objects detected.")

if __name__ == "__main__":
    main()
