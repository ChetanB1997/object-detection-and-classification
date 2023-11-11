import cv2
import streamlit as st
import numpy as np

def load_yolo_model():
    yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []

    with open("coco.names", "r") as file:
        classes = [line.strip() for line in file.readlines()]

    layer_names = yolo.getLayerNames()
    output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers()]

    return yolo, classes, output_layers

def load_image(image):
    if isinstance(image, str):
        # If the input is a file path, load the image using cv2.imread
        img = cv2.imread(image)
    else:
        # If the input is an uploaded file, read it using OpenCV and convert to RGB
        image_content = np.asarray(bytearray(image.read()), dtype=np.uint8)
        img = cv2.imdecode(image_content, cv2.IMREAD_COLOR)

    return img

def detect_objects(yolo, img, output_layers):
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    yolo.setInput(blob)
    outputs = yolo.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return class_ids, confidences, boxes

def classify_room(objects_detected):
    
    bedroom_objects = ["bed", "wardrobe", "nightstand"]
    kitchen_objects = ["oven", "microwave", "sink", "refrigerator"]
    living_room_objects = ["sofa", "chair", "coffee table", "television"]

    detected_objects_set = set(objects_detected)

    if detected_objects_set.intersection(bedroom_objects):
        return "Bedroom"
    elif detected_objects_set.intersection(kitchen_objects):
        return "Kitchen"
    elif detected_objects_set.intersection(living_room_objects):
        return "Living Room"
    else:
        return "Unclassified"

def draw_objects(image, class_ids, boxes, indexes, classes):
    colorRed = (0, 0, 255)
    colorGreen = (0, 255, 0)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(image, (x, y), (x + w, y + h), colorGreen, 3)
            cv2.putText(image, label, (x, y + 10), cv2.FONT_HERSHEY_PLAIN, 1, colorRed, 1)

    return image

def main():
    st.title("Object Detection and Room Classification App")

    image_path = "img5.jpg"

    yolo, classes, output_layers = load_yolo_model()
    img = load_image(image_path)
    class_ids, confidences, boxes = detect_objects(yolo, img, output_layers)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    if any(indexes):
        class_labels = [classes[class_id] for i, class_id in enumerate(class_ids) if i in indexes]
        room_type = classify_room(class_labels)

        img_result = draw_objects(img.copy(), class_ids, boxes, indexes, classes)

        st.image(img_result, caption=f"Result: The detected objects suggest that the room is a {room_type}.", use_column_width=True)
    else:
        st.warning("No objects detected.")

    ################
    # yolo, classes, output_layers = load_yolo_model()
    # image_path = "img5.jpg"

    # img = load_image(image_path)
    # class_ids, confidences, boxes = detect_objects(yolo, img, output_layers)

    # indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # if any(indexes):
    #     class_labels = [classes[class_id] for i, class_id in enumerate(class_ids) if i in indexes]
    #     room_type = classify_room(class_labels)

    #     img_result = draw_objects(img.copy(), class_ids, boxes, indexes, classes)

    #     print(f"The detected objects suggest that the room is a {room_type}.")

    #     cv2.imwrite("output5.jpg", img_result)
    # else:
    #     print("No objects detected.")

if __name__ == "__main__":
    main()
