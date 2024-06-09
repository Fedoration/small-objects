import time
import streamlit as st
from PIL import Image
import cv2
from ultralytics import YOLO


class SmallObjectsDetectorApp:
    def __init__(self, model_path="checkpoints/yolov8n.pt"):
        self.model = YOLO(model_path)
        self.images = []
        self.detected_images = []

    def detect_objects(self, image):
        results = self.model(image)
        detected_image = results[0].plot()
        detected_image_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(detected_image_rgb)

    def slideshow(self, images, delay=0.01):
        for img in images:
            st.image(img)
            time.sleep(delay)

    def load_images(self, uploaded_files):
        self.images = [Image.open(file) for file in uploaded_files]
        self.detected_images = [self.detect_objects(img) for img in self.images]

    def run(self):
        st.title("Finding small objects")

        # Виджет для загрузки изображений
        uploaded_files = st.file_uploader(
            "Upload images",
            type=["png", "jpg", "jpeg", "gif", "bmp"],
            accept_multiple_files=True,
        )

        detect_people = st.checkbox("Find people", value=True)
        detect_vehicles = st.checkbox("Find vehicles", value=True)
        detect_buildings = st.checkbox("Find buildings", value=True)

        if uploaded_files:
            self.load_images(uploaded_files)
            if self.images:
                st.write("Starting slideshow with YOLOv8 detection...")
                selected_images = []
                display_detection = st.toggle("Show detections", value=True)
                for i, img in enumerate(self.images):
                    if display_detection:
                        selected_images.append(self.detected_images[i])
                    else:
                        selected_images.append(img)

                self.slideshow(selected_images)
            else:
                st.error("No images found in the uploaded files.")
        else:
            st.info("Please upload images.")


# Запуск приложения
if __name__ == "__main__":
    app = SmallObjectsDetectorApp()
    app.run()
