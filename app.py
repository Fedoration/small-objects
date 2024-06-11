import time
import streamlit as st
from PIL import Image
import numpy as np
from typing import Optional
import cv2
import copy
from pathlib import Path
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.utils.yolov8 import download_yolov8s_model
from utils import visualize_object_predictions


class SmallObjectsDetectorApp:
    def __init__(self, model_path="checkpoints/yolo_sahi_small.pt"):
        self.model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=model_path,
            confidence_threshold=0.6,
            device="cpu",
        )
        self.images = []
        self.detected_images = []

    def detect_objects(self, image):
        # results = get_prediction(image, self.model)
        results = get_sliced_prediction(
            image,
            self.model,
            slice_height=1024,
            slice_width=1024,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.1,
        )
        result_image = visualize_object_predictions(
            np.ascontiguousarray(results.image), results.object_prediction_list
        )["image"]

        return result_image

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

        # Проверка загрузки новых файлов
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = None
        if "images" not in st.session_state:
            st.session_state.images = None
        if "detected_images" not in st.session_state:
            st.session_state.detected_images = None

        if uploaded_files and uploaded_files != st.session_state.uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            self.load_images(uploaded_files)
            st.session_state.images = self.images
            st.session_state.detected_images = self.detected_images

        if st.session_state.images:
            st.write("Starting slideshow with detection...")
            display_detection = st.checkbox("Show detections", value=True)
            selected_images = []
            for i, img in enumerate(st.session_state.images):
                if display_detection:
                    selected_images.append(st.session_state.detected_images[i])
                else:
                    selected_images.append(img)

            self.slideshow(selected_images)
        else:
            st.info("Please upload images.")

        # detect_people = st.checkbox("Find people", value=True)
        # detect_vehicles = st.checkbox("Find vehicles", value=True)
        # detect_buildings = st.checkbox("Find buildings", value=True)

        # if uploaded_files:
        #     self.load_images(uploaded_files)
        #     if self.images:
        #         st.write("Starting slideshow with YOLOv8 detection...")
        #         selected_images = []
        #         display_detection = st.toggle("Show detections", value=True)
        #         for i, img in enumerate(self.images):
        #             if display_detection:
        #                 selected_images.append(self.detected_images[i])
        #             else:
        #                 selected_images.append(img)

        #         self.slideshow(selected_images)
        #     else:
        #         st.error("No images found in the uploaded files.")
        # else:
        #     st.info("Please upload images.")


# Запуск приложения
if __name__ == "__main__":
    app = SmallObjectsDetectorApp()
    app.run()
