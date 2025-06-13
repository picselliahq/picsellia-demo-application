import asyncio
import base64
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QComboBox, QInputDialog, QGridLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsProxyWidget
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QRectF, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPainterPath, QBrush
import datetime
import os
from picsellia import Client
from config import config
from ultralytics import YOLO
import numpy as np
import random
from picsellia.types.schemas_prediction import DetectionPredictionFormat 
import torch
import uuid
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
from PyQt5.QtCore import QUrl
import picsellia

class PredictionGridView(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_StyledBackground, True)
        
        self.main_layout = QVBoxLayout(self) # New main layout to stack title and images
        self.main_layout.setContentsMargins(0, 0, 0, 0) # Set margins to 0
        self.main_layout.setSpacing(0) # Set spacing to 0

        # Title label
        self.title_label = QLabel("Predictions monitored and pulled from Picsellia", self)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        self.main_layout.addWidget(self.title_label)

        self.image_layout = QHBoxLayout() # Original image layout
        self.image_layout.setContentsMargins(0, 0, 0, 0) # Set margins to 0
        self.image_layout.setSpacing(0) # Set spacing to 0
        
        # Create 6 image labels
        self.image_labels = []
        for i in range(6):
            label = QLabel()
            label.setFixedSize(int(self.width() / 6), 200)
            label.setStyleSheet("""
                QLabel {
                    background-color: rgba(0,0,0,0); /* Explicitly set fully transparent background */
                    border: none;
                    border-radius: 0px; /* Set border-radius to 0px to rule out artifacts */
                    margin: 0px; /* Explicitly set margin to 0px */
                }
            """)
            label.setAlignment(Qt.AlignCenter)
            label.setFrameShape(QLabel.NoFrame)
            label.setContentsMargins(0,0,0,0)
            label.setAutoFillBackground(False) # Prevent default background filling
            self.image_labels.append(label)
            self.image_layout.addWidget(label)
        
        self.main_layout.addLayout(self.image_layout)
        self.setLayout(self.main_layout) # Set the new main layout

        # Initialize QNetworkAccessManager for downloading images
        self.network_manager = QNetworkAccessManager()

        # Ensure the PredictionGridView itself is transparent
        self.setStyleSheet("background-color: transparent;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # Explicitly clear the background to ensure transparency for the PredictionGridView itself
        painter.setCompositionMode(QPainter.CompositionMode_Clear)
        painter.fillRect(self.rect(), Qt.transparent)

        # FOR DEBUGGING: Draw a semi-transparent red rectangle
        painter.setBrush(QColor(255, 0, 0, 128)) # Red with 50% opacity
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())
        # END DEBUGGING
        super().paintEvent(event) # Call base class to allow children to be painted

    def update_images(self, predictions):
        # Only take first 6 predictions
        predictions = predictions[:6]
        for i, (label, pred) in enumerate(zip(self.image_labels, predictions)):
            if pred and hasattr(pred, 'url'):
                # Create a request to download the image
                request = QNetworkRequest(QUrl(pred.url))
                reply = self.network_manager.get(request)
                # Connect the finished signal to a lambda that will update the label
                reply.finished.connect(
                    lambda r=reply, l=label: self.handle_image_download(r, l)
                )
            else:
                label.clear()

    def handle_image_download(self, reply, label):
        if reply.error() == QNetworkReply.NoError:
            # Read the image data
            data = reply.readAll()
            # Create QImage from the data
            image = QImage.fromData(data)
            if not image.isNull():
                # Ensure the image has an alpha channel for proper transparency
                if image.format() != QImage.Format_ARGB32:
                    image = image.convertToFormat(QImage.Format_ARGB32)

                # Calculate scaling to fit the container while maintaining aspect ratio
                container_width = label.width()
                container_height = label.height()
                container_ratio = container_width / container_height
                image_ratio = image.width() / image.height()
                
                if image_ratio > container_ratio:
                    # Image is wider than container ratio, fit to width
                    scaled_width = container_width
                    scaled_height = int(container_width / image_ratio)
                else:
                    # Image is taller than container ratio, fit to height
                    scaled_height = container_height
                    scaled_width = int(container_height * image_ratio)
                
                # Scale pixmap to fit container while maintaining aspect ratio
                pixmap = QPixmap.fromImage(image)
                scaled_pixmap = pixmap.scaled(
                    scaled_width, scaled_height,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                
                # Create a new pixmap with the container size and transparent background
                final_pixmap = QPixmap(container_width, container_height)
                final_pixmap.fill(Qt.transparent)
                
                # Draw the scaled image centered in the container
                painter = QPainter(final_pixmap)
                painter.setRenderHint(QPainter.Antialiasing)
                painter.drawPixmap((container_width - scaled_width) // 2,
                                   (container_height - scaled_height) // 2,
                                   scaled_pixmap)
                painter.end()
                
                label.setPixmap(final_pixmap)
            else:
                label.clear()
        else:
            label.clear()
        reply.deleteLater()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Update label sizes when the widget is resized
        for label in self.image_labels:
            label.setFixedSize(int(self.width() / 6), 200)


class PredictionWorker(QThread):
    finished = pyqtSignal(tuple)
    error = pyqtSignal(str)

    def __init__(self, deployment, frame, results):
        super().__init__()
        self.deployment = deployment
        self.frame = frame
        self.results = results

    def run(self):
        try:
            detection_scores = []
            detection_boxes = []
            detection_classes = []
            speed = 0
            for result in self.results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    detection_scores.append(conf)
                    detection_boxes.append([y1, x1, y2, x2])
                    detection_classes.append(cls)
                speed += (result.speed['preprocess']+result.speed['inference']+result.speed['postprocess'])/1000

            predictions = DetectionPredictionFormat(
                detection_scores=detection_scores,
                detection_boxes=detection_boxes,
                detection_classes=detection_classes
            )
            now_timestamp = datetime.datetime.now().timestamp()
            ret, jpeg = cv2.imencode('.jpg', self.frame)
            if not ret:
                self.error.emit("Error: Failed to encode frame to JPEG")
                return

            jpeg_bytes = jpeg.tobytes()
            self.deployment.monitor_bytes(
                raw_image=jpeg_bytes,
                content_type="image/jpeg",
                prediction=predictions,
                filename=str(uuid.uuid4())+".jpg",
                source="CVPR",
                latency=speed,
                timestamp=datetime.datetime.now().timestamp(),
                height=self.frame.shape[0],
                width=self.frame.shape[1],
            )
            self.finished.emit((predictions, now_timestamp))
        except Exception as e:
            self.error.emit(str(e))


class PredictWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, deployment_latent: picsellia.Deployment, frame):
        super().__init__()
        self.deployment_latent = deployment_latent
        self.frame = frame
        
    def run(self):
        try:
            ret, jpeg = cv2.imencode('.jpg', self.frame)
            if not ret:
                self.error.emit("Error: Failed to encode frame to JPEG")
                return

            jpeg_bytes = jpeg.tobytes()
            prediction = self.deployment_latent.predict_bytes(
                raw_image=jpeg_bytes,
                filename=str(uuid.uuid4())+".jpg",
                tags="ATTENDANCE-MONITORING"
            )
            self.finished.emit(prediction)
        except Exception as e:
            self.error.emit(str(e))


class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Stream")
        # Set window attributes for transparency
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setWindowFlags(Qt.FramelessWindowHint)
        # Set main window background to transparent
        self.setStyleSheet("QMainWindow { background: transparent; }")

        self.yolo_device = self.get_yolo_device()
        self.confidence_threshold = 0.7  # Default value
        self.current_source = 0
        self.class_colors = {}
        self.current_frame = None
        self.current_predictions = None

        os.makedirs('models', exist_ok=True) # Ensure models directory exists
        self.available_models = self.scan_models_directory()
        self.current_model_path = self.available_models[0] if self.available_models else 'yolov8n.pt'
        self.deployment_id_map = {
            'yolov8s-coco.pt': config.COCO_DEPLOYMENT_ID,
            'yolov11s-face.pt': config.FACE_DEPLOYMENT_ID,
            'yolov8n.pt': config.COCO_DEPLOYMENT_ID
        }
        self.setup_ui()
        self.init_model(self.current_model_path)
        self.init_picsellia()
        self.showFullScreen()
        self.position_elements()
        self.start_camera_timer()
        self.start_prediction_refresh_timer()
        self.start_auto_predict_timer()

    # --- UI SETUP ---
    def setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        # Set central widget background to transparent and ensure attributes for transparency
        self.central_widget.setStyleSheet("background: transparent;")
        self.central_widget.setAttribute(Qt.WA_TranslucentBackground)
        self.central_widget.setAttribute(Qt.WA_NoSystemBackground)

        self.setup_logo()
        self.setup_source_dropdown()
        self.setup_model_dropdown()
        self.setup_slider()
        self.setup_buttons()

        # Setup Graphics View for video and overlays
        self.graphics_view = QGraphicsView(self.central_widget)
        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setStyleSheet("background-color: transparent; border: none;") # Explicitly set background-color
        self.graphics_view.setAttribute(Qt.WA_TranslucentBackground)
        self.graphics_view.setAttribute(Qt.WA_NoSystemBackground)

        self.graphics_scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_scene.setBackgroundBrush(QBrush(Qt.NoBrush)) # Create a QBrush with NoBrush style

        self.video_item = QGraphicsPixmapItem() # To display video frames
        self.graphics_scene.addItem(self.video_item)

        # Now setup prediction grid as a QGraphicsProxyWidget
        self.prediction_grid = PredictionGridView() # No parent passed here initially
        self.prediction_grid_proxy = self.graphics_scene.addWidget(self.prediction_grid)
        # Set proxy widget's z-value to ensure it's on top of the video
        self.prediction_grid_proxy.setZValue(1)
        # Explicitly ensure the proxy widget itself is transparent
        self.prediction_grid_proxy.setOpacity(1.0) # Set to opaque
        self.prediction_grid_proxy.setOpacity(0.999) # Then slightly transparent to force transparency

    def setup_logo(self):
        self.logo_label = QLabel(self.central_widget)
        self.logo_pixmap = QPixmap('assets/picsellia_logo.png')
        self.logo_label.setPixmap(self.logo_pixmap.scaledToWidth(160, Qt.SmoothTransformation))
        self.logo_label.setFixedHeight(50)
        self.logo_label.setFixedWidth(180)
        self.logo_label.setStyleSheet("background: transparent;")
        self.logo_label.show()

    def setup_source_dropdown(self):
        self.source_dropdown = QComboBox(self.central_widget)
        self.source_dropdown.setFixedWidth(120)
        self.source_dropdown.addItem("Camera 0", 0)
        self.source_dropdown.addItem("Camera 1", 1)
        self.source_dropdown.addItem("Camera 2", 2)
        self.source_dropdown.addItem("Custom URL...")
        self.source_dropdown.currentIndexChanged.connect(self.change_video_source)
        self.source_dropdown.show()

    def setup_model_dropdown(self):
        self.model_dropdown = QComboBox(self.central_widget)
        self.model_dropdown.setFixedWidth(150)
        if not self.available_models:
            self.model_dropdown.addItem("No models found", None)
            self.model_dropdown.setEnabled(False)
            print("Warning: No YOLO models found in 'models/' directory.")
        else:
            for model_path in self.available_models:
                model_name = os.path.basename(model_path)
                self.model_dropdown.addItem(model_name, model_path)
            # Set initial selection
            if 'yolov8n.pt' in [os.path.basename(p) for p in self.available_models]:
                initial_index = [os.path.basename(p) for p in self.available_models].index('yolov8n.pt')
                self.model_dropdown.setCurrentIndex(initial_index)
            self.model_dropdown.currentIndexChanged.connect(self.change_model_source)
        self.model_dropdown.show()

    def setup_slider(self):
        self.slider_container = QWidget(self.central_widget)
        self.slider_layout = QHBoxLayout(self.slider_container)
        self.slider_layout.setContentsMargins(0, 0, 0, 0)
        self.slider_label = QLabel(f"Confidence: {self.confidence_threshold:.2f}", self.slider_container)
        self.slider = QSlider(Qt.Horizontal, self.slider_container)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(int(self.confidence_threshold * 100))
        self.slider.valueChanged.connect(self.update_confidence_threshold)
        self.slider_layout.addWidget(self.slider_label)
        self.slider_layout.addWidget(self.slider)
        self.slider_container.setFixedWidth(220)
        self.slider_container.setFixedHeight(40)
        self.slider_container.show()

    def setup_buttons(self):
        self.button_container = QWidget(self.central_widget)
        self.button_layout = QHBoxLayout(self.button_container)
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        self.button_layout.setSpacing(10)

        self.predict_button = QPushButton("Send prediction to picsellia", self.button_container)
        self.predict_button.setFixedSize(250, 30)
        self.predict_button.setStyleSheet("""
            QPushButton {
                background-color: #9780FF;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 14px;
                font-weight: normal;
            }
            QPushButton:hover {
                background-color: #BAAEFF;
            }
            QPushButton:pressed {
                background-color: #BAAEFF;
            }
        """)
        self.predict_button.clicked.connect(self.predict_frame)
        self.button_layout.addWidget(self.predict_button)

        self.stop_button = QPushButton("Stop", self.central_widget)
        self.stop_button.setFixedSize(80, 30)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #FF4444;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #FF6666;
            }
            QPushButton:pressed {
                background-color: #CC3333;
            }
        """)
        self.stop_button.clicked.connect(self.close)

    # --- MODEL & INTEGRATION SETUP ---
    def get_yolo_device(self):
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'

    def scan_models_directory(self):
        models_dir = 'models'
        if not os.path.exists(models_dir):
            return []
        model_files = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith('.pt')]
        # Add yolov8n.pt if it's not in the models directory but might be downloaded by YOLO itself
        if 'yolov8n.pt' not in [os.path.basename(m) for m in model_files]:
            model_files.insert(0, 'yolov8n.pt') # Prioritize default model
        return model_files

    def init_model(self, model_path):
        try:
            self.model = YOLO(model_path)
            self.model.to(self.yolo_device)
            self.deployment = self.client.get_deployment_by_id(self.deployment_id_map[model_path.split('/')[-1]])
        except Exception as e:
            self.model = None

    def init_picsellia(self):
        print(config.PICSELLIA_TOKEN, config.PICSELLIA_ORGANIZATION)
        try:
            self.client = Client(api_token=config.PICSELLIA_TOKEN, organization_id=config.PICSELLIA_ORGANIZATION)
            self.deployment = None
            self.deployment_latent = self.client.get_deployment_by_id('01976966-d67c-7b71-bf3b-65b9f2c0a177')
        except Exception as e:
            print("WARNING: not logged", str(e))
            self.client = None
            self.deployment = None

    # --- CAMERA & TIMER ---
    def start_camera_timer(self):
        self.camera = cv2.VideoCapture(self.current_source)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30) # Update every 30ms

    def change_video_source(self, index):
        data = self.source_dropdown.itemData(index)
        if data is not None:
            self.current_source = data
            if hasattr(self, 'camera') and self.camera is not None:
                self.camera.release()
            self.camera = cv2.VideoCapture(self.current_source)
        else:
            url, ok = QInputDialog.getText(self, "Custom Stream URL", "Enter video stream URL:")
            if ok and url:
                self.current_source = url
                if hasattr(self, 'camera') and self.camera is not None:
                    self.camera.release()
                self.camera = cv2.VideoCapture(self.current_source)
            else:
                self.source_dropdown.setCurrentIndex(0)

    def change_model_source(self, index):
        new_model_path = self.model_dropdown.itemData(index)
        if new_model_path:
            self.current_model_path = new_model_path
            self.init_model(new_model_path)
        else:
            print("Error: Invalid model selection.")

    # --- MAIN LOOP ---
    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            if self.model:
                try:
                    results = self.model.track(frame, persist=True, device=self.yolo_device, verbose=False)
                    self.current_frame = frame.copy()
                    self.current_predictions = results
                except Exception as e:
                    print(f"Prediction Error: {str(e)}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)
            if self.current_predictions is not None:
                frame = self.draw_detections(frame.copy(), self.current_predictions)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Update the QGraphicsPixmapItem with the new frame
            pixmap = QPixmap.fromImage(qt_image)
            # Scale the pixmap to fill the graphics view
            scaled_pixmap = pixmap.scaled(
                self.graphics_view.width(), self.graphics_view.height(),
                Qt.IgnoreAspectRatio, # Stretch to fill
                Qt.SmoothTransformation
            )
            self.video_item.setPixmap(scaled_pixmap)

    # --- DETECTION, TRACKING, MONITORING ---
    def draw_detections(self, frame, results):
        """Draw bounding boxes, labels, and track IDs on the frame with a modern design, using confidence threshold."""
        frame_width = frame.shape[1]
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x1_flipped = frame_width - x2
                x2_flipped = frame_width - x1
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                if conf < self.confidence_threshold:
                    continue
                class_name = result.names[cls]
                color = self.get_class_color(class_name)
                track_id = None
                if hasattr(box, 'id') and box.id is not None:
                    track_id = int(box.id[0].cpu().numpy())
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1_flipped, y1), (x2_flipped, y2), color, -1)
                alpha = 0.2
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                cv2.rectangle(frame, (x1_flipped, y1), (x2_flipped, y2), color, 2)
                label = f"{class_name}: {conf:.2f}"
                if track_id is not None:
                    label = f"ID {track_id} | {label}"
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                label_x = x1_flipped
                label_y = y1 - 10
                cv2.rectangle(frame, 
                            (label_x - 5, label_y - label_height - 5),
                            (label_x + label_width + 5, label_y + 5),
                            color, -1)
                cv2.putText(frame, label, (label_x, label_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return frame

    def get_class_color(self, class_name):
        if class_name not in self.class_colors:
            hue = random.random()
            saturation = 0.8 + random.random() * 0.2
            value = 0.8 + random.random() * 0.2
            h = int(hue * 6)
            f = hue * 6 - h
            p = value * (1 - saturation)
            q = value * (1 - f * saturation)
            t = value * (1 - (1 - f) * saturation)
            if h == 0:
                r, g, b = value, t, p
            elif h == 1:
                r, g, b = q, value, p
            elif h == 2:
                r, g, b = p, value, t
            elif h == 3:
                r, g, b = p, q, value
            elif h == 4:
                r, g, b = t, p, value
            else:
                r, g, b = value, p, q
            self.class_colors[class_name] = (int(b * 255), int(g * 255), int(r * 255))
        return self.class_colors[class_name]

    # def monitor_predictions(self, frame, results):
    async def monitor_predictions(self, frame, results):
        if not results or not self.deployment:
            return None
        try:
            detection_scores = []
            detection_boxes = []
            detection_classes = []
            speed = 0
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    detection_scores.append(conf)
                    detection_boxes.append([y1, x1, y2, x2])
                    detection_classes.append(cls)
                speed += (result.speed['preprocess']+result.speed['inference']+result.speed['postprocess'])/1000
            predictions = DetectionPredictionFormat(
                detection_scores=detection_scores,
                detection_boxes=detection_boxes,
                detection_classes=detection_classes
            )
            now_timestamp = datetime.datetime.now().timestamp()
            ret, jpeg = cv2.imencode('.jpg', self.current_frame)
            if not ret:
                print("Error: Failed to encode frame to JPEG")
                return None
            jpeg_bytes = jpeg.tobytes()
            self.deployment.monitor_bytes(
                raw_image=jpeg_bytes,
                content_type="image/jpeg",
                prediction=predictions,
                filename=str(uuid.uuid4())+".jpg",
                source="CVPR",
                latency=speed,
                timestamp=datetime.datetime.now().timestamp(),
                height=frame.shape[0],
                width=frame.shape[1],
            )
            return predictions, now_timestamp
        except Exception as e:
            print(f"Monitoring Error: {str(e)}")
            return None

    def predict_frame(self):
        if not self.model or self.current_predictions is None or self.current_frame is None:
            print("Error: No predictions to monitor")
            return
        try:
            if self.client and self.deployment:
                # Create and start the worker thread
                self.worker = PredictionWorker(
                    self.deployment,
                    self.current_frame.copy(),
                    self.current_predictions
                )
                self.worker.finished.connect(self.handle_prediction_finished)
                self.worker.error.connect(self.handle_prediction_error)
                self.worker.start()
        except Exception as e:
            print(f"Prediction Error: {str(e)}")

    def handle_prediction_finished(self, result):
        predictions, timestamp = result
        # Update UI or perform any necessary actions with the results

    def handle_prediction_error(self, error_msg):
        print(f"Prediction Error: {error_msg}")

    def update_confidence_threshold(self, value):
        self.confidence_threshold = value / 100.0
        self.slider_label.setText(f"Confidence: {self.confidence_threshold:.2f}")

    def start_prediction_refresh_timer(self):
        self.prediction_timer = QTimer()
        self.prediction_timer.timeout.connect(self.refresh_predictions)
        self.prediction_timer.start(2000)  # Refresh every 2 seconds

    def refresh_predictions(self):
        if hasattr(self, 'deployment') and self.deployment:
            try:
                predictions = self.deployment.list_predicted_assets(limit=6)
                self.prediction_grid.update_images(predictions)
            except Exception as e:
                print(f"Error refreshing predictions: {str(e)}")

    def start_auto_predict_timer(self):
        self.auto_predict_timer = QTimer()
        self.auto_predict_timer.timeout.connect(self.auto_predict)
        self.auto_predict_timer.start(5000)  # 5000ms = 5 seconds

    def auto_predict(self):
        if not self.deployment or self.current_frame is None:
            return

        self.predict_worker = PredictWorker(self.deployment_latent, self.current_frame.copy())
        self.predict_worker.finished.connect(self.handle_predict_finished)
        self.predict_worker.error.connect(self.handle_predict_error)
        self.predict_worker.start()

    def handle_predict_finished(self, prediction):
        # Handle the prediction result here
        print("Prediction received:", prediction)

    def handle_predict_error(self, error_msg):
        print(f"Prediction Error: {error_msg}")

    def closeEvent(self, event):
        if hasattr(self, 'prediction_timer'):
            self.prediction_timer.stop()
        if hasattr(self, 'auto_predict_timer'):
            self.auto_predict_timer.stop()
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        if hasattr(self, 'predict_worker') and self.predict_worker.isRunning():
            self.predict_worker.terminate()
            self.predict_worker.wait()
        self.camera.release()
        event.accept()

    # --- UI POSITIONING ---
    def position_elements(self):
        # Position graphics view to fill the entire window
        self.graphics_view.setGeometry(0, 0, self.width(), self.height())
        
        # Logo at top left
        logo_x = 20
        logo_y = 20
        self.logo_label.move(logo_x, logo_y)

        # Video source dropdown
        dropdown_x = logo_x + self.logo_label.width() + 10
        dropdown_y = logo_y + (self.logo_label.height() - self.source_dropdown.height()) // 2
        self.source_dropdown.move(dropdown_x, dropdown_y)

        # Position model dropdown to the right of the video source dropdown
        model_dropdown_x = dropdown_x + self.source_dropdown.width() + 10
        model_dropdown_y = dropdown_y
        self.model_dropdown.move(model_dropdown_x, model_dropdown_y)
        
        # Position slider next to model dropdown
        slider_x = model_dropdown_x + self.model_dropdown.width() + 10
        slider_y = model_dropdown_y + (self.model_dropdown.height() - self.slider_container.height()) // 2
        self.slider_container.move(slider_x, slider_y)

        # Stop button at top right
        stop_x = self.width() - self.stop_button.width() - 40
        stop_y = 40
        self.stop_button.move(stop_x, stop_y)

        # Button container to the left of stop button
        button_container_width = self.predict_button.width()
        button_container_height = self.predict_button.height()
        button_container_x = stop_x - button_container_width - 20
        button_container_y = stop_y
        self.button_container.setGeometry(button_container_x, button_container_y, button_container_width, button_container_height)

        # Raise elements (order matters for drawing)
        self.button_container.raise_()
        self.stop_button.raise_()
        self.logo_label.raise_()
        self.slider_container.raise_()
        self.source_dropdown.raise_()
        self.model_dropdown.raise_()
        # Proxy widget is handled by graphics scene, but ensure its parent widget is raised.
        # self.prediction_grid_proxy is an item in the scene, not a direct child of central_widget.
        # Instead, we position the prediction_grid_proxy within the scene.

        # Position prediction grid within the graphics scene
        # Convert scene coordinates to widget coordinates for sizing
        grid_width = self.width()
        grid_height = int(self.height() * 0.2)
        grid_x = 0
        grid_y = self.height() - grid_height
        self.prediction_grid_proxy.setGeometry(QRectF(grid_x, grid_y, grid_width, grid_height))

    def resizeEvent(self, event):
        # Resize graphics view to fill the main window
        self.graphics_view.setGeometry(0, 0, self.width(), self.height())
        # Resize the scene to match the view's size
        self.graphics_scene.setSceneRect(0, 0, self.width(), self.height())
        self.position_elements() # Reposition other UI elements
        super().resizeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_()) 