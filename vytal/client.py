import scipy
import time, requests, numpy as np, base64, torch, cv2, websockets, asyncio, sys, pygame, cv2, json, math, threading, pickle, io, os, ctypes
from typing import Union
from PyQt5.QtWidgets import QApplication
import torch.nn.functional as F
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QPushButton, QLineEdit, QCheckBox, QMessageBox
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QPen, QImage
from PyQt5.QtCore import Qt, QTimer, QRect, QPoint

def numpy_to_bytes(array):
    metadata = {
        'dtype': str(array.dtype),
        'shape': array.shape
    }
    data = array.tobytes()
    metadata_encoded = base64.b64encode(str(metadata).encode('utf-8')).decode('utf-8')
    data_encoded = base64.b64encode(data).decode('utf-8')
    return {"metadata": metadata_encoded, "data": data_encoded}

def bytes_to_numpy(data):
    metadata_bstring = data['metadata']
    data_bstring = data['data']
    metadata_decoded = eval(base64.b64decode(metadata_bstring).decode('utf-8'))
    data_decoded = base64.b64decode(data_bstring)
    array = np.frombuffer(data_decoded, dtype=metadata_decoded['dtype']).reshape(metadata_decoded['shape'])
    return array

def tensor_to_bytes(tensor):
    # Convert the PyTorch dtype to a string format that NumPy understands
    numpy_dtype_str = str(tensor.numpy().dtype)
    
    metadata = {
        'dtype': numpy_dtype_str,
        'shape': tensor.shape
    }
    data = tensor.numpy().tobytes()
    metadata_encoded = base64.b64encode(str(metadata).encode('utf-8')).decode('utf-8')
    data_encoded = base64.b64encode(data).decode('utf-8')
    return {"metadata": metadata_encoded, "data": data_encoded}

def bytes_to_tensor(data):
    metadata_bstring = data['metadata']
    data_bstring = data['data']
    metadata_decoded = eval(base64.b64decode(metadata_bstring).decode('utf-8'))
    data_decoded = base64.b64decode(data_bstring)
    
    # Convert the dtype string back into a NumPy dtype
    numpy_dtype = np.dtype(metadata_decoded['dtype'])
    
    array = np.frombuffer(data_decoded, dtype=numpy_dtype).reshape(metadata_decoded['shape'])
    tensor = torch.from_numpy(array)
    return tensor

def convert_dict_values_to_bytes(d):
    if d is None: return None

    result = {}
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            result[key] = numpy_to_bytes(value)
        elif isinstance(value, torch.Tensor):
            result[key] = tensor_to_bytes(value)
        else:
            result[key] = value  # Leave other types unchanged
    return result

def convert_dict_values_to_tensor(d):
    result = {}
    for key, value in d.items():
        result[key] = bytes_to_tensor(value)
    return result

def convert_dict_values_to_numpy(d):
    result = {}
    for key, value in d.items():
        result[key] = bytes_to_numpy(value)
    return result

class TextInput(QLineEdit):
    def __init__(self, x, y, width, height, screen_width, color="dodgerblue2"):
        super().__init__()
        self.setGeometry(x, y, width, height)
        self.setStyleSheet(f"background-color: white; border: 3px solid {color};")
        font = QFont("Inter", int(48 * screen_width / 1500))
        self.setFont(font)

class Button(QPushButton):
    def __init__(self, x, y, width, height, screen_width, text, color):
        super().__init__(text)
        self.setGeometry(x, y, width, height)
        self.original_text = text
        self.color = QColor(color)
        self.hover_color = self.color.lighter(130)
        font = QFont("Inter", int(48 * screen_width / 1500))
        self.setFont(font)
        self.setStyleSheet(f"background-color: {color}; color: #04545c; border-radius: {int(20 * screen_width / 1500)}px;")

    def enterEvent(self, event):
        self.setStyleSheet(f"background-color: {self.hover_color.name()}; color: white; border-radius: {int(20 * self.width() / 75)}px;")

    def leaveEvent(self, event):
        self.setStyleSheet(f"background-color: {self.color.name()}; color: #04545c; border-radius: {int(20 * self.width() / 75)}px;")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setText("Starting...")

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setText(self.original_text)
            self.clicked.emit()

class Checkbox(QCheckBox):
    def __init__(self, x, y, width, height):
        super().__init__()
        self.setGeometry(x, y, width, height)
        self.setStyleSheet("QCheckBox::indicator { width: 30px; height: 30px; }")

class CalibrationInputWindow(QMainWindow):
    def __init__(self, client, bg_offset_x, bg_offset_y, screen_width, screen_height, background, dynamic_photo_region, text_box_1_region, text_box_2_region,
                 begin_button_region, checkbox_region):
        super().__init__()
        self.client = client
        self.background = background
        self.dynamic_photo_region = dynamic_photo_region
        self.text_box_1_region = text_box_1_region
        self.text_box_2_region = text_box_2_region
        self.begin_button_region = begin_button_region
        self.checkbox_region = checkbox_region

        self.setGeometry(0, 0, screen_width, screen_height)
        self.setWindowTitle("Calibration Input")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.bg_offset_x = bg_offset_x
        self.bg_offset_y = bg_offset_y

        style = """
        QLineEdit {
            background-color: white;
            border: 2px solid #04545c;
            border-radius: 5px;
            padding: 5px;
            font-size: 16px;
        }
        QLineEdit:hover {
            border: 2px solid #067c87;
        }
        QLineEdit:focus {
            border: 2px solid #079eab;
            background-color: #f0f8ff;
        }
        QPushButton {
            background-color: #04545c;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
        }
        QPushButton:hover {
            background-color: #067c87;
        }
        QPushButton:pressed {
            background-color: #079eab;
        }
        QCheckBox {
            spacing: 5px;
        }
        QCheckBox::indicator {
            width: 20px;
            height: 20px;
        }
        QCheckBox::indicator:unchecked {
            border: 2px solid #04545c;
            background-color: white;
        }
        QCheckBox::indicator:unchecked:hover {
            border: 2px solid #067c87;
        }
        QCheckBox::indicator:checked {
            border: 2px solid #04545c;
            background-color: #04545c;
        }
        QCheckBox::indicator:checked:hover {
            border: 2px solid #067c87;
            background-color: #067c87;
        }
        """
        self.setStyleSheet(style)

        self.text_input1 = QLineEdit(self)
        self.text_input1.setGeometry(self.text_box_1_region[0][0], self.text_box_1_region[0][1],
                                     self.text_box_1_region[1][0] - self.text_box_1_region[0][0],
                                     self.text_box_1_region[1][1] - self.text_box_1_region[0][1])

        self.text_input2 = QLineEdit(self)
        self.text_input2.setGeometry(self.text_box_2_region[0][0], self.text_box_2_region[0][1],
                                     self.text_box_2_region[1][0] - self.text_box_2_region[0][0],
                                     self.text_box_2_region[1][1] - self.text_box_2_region[0][1])

        self.begin_button = QPushButton("Begin", self)
        self.begin_button.setGeometry(self.begin_button_region[0][0], self.begin_button_region[0][1],
                                      self.begin_button_region[1][0] - self.begin_button_region[0][0],
                                      self.begin_button_region[1][1] - self.begin_button_region[0][1])

        self.checkbox = QCheckBox(self)
        self.checkbox.setGeometry(self.checkbox_region[0][0], self.checkbox_region[0][1],
                                  self.checkbox_region[1][0] - self.checkbox_region[0][0],
                                  self.checkbox_region[1][1] - self.checkbox_region[0][1])

        self.begin_button.clicked.connect(self.on_begin)

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(33)  # ~30 FPS

        self.input1 = None
        self.input2 = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.bg_offset_x, self.bg_offset_y, self.background)

        if len(self.client.preds) > 0:
            latest_pred = self.client.preds[-1]
            if 'left_eye_patch' in latest_pred and 'right_eye_patch' in latest_pred:
                left_eye_pixmap = tensor_to_pixmap(latest_pred['left_eye_patch'])
                right_eye_pixmap = tensor_to_pixmap(latest_pred['right_eye_patch'])

                combined_width = left_eye_pixmap.width() + right_eye_pixmap.width()
                combined_height = max(left_eye_pixmap.height(), right_eye_pixmap.height())
                combined_pixmap = QPixmap(combined_width, combined_height)
                combined_pixmap.fill(Qt.transparent)

                temp_painter = QPainter(combined_pixmap)
                temp_painter.drawPixmap(0, 0, left_eye_pixmap)
                temp_painter.drawPixmap(left_eye_pixmap.width(), 0, right_eye_pixmap)
                temp_painter.end()

                scaled_pixmap = combined_pixmap.scaled(self.dynamic_photo_region[1][0] - self.dynamic_photo_region[0][0],
                                                       self.dynamic_photo_region[1][1] - self.dynamic_photo_region[0][1],
                                                       Qt.KeepAspectRatio)
                painter.drawPixmap(self.dynamic_photo_region[0][0], self.dynamic_photo_region[0][1], scaled_pixmap)

    def update_display(self):
        self.update()

    def on_begin(self):
        if self.text_input1.text() and self.text_input2.text() and self.checkbox.isChecked():
            self.input1 = self.text_input1.text()
            self.input2 = self.text_input2.text()
            self.close()
        else:
            print("User did not input all information")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

class CalibrationScreen(QWidget):
    def __init__(self, client, screen_width, screen_height):
        super().__init__()
        self.client = client
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.setGeometry(0, 0, screen_width, screen_height)
        self.setWindowTitle("Calibration")

        self.dots = calculate_dot_positions(screen_width, screen_height)
        self.current_dot_index = 0
        self.angle = 0
        self.spin_speed = 0.15
        self.results = []

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_calibration)
        self.timer.start(33)  # ~30 FPS

        self.start_time = time.time()
        self.last_complete = len(self.client.preds)
        time.sleep(1)
        self.window = 2 * (len(self.client.preds) - self.last_complete)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.white)

        if hasattr(self, 'break_time') and self.break_time > 0:
            font = QFont("Inter", 36)
            painter.setFont(font)
            text = f"Break time: {self.break_time} second{'s' if self.break_time > 1 else ''}"
            painter.drawText(self.rect(), Qt.AlignCenter, text)
        elif self.current_dot_index < len(self.dots):
            dot = self.dots[self.current_dot_index]
            self.draw_spinning_triangle(painter, dot, 15, self.angle)

    def draw_spinning_triangle(self, painter, position, size, angle):
        triangle_shape = [
            QPoint(0, -size),
            QPoint(int(size * math.sqrt(3) / 2), int(size / 2)),
            QPoint(int(-size * math.sqrt(3) / 2), int(size / 2))
        ]
        painter.save()
        painter.translate(position[0], position[1])
        painter.rotate(math.degrees(angle))
        painter.setBrush(Qt.black)
        painter.drawPolygon(triangle_shape)
        painter.restore()

    def update_calibration(self):
        self.angle += self.spin_speed
        self.spin_speed = min(self.spin_speed + 0.01, 0.4)

        if self.current_dot_index < len(self.dots):
            if len(self.client.preds) - self.last_complete >= self.window:
                stable, stats = is_gaze_stable(self.client.preds[-self.window:])
                if stable:
                    predictions = [{k: v.tolist() for k, v in p.items() if 'eye_patch' not in k} for p in self.client.preds[-self.window:]]
                    self.results.append({
                        "dot": self.dots[self.current_dot_index],
                        "predictions": predictions,
                        "stats": stats,
                    })
                    self.current_dot_index += 1
                    if self.current_dot_index == 8:
                        self.start_break()
                    self.last_complete = len(self.client.preds)
                    self.angle = 0
                    self.spin_speed = 0.15
                    self.start_time = time.time()

            if time.time() - self.start_time > 10:
                print(f"Calibration failed on dot {self.dots[self.current_dot_index]}")
                self.current_dot_index += 1
                if self.current_dot_index == 8:
                    self.start_break()
                self.last_complete = len(self.client.preds)
                self.angle = 0
                self.spin_speed = 0.15
                self.start_time = time.time()

        elif self.current_dot_index == len(self.dots):
                self.finish_calibration()

        self.update()

    def start_break(self):
        self.timer.stop()
        self.break_time = 5
        self.break_timer = QTimer(self)
        self.break_timer.timeout.connect(self.update_break)
        self.break_timer.start(1000)

    def update_break(self):
        self.break_time -= 1
        if self.break_time <= 0:
            self.break_timer.stop()
            self.last_complete = len(self.client.preds)
            self.angle = 0
            self.spin_speed = 0.15
            self.start_time = time.time()
            self.timer.start()
        self.update()

    def finish_calibration(self):
        self.timer.stop()
        self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.timer.stop()
            self.close()

def tensor_to_pixmap(tensor):
    np_array = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    np_array = (np_array * 255).astype(np.uint8)
    # np_array = np_array.astype(np.uint8)
    # np_array = np.rot90(np_array, k=-3)
    height, width, channel = np_array.shape
    bytes_per_line = 3 * width

    # Create a copy of the array to ensure it's contiguous in memory
    np_array = np.ascontiguousarray(np_array)

    q_image = QImage(np_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(q_image)

def calculate_dot_positions(width, height):
    x_positions = np.linspace(0.1 * width, 0.8 * width, 4)
    y_positions = np.linspace(0.1 * height, 0.8 * height, 4)
    return [(int(x), int(y)) for y in y_positions for x in x_positions]

def is_gaze_stable(predictions_window, threshold=3):
    # if len(predictions_window) < 60:
    #     return False, {}

    if not predictions_window:
        return False, {}
    
    original_length = len(predictions_window)
    predictions_window = [p for p in predictions_window if not (p['left'].isnan().any() or p['right'].isnan().any())]

    if original_length - len(predictions_window) > 10:
        return False, {}

    lefts = np.squeeze(np.array([p['left'].numpy() for p in predictions_window]))
    rights = np.squeeze(np.array([p['right'].numpy() for p in predictions_window]))

    if lefts.shape[-1] != 2 or rights.shape[-1] != 2 or lefts.ndim != 2 or rights.ndim != 2:
        return False, {}

    angles = (lefts + rights) / 2

    avg_face_angle = np.nanmean(angles, axis=0)

    if np.isnan(avg_face_angle).any():
            return False, {}

    def pitchyaw_to_vector(py: torch.Tensor) -> torch.Tensor:
        # Converts pitch and yaw angles (n x 2 x 1) to a 3D direction vector (polar to cartesian) (n x 3 x 1).
        if (py.shape[1] != 2):
            raise ValueError('Input tensor must have 2 columns, not %d' % py.shape[1])

        sin, cos = torch.sin(py), torch.cos(py)
        direction = torch.stack([cos[:, 0] * sin[:, 1],
                                 sin[:, 0],
                                 cos[:, 0] * cos[:, 1]],
                                dim=1)
        return direction

    def angular_error(a, b):
        try:
            a = torch.tensor(a, dtype=torch.float32).reshape((1, 2, 1))
            b = torch.tensor(b, dtype=torch.float32).reshape((1, 2, 1))
        except Exception as e:
            print(f"Error reshaping tensors: {e}")
            return float('inf')  # Return a large value to indicate error
        a = pitchyaw_to_vector(a)
        b = pitchyaw_to_vector(b)
        sim = F.cosine_similarity(a, b, dim=1, eps=1e-8)
        sim = F.hardtanh_(sim, min_val=-1 + 1e-8, max_val=1 - 1e-8)
        return (torch.acos(sim) * 180. / np.pi).item()

    angular_diffs = [angular_error(avg_face_angle, angle) for angle in angles]
    mean_angular_diff = np.mean(angular_diffs)
    median_angular_diff = np.median(angular_diffs)
    std_angular_diff = np.std(angular_diffs)

    stats = {
        "mean": mean_angular_diff,
        "median": median_angular_diff,
        "std": std_angular_diff
    }

    return (mean_angular_diff < threshold), stats

def process_calib_type(calib):
    if type(calib) == str:
        with open(calib, 'rb') as f:
            calib = pickle.load(f)
            b = io.BytesIO()
            pickle.dump(calib, b)
            calib = str(b.getvalue())
    elif type(calib) == bytes:
        calib = str(calib)
    elif type(calib) == scipy.interpolate._rbfinterp.RBFInterpolator:
        b = io.BytesIO()
        pickle.dump(calib, b)
        calib = str(b.getvalue())

    return calib

class Client:
    def __init__(self, api_key: str, ipd: float = None):
        self.api_key = api_key
        self.ipd = ipd
        self.cap = None
        self.websocket = None
        self.frame = None
        self.preds = []
    
    def calibrate(self, save_directory: str = None):
        # video_path, w, h, calibration_points, time_slices = calibration_gui()
        app = QApplication(sys.argv)
        screen = app.primaryScreen().geometry()
        screen_width, screen_height = screen.width(), screen.height()
        
        vytal_api_loop = self.start_thread(eye_frames=True)
        original_bg = QPixmap(os.path.join(os.path.dirname(__file__), "background.png"))
        bg_width, bg_height = original_bg.width(), original_bg.height()
        scale_factor = min(screen_width / bg_width, screen_height / bg_height)
        new_bg_width, new_bg_height = int(bg_width * scale_factor), int(bg_height * scale_factor)
        background = original_bg.scaled(new_bg_width, new_bg_height, Qt.KeepAspectRatio)
        bg_offset_x, bg_offset_y = (screen_width - new_bg_width) // 2, (screen_height - new_bg_height) // 2
        
        def scale_region(region):
            return (
                (int(region[0][0] * scale_factor) + bg_offset_x,
                 int(region[0][1] * scale_factor) + bg_offset_y),
                (int(region[1][0] * scale_factor) + bg_offset_x,
                 int(region[1][1] * scale_factor) + bg_offset_y)
            )
        
        dynamic_photo_region = scale_region(((2240, 350), (3628, 975)))
        text_box_1_region = scale_region(((3092, 1120), (3587, 1306)))
        text_box_2_region = scale_region(((3092, 1432), (3587, 1618)))
        begin_button_region = scale_region(((2240, 1710), (2995, 1910)))
        checkbox_region = scale_region(((3110, 159), (3230, 261)))
        
        input_window = CalibrationInputWindow(self, bg_offset_x, bg_offset_y, screen_width, screen_height, background,
                                              dynamic_photo_region, text_box_1_region, text_box_2_region,
                                              begin_button_region, checkbox_region)
        input_window.show()

        app.exec_()

        if input_window.input1 is None or input_window.input2 is None:
            self.end_thread(vytal_api_loop)
            return

        calibration_window = CalibrationScreen(self, screen_width, screen_height)
        calibration_window.show()

        app.exec_()

        results = {"input1": input_window.input1, "input2": input_window.input2, "calibration_results": calibration_window.results}

        self.end_thread(vytal_api_loop)

        self.preds = []
        self.cap = None
        self.websocket = None
        self.frame = None

        function_endpoint = "video/calibrate"
        params = {"api_key": self.api_key}
        response = requests.post(
            f'http://ec2-54-208-48-146.compute-1.amazonaws.com:8000/{function_endpoint}',
            params=params,
            data=json.dumps(results),
            timeout=1000,
        )

        res = response.json()

        if save_directory is not None and "data" in res:
            save_path = os.path.join(save_directory, "calibration.pkl")
            with open(save_path, 'wb') as f:
                f.write(eval(res["data"]))

        if "data" in res:
            b = io.BytesIO()
            b.write(eval(res["data"]))
            b.seek(0)
            return pickle.load(b)
        else:
            return res
                   
    def predict_from_video(self, video_path: str, calib: Union[scipy.interpolate._rbfinterp.RBFInterpolator, str, bytes] = None, eye_frames: bool = False):    
        function_endpoint = "video/handle_video"

        # Check if the video file exists
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"No such file or directory: '{video_path}'")
        
        if calib is not None:
            try:
                calib = process_calib_type(calib)
            except Exception as e:
                print(f"Error processing calibration data: {e}")
                return None

        with open(video_path, 'rb') as f:
            params = {"api_key": self.api_key}
            if calib is not None: params["calib_rbf"] = calib
            if self.ipd is not None:
                params["ipd"] = "%.3f" % self.ipd
            params["eye_frames"] = eye_frames

            try:
                response = requests.post(
                    f'http://ec2-54-208-48-146.compute-1.amazonaws.com:8000/{function_endpoint}',
                    params=params,
                    data=f.read(),
                    timeout=1000,
                )
                response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                return None
            
            try:
                result = {}
                for key, value in response.json().items():
                    result[key] = bytes_to_tensor(value)
            except:
                result = response.json()

        return result
    
    async def init_websocket(self, cam_id: int = 0, calib: Union[scipy.interpolate._rbfinterp.RBFInterpolator, str, bytes] = None, eye_frames: bool = False):
        if calib is not None:
            calib = process_calib_type(calib)

        function_endpoint = f"ws://ec2-54-208-48-146.compute-1.amazonaws.com:8000/ws/predict?api_key={self.api_key}&eye_frames={eye_frames}"
        # function_endpoint = f"ws://127.0.0.1:8000/ws/predict?api_key={self.api_key}"
        if calib is not None: function_endpoint += f"&calib_rbf={calib}"
        if self.ipd is not None: function_endpoint += "&ipd=%.3f" % self.ipd
        self.websocket = await websockets.connect(function_endpoint)
        self.cap = cv2.VideoCapture(cam_id)
        
    async def close_websocket(self):
        await self.websocket.close()
        self.cap.release()
        cv2.destroyAllWindows()
    
    async def send_websocket_frame(self, show_frame: bool = False, verbose: bool = False):
        while True:
            ret, frame = self.cap.read()
            self.frame = frame
            if not ret:
                break

            _, buffer = cv2.imencode('.jpg', frame)
            image_bytes = base64.b64encode(buffer).decode('utf-8')
                        
            await self.websocket.send(str(image_bytes) + "==abc==")
            
            response = await self.websocket.recv()
            response = convert_dict_values_to_tensor(eval(response))

            if show_frame:
                cv2.imshow('Live Stream', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                if verbose:
                    print(f"Response from server: {response}")
                    print()
                
            return response
    
    def start_thread(self, cam_id: int = 0, calib: Union[scipy.interpolate._rbfinterp.RBFInterpolator, str, bytes] = None, verbose: bool = False, show_frame: bool = False, eye_frames: bool = False):
        self.preds = []
        async def main():
            try:
                await self.init_websocket(cam_id, calib, eye_frames)
                # print("WebSocket connection established.")
            except Exception as e:
                print("Failed to connect to WebSocket server:", e)
                await self.close_websocket()
                return
            
            while True:
                if not self.cap or not self.websocket:
                    continue
                try:
                    pred = await self.send_websocket_frame(show_frame, verbose)
                    self.preds.append(pred)
                except Exception as e:
                    await self.close_websocket()
                    break

        def loop_in_thread(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        loop = asyncio.get_event_loop()
        t = threading.Thread(target=loop_in_thread, args=(loop,))
        t.start()

        task = asyncio.run_coroutine_threadsafe(main(), loop)
        while not self.preds:
            continue
        
        return loop

    def end_thread(self, loop):
        tasks = asyncio.all_tasks(loop)
        for t in tasks:
            t.cancel()
        loop.stop()

    async def predict_from_websocket(self, cam_id: int = 0, calib: Union[scipy.interpolate._rbfinterp.RBFInterpolator, str, bytes] = None, verbose: bool = False, show_frame: bool = False):
        function_endpoint = f"ws://ec2-54-208-48-146.compute-1.amazonaws.com:8000/ws/predict?api_key={self.api_key}"
        # function_endpoint = f"ws://ec2-54-208-48-146.compute-1.amazonaws.com:5000/ws/predict?api_key={self.api_key}"
        # function_endpoint = f"ws://127.0.0.1:8000/ws/predict?api_key={self.api_key}"
        if calib is not None:
            calib = process_calib_type(calib)
            function_endpoint += f"&calib_rbf={calib}"
        if self.ipd is not None: function_endpoint += "&ipd=%.3f" % self.ipd
        start_time = time.time()
        self.preds = []
        try:
            async with websockets.connect(function_endpoint) as websocket:
                print("WebSocket connection established")
                cap = cv2.VideoCapture(cam_id) 
                start_time = time.time()
                print("Opened camera feed")

                async def send_frames():
                    try:
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            _, buffer = cv2.imencode('.jpg', frame)
                            image_bytes = base64.b64encode(buffer).decode('utf-8')

                            await websocket.send(str(image_bytes) + "==abc==")

                            if show_frame:
                                cv2.imshow('Live Stream', frame)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                    finally:
                        cap.release()
                        cv2.destroyAllWindows()

                async def receive_responses():
                    while True:
                        response = await websocket.recv()
                        response = convert_dict_values_to_tensor(eval(response))
                        self.preds.append(response)
                        
                        if verbose:
                            print(f"Response from server: {response}")
                            print(f"Time per frame for {len(self.preds)} frames: {(time.time() - start_time) / len(self.preds):.3f} seconds")
                            print()

                await asyncio.gather(send_frames(), receive_responses())
        except Exception as e:
            print(f"An error occurred: {e}")

        return self.preds
    
    def real_time_pred(self, cam_id: int = 0, calib: Union[scipy.interpolate._rbfinterp.RBFInterpolator, str, bytes] = None, verbose: bool = False, show_frame: bool = False):
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(self.predict_from_websocket(cam_id, calib, verbose, show_frame))
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            loop.close()
        
        return self.preds
