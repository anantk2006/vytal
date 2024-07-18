import os
import time
import requests
import numpy as np
import base64
import torch
import cv2
import pygame
from datetime import datetime
import websockets
import asyncio
import sys
from PyQt5.QtWidgets import QApplication
import threading

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

def calibration_gui():
    # Initialize pygame
    pygame.init()

    # Set up the screen for fullscreen display
    infoObject = pygame.display.Info()
    screen_width, screen_height = infoObject.current_w, infoObject.current_h
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
    pygame.display.set_caption("Calibration")

    # Set colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)

    # Define calibration points
    calibration_points = [
        (screen_width // 2, screen_height // 2),  # Center
        (screen_width // 2, 25),  # Top center near edge
        (screen_width - 25, 25),  # Top right near edge
        (screen_width - 25, screen_height // 2),  # Middle right near edge
        (screen_width - 25, screen_height - 25),  # Bottom right near edge
        (screen_width // 2, screen_height - 25),  # Bottom center near edge
        (25, screen_height - 25),  # Bottom left near edge
        (25, screen_height // 2),  # Middle left near edge
        (25, 25)  # Top left near edge
    ]

    # Set up camera
    cap = cv2.VideoCapture(0)  # Adjust the device index based on your camera

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Video recording setup
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('calibration.mp4', fourcc, 20.0, (640, 480))

    current_point_index = 0
    is_clicked = False
    click_times = []
    click_frames = []
    clicked_points = []
    start_time = datetime.now()

    # Game loop
    running = True
    clock = pygame.time.Clock()

    frame_count = 0
    gt = []
    frame_times = []

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:  # Press ESC to quit
                    running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_x, mouse_y = event.pos
                    point_x, point_y = calibration_points[current_point_index]
                    distance = np.sqrt((point_x - mouse_x) ** 2 + (point_y - mouse_y) ** 2)
                    if distance < 25:  # Sensitivity radius
                        is_clicked = True
                        clicked_time = datetime.now()
                    else:
                        is_clicked = False
                    
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and is_clicked:
                    elapsed_time = (datetime.now() - clicked_time).total_seconds()
                    if elapsed_time >= 3:
                        click_times.append(int((datetime.now() - start_time).total_seconds() * 1000))
                        clicked_points.append(calibration_points[current_point_index])
                        click_frames.append(frame_count)
                        current_point_index = (current_point_index + 1) % len(calibration_points)
                    is_clicked = False

        # Webcam frame capture
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        out.write(frame)  # Write frame to video file

        # Clear screen
        screen.fill(BLACK)

        # Draw the current calibration point
        if current_point_index < len(calibration_points) and len(click_times) < len(calibration_points):
            point_x, point_y = calibration_points[current_point_index]
            if is_clicked:
                size = 24*(1 - (int((datetime.now() - clicked_time).total_seconds()*1000) % 3000) / 3000)
                pygame.draw.circle(screen, GREEN, (point_x, point_y), size)
                if is_clicked:
                    elapsed_time = (datetime.now() - clicked_time).total_seconds()
                    if elapsed_time >= 3:
                        click_times.append(int((datetime.now() - start_time).total_seconds() * 1000))
                        clicked_points.append(calibration_points[current_point_index])
                        click_frames.append(frame_count)
                        current_point_index = (current_point_index + 1) % len(calibration_points)
                        is_clicked = False
            else:
                pygame.draw.circle(screen, RED, (point_x, point_y), 25)  # Larger circle for easier targeting
        else:
            running = False

        pygame.display.flip()
        clock.tick(30)
        frame_count += 1
        gt.append(calibration_points[current_point_index])
        frame_times.append(int((datetime.now() - start_time).total_seconds() * 1000))

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    pygame.quit()
    
    ranges = [] # list of tuples (start_frame, end_frame) for each point
    ratio = frame_count / click_times[-1]
    for i in range(len(click_frames)):
        start_frame = int((click_times[i] - 3000) * ratio)
        ranges.append((start_frame, click_frames[i]))
    
    # update calibration.mp4 to only include the frames from start_frame to end_frame for each point
    cap = cv2.VideoCapture('calibration.mp4')
    frames = []
    
    frame_num = 0
    # dataPoints = []
    time_slices = []
    for i in range(len(ranges)):
        start_frame, end_frame = ranges[i]
        while frame_num < start_frame:
            ret, frame = cap.read()
            frame_num += 1
        while frame_num <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            # dataPoints.append(
            #     {
            #         "PoT": [gt[i][0], gt[i][1]],
            #         "time": frame_times[frame_num]
            #     }
            # )
            frame_num += 1
        time_slices.append(len(frames))
    
    cap.release()
    out = cv2.VideoWriter('calibration.mp4', fourcc, 20.0, (640, 480))
    for frame in frames:
        out.write(frame)
    out.release()
    
    del cap, out, frames, frame
    
    return "calibration.mp4", screen_width, screen_height, calibration_points, time_slices

class Client:
    def __init__(self, api_key: str, ipd: float = None):
        self.api_key = api_key
        self.ipd = ipd
        self.cap = None
        self.websocket = None
    
    def calibrate(self):
        video_path, w, h, calibration_points, time_slices = calibration_gui()
        app = QApplication(sys.argv)
        screen = app.screens()[0]
        dpi = screen.physicalDotsPerInch()
        app.quit()

        function_endpoint = "video/calibrate"
        with open(video_path, 'rb') as f:
            params = {"api_key": self.api_key, "width": w, "height": h, "points": str(calibration_points), "time_slices": str(time_slices), "dpi": "%.3f" % dpi}
            if self.ipd is not None:
                params["ipd"] = "%.3f" % self.ipd
            response = requests.post(
                f'http://ec2-54-208-48-146.compute-1.amazonaws.com:8000/{function_endpoint}',
                # f'http://127.0.0.1:8000/{function_endpoint}',
                params=params,
                data=f.read(),
                timeout=1000, 
            )
            
            try:
                result = bytes_to_tensor(response.json())
            except:
                result = response.json()
                
        try:
            # os.remove(video_path)
            pass
        except:
            pass
        
        return result
                   
    def predict_from_video(self, video_path: str, calib_mat: torch.Tensor = None, eye_frames: bool = False):    
        function_endpoint = "video/handle_video"

        with open(video_path, 'rb') as f:
            calib_mat_bytes = tensor_to_bytes(calib_mat) if calib_mat is not None else None
            params = {"api_key": self.api_key} if calib_mat is None else {"api_key": self.api_key, "calib_mat": str(calib_mat_bytes)}
            if self.ipd is not None:
                params["ipd"] = "%.3f" % self.ipd
            params["eye_frames"] = eye_frames
            response = requests.post(
                # f'http://ec2-54-208-48-146.compute-1.amazonaws.com:8000/{function_endpoint}',
                f'http://ec2-54-208-48-146.compute-1.amazonaws.com:8000/{function_endpoint}',
                # f'http://127.0.0.1:8000/{function_endpoint}',
                params=params,
                # files = {"file": f},
                data=f.read(),
                timeout=1000,  
            )
            
            try:
                result = {}
                for key, value in response.json().items():
                    result[key] = bytes_to_tensor(value)
            except:
                result = response.json()

        return result
    
    async def init_websocket(self, cam_id: int = 0, calib_mat: np.array = None, eye_frames: bool = False):
        calib_mat_bytes = tensor_to_bytes(torch.from_numpy(calib_mat)) if calib_mat is not None else None
        function_endpoint = f"ws://ec2-54-208-48-146.compute-1.amazonaws.com:8000/ws/predict?api_key={self.api_key}&eye_frames={eye_frames}"
        # function_endpoint = f"ws://127.0.0.1:8000/ws/predict?api_key={self.api_key}"
        if calib_mat is not None: function_endpoint += f"&calib_mat={str(calib_mat_bytes)}"
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
    
    def start_thread(self, cam_id: int = 0, calib_mat: np.array = None, verbose: bool = False, show_frame: bool = False, eye_frames: bool = False):
        self.preds = []
        async def main():
            await self.init_websocket(cam_id, calib_mat, eye_frames)
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

    async def predict_from_websocket(self, cam_id: int = 0, calib_mat: np.array = None, verbose: bool = False, show_frame: bool = False):
        calib_mat_bytes = tensor_to_bytes(torch.from_numpy(calib_mat)) if calib_mat is not None else None
        function_endpoint = f"ws://ec2-54-208-48-146.compute-1.amazonaws.com:8000/ws/predict?api_key={self.api_key}"
        # function_endpoint = f"ws://ec2-54-208-48-146.compute-1.amazonaws.com:5000/ws/predict?api_key={self.api_key}"
        # function_endpoint = f"ws://127.0.0.1:8000/ws/predict?api_key={self.api_key}"
        if calib_mat is not None: function_endpoint += f"&calib_mat={str(calib_mat_bytes)}"
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
    
    def real_time_pred(self, cam_id: int = 0, calib_mat: np.array = None, verbose: bool = False, show_frame: bool = False):
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(self.predict_from_websocket(cam_id, calib_mat, verbose, show_frame))
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            loop.close()
        
        return self.preds
from math import sqrt
from statistics import mean
import numpy as np
class hci:
    def fixation_detection(gaze_points, distance_threshold=30, time_threshold=1.5):
        fixations = []
        current_fixation = []
        
        for i, point in enumerate(gaze_points):
            x, y, timestamp = point
            
            if not current_fixation:
                current_fixation.append(point)
                continue
            
            # Calculate centroid of current fixation
            centroid_x = mean(p[0] for p in current_fixation)
            centroid_y = mean(p[1] for p in current_fixation)
            
            # Calculate distance from current point to centroid
            distance = sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
            
            if distance <= distance_threshold:
                current_fixation.append(point)
            else:
                # Check if the current fixation meets the time threshold
                fixation_duration = current_fixation[-1][2] - current_fixation[0][2]
                if fixation_duration >= time_threshold:
                    fixation_centroid = (centroid_x, centroid_y)
                    fixations.append((fixation_centroid, fixation_duration))
                
                # Start a new fixation with the current point
                current_fixation = [point]
        
        # Check if the last fixation meets the time threshold
        if current_fixation:
            fixation_duration = current_fixation[-1][2] - current_fixation[0][2]
            if fixation_duration >= time_threshold:
                centroid_x = mean(p[0] for p in current_fixation)
                centroid_y = mean(p[1] for p in current_fixation)
                fixation_centroid = (centroid_x, centroid_y)
                fixations.append((fixation_centroid, fixation_duration))
        
        return fixations


    def saccade_detection(gaze_points, velocity_threshold=1000):
        saccades = []
        current_saccade = None
        
        for i in range(1, len(gaze_points)):
            x1, y1, t1 = gaze_points[i-1]
            x2, y2, t2 = gaze_points[i]
            
            # Calculate distance between consecutive points
            distance = sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Calculate time difference in seconds
            time_diff = (t2 - t1) / 1000  # Convert milliseconds to seconds
            
            # Calculate velocity in pixels per second
            if time_diff > 0:
                velocity = distance / time_diff
            else:
                velocity = 0
            
            # Check if velocity exceeds the threshold
            if velocity >= velocity_threshold:
                if current_saccade is None:
                    # Start a new saccade
                    current_saccade = {
                        'start_point': gaze_points[i-1],
                        'end_point': gaze_points[i],
                        'duration': t2 - t1,
                        'amplitude': distance,
                        'velocities': [velocity]
                    }
                else:
                    # Continue the current saccade
                    current_saccade['end_point'] = gaze_points[i]
                    current_saccade['duration'] = gaze_points[i][2] - current_saccade['start_point'][2]
                    current_saccade['amplitude'] += distance
                    current_saccade['velocities'].append(velocity)
            else:
                if current_saccade is not None:
                    # End the current saccade
                    current_saccade['peak_velocity'] = max(current_saccade['velocities'])
                    current_saccade['average_velocity'] = sum(current_saccade['velocities']) / len(current_saccade['velocities'])
                    saccades.append(current_saccade)
                    current_saccade = None
        
        # Add the last saccade if it's still open
        if current_saccade is not None:
            current_saccade['peak_velocity'] = max(current_saccade['velocities'])
            current_saccade['average_velocity'] = sum(current_saccade['velocities']) / len(current_saccade['velocities'])
            saccades.append(current_saccade)
        
        return saccades


    def detect_smooth_pursuit(gaze_points, time_window=100, velocity_threshold=30, direction_threshold=30):
        """
        Detect smooth pursuit in a sequence of gaze points.
        
        :param gaze_points: List of tuples (x, y, timestamp)
        :param time_window: Time window in milliseconds to consider for smooth pursuit
        :param velocity_threshold: Maximum velocity (pixels/second) to be considered smooth pursuit
        :param direction_threshold: Maximum direction change (degrees) to be considered smooth pursuit
        :return: List of smooth pursuit segments (start_index, end_index, duration)
        """
        smooth_pursuits = []
        n = len(gaze_points)
        
        def calculate_velocity(p1, p2):
            x1, y1, t1 = p1
            x2, y2, t2 = p2
            distance = sqrt((x2 - x1)**2 + (y2 - y1)**2)
            time_diff = (t2 - t1) / 1000  # Convert to seconds
            return distance / time_diff if time_diff > 0 else 0
        
        def calculate_direction(p1, p2):
            x1, y1, _ = p1
            x2, y2, _ = p2
            return np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        
        start_index = 0
        while start_index < n - 1:
            end_index = start_index + 1
            prev_direction = calculate_direction(gaze_points[start_index], gaze_points[end_index])
            
            while end_index < n:
                current_velocity = calculate_velocity(gaze_points[end_index-1], gaze_points[end_index])
                current_direction = calculate_direction(gaze_points[end_index-1], gaze_points[end_index])
                direction_change = abs(current_direction - prev_direction)
                
                if current_velocity > velocity_threshold or direction_change > direction_threshold:
                    break
                
                if gaze_points[end_index][2] - gaze_points[start_index][2] >= time_window:
                    duration = gaze_points[end_index][2] - gaze_points[start_index][2]
                    smooth_pursuits.append((start_index, end_index, duration))
                    break
                
                prev_direction = current_direction
                end_index += 1
            
            start_index = end_index
        
        return smooth_pursuits
import numpy as np
from scipy import stats
import csv
from typing import Dict, List, Tuple
import logging

class adtech:
    def analyze_eye_tracking_data(results, aois, fps, fixation_threshold_sec=0.5, distance_threshold=50):
        """
        Analyze eye tracking data to calculate metrics for Areas of Interest (AOIs) and general viewing behavior.

        This function processes a series of eye gaze predictions and calculates various metrics
        for predefined Areas of Interest (AOIs) as well as general viewing metrics.

        Parameters:
        results (list of dict): A list of dictionaries, each containing 'pred_x' and 'pred_y' keys
                                representing the predicted x and y coordinates of the eye gaze.
        aois (dict): A dictionary where keys are AOI names and values are tuples representing
                    the bounding rectangle of each AOI in the format (x1, y1, x2, y2).
        fps (int): The frames per second of the recorded eye tracking data.
        fixation_threshold_sec (float): Minimum duration in seconds for a gaze point to be considered a fixation. Default is 0.5 seconds.
        distance_threshold (float): Maximum distance in pixels between consecutive gaze points to be considered part of the same fixation. Default is 50 pixels.

        Returns:
        tuple: A tuple containing two dictionaries:
            1. aoi_metrics: A dictionary with metrics for each AOI:
                - 'TFF' (Time to First Fixation): Time in seconds before the AOI was first looked at.
                - 'Fixation_Count': Number of fixations on the AOI.
                - 'Total_Fixation_Duration': Total time in seconds spent looking at the AOI.
                - 'Avg_Fixation_Duration': Average duration of fixations on the AOI in seconds.
                - 'Revisits': Number of times the gaze returned to the AOI after looking elsewhere.
            2. general_metrics: A dictionary with general viewing metrics:
                - 'Entry_Point': The coordinates (x, y) where the gaze first entered the stimulus.
                - 'Exit_Point': The coordinates (x, y) where the gaze last left the stimulus.

        Note:
        - This function assumes that the eye tracking data points are equally spaced in time.
        - The fixation detection uses a simple distance-based threshold method.
        """



        fixation_threshold_frames = int(fixation_threshold_sec * fps)

        def point_in_rect(x, y, rect):
            return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]

        aoi_metrics = {aoi_name: {
            'TFF': None,
            'Fixation_Duration': [],
            'Fixation_Count': 0,
            'Total_Fixation_Duration': 0,
            'Revisits': 0
        } for aoi_name in aois}

        general_metrics = {
            'Entry_Point': (results[0]['pred_x'], results[0]['pred_y']),
            'Exit_Point': (results[-1]['pred_x'], results[-1]['pred_y'])
        }

        current_fixation = None
        last_aoi = None

        for i, result in enumerate(results):
            x, y = result['pred_x'], result['pred_y']
            timestamp_sec = i / fps

            for aoi_name, aoi_rect in aois.items():
                if point_in_rect(x, y, aoi_rect):
                    if aoi_metrics[aoi_name]['TFF'] is None:
                        aoi_metrics[aoi_name]['TFF'] = timestamp_sec

                    if current_fixation is None:
                        current_fixation = (x, y, i)
                    elif i == len(results) - 1 or i - current_fixation[2] >= fixation_threshold_frames or \
                            (i < len(results) - 1 and
                            ((results[i + 1]['pred_x'] - x) ** 2 + (results[i + 1]['pred_y'] - y) ** 2) ** 0.5 > distance_threshold):
                        fixation_duration_sec = (i - current_fixation[2]) / fps
                        aoi_metrics[aoi_name]['Fixation_Duration'].append(fixation_duration_sec)
                        aoi_metrics[aoi_name]['Fixation_Count'] += 1
                        aoi_metrics[aoi_name]['Total_Fixation_Duration'] += fixation_duration_sec

                        if last_aoi != aoi_name:
                            aoi_metrics[aoi_name]['Revisits'] += 1

                        current_fixation = None

                    last_aoi = aoi_name
                    break
            else:
                current_fixation = None
                last_aoi = None

        for aoi_name in aoi_metrics:
            if aoi_metrics[aoi_name]['Fixation_Count'] > 0:
                aoi_metrics[aoi_name]['Avg_Fixation_Duration'] = (
                    aoi_metrics[aoi_name]['Total_Fixation_Duration'] / aoi_metrics[aoi_name]['Fixation_Count']
                )
            else:
                aoi_metrics[aoi_name]['Avg_Fixation_Duration'] = 0

            del aoi_metrics[aoi_name]['Fixation_Duration']

        return aoi_metrics, general_metrics

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.widgets import RectangleSelector, Button, TextBox
    from typing import Dict, Tuple
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    class DraggableRectangle:
        def __init__(self, rect, name_text):
            self.rect = rect
            self.name_text = name_text
            self.press = None
            self.connect()

        def connect(self):
            self.cidpress = self.rect.figure.canvas.mpl_connect('button_press_event', self.on_press)
            self.cidrelease = self.rect.figure.canvas.mpl_connect('button_release_event', self.on_release)
            self.cidmotion = self.rect.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

        def on_press(self, event):
            if event.inaxes != self.rect.axes:
                return
            contains, attrd = self.rect.contains(event)
            if not contains:
                return
            self.press = self.rect.xy, (event.xdata, event.ydata)

        def on_motion(self, event):
            if self.press is None:
                return
            if event.inaxes != self.rect.axes:
                return
            xy, (xpress, ypress) = self.press
            dx = event.xdata - xpress
            dy = event.ydata - ypress
            self.rect.set_xy(xy + (dx, dy))
            self.name_text.set_position((xy[0] + dx, xy[1] + dy))
            self.rect.figure.canvas.draw()

        def on_release(self, event):
            self.press = None
            self.rect.figure.canvas.draw()

    def define_aois(image_path: str) -> Dict[str, Tuple[float, float, float, float]]:
        """
        Provides an interactive interface for defining Areas of Interest (AOIs) on an image.

        This function opens a matplotlib window displaying the specified image and allows
        the user to create, select, rename, move, and delete AOIs using mouse interactions
        and GUI buttons.

        Args:
        image_path (str): Path to the image file on which AOIs will be defined.

        Returns:
        Dict[str, Tuple[float, float, float, float]]: A dictionary where keys are AOI names
        and values are tuples representing the bounding box of each AOI in the format
        (x1, y1, x2, y2), where (x1, y1) is the top-left corner and (x2, y2) is the
        bottom-right corner of the AOI.

        Functionality:
        - Create Mode: Left-click and drag to create a new AOI.
        - Select Mode: Click on an existing AOI to select it.
        - Rename: Type a new name in the text box and click 'Rename' to rename the selected AOI.
        - Delete: Click 'Delete' to remove the selected AOI.
        - Move: Click and drag an existing AOI to move it.
        - Mode Toggle: Use the 'Mode' button to switch between 'Create' and 'Select' modes.
        - Display AOIs: Press 'd' key to display current AOIs in the console.
        - Quit: Press 'q' key or click 'Close' button to finish and close the window.

        Note:
        - The function will return an empty dictionary if there's an error reading the image file.
        - AOIs are represented as rectangles on the image.
        - The function uses matplotlib for rendering and interaction.

        Raises:
        FileNotFoundError: If the specified image file is not found.
        Exception: For any other error occurring while reading the image file.
        """

        try:
            img = plt.imread(image_path)
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            return {}
        except Exception as e:
            logger.error(f"Error reading image file: {e}")
            return {}

        fig, ax = plt.subplots(figsize=(12, 10))  # Increased figure height
        plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin
        ax.imshow(img)

        aois = {}
        draggable_rects = {}
        selected_aoi = [None]
        mode = ['create']

        def print_instructions():
            print("\nInstructions:")
            print("- Use the 'Mode' button to switch between 'Create' and 'Select' modes")
            print("- In 'Create' mode, left-click and drag to create a new AOI")
            print("- In 'Select' mode, click on an AOI to select it")
            print("- Type a new name in the text box and click 'Rename' to rename the selected AOI")
            print("- Click 'Delete' to remove the selected AOI")
            print("- Click and drag an AOI to move it (works in both modes)")
            print("- Press 'd' to display current AOIs")
            print("- Press 'q' to finish and quit\n")

        def onselect(eclick, erelease):
            if mode[0] == 'create':
                x1, y1 = eclick.xdata, eclick.ydata
                x2, y2 = erelease.xdata, erelease.ydata
                temp_name = f"AOI_{len(aois) + 1}"
                aois[temp_name] = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                rect = patches.Rectangle((min(x1, x2), min(y1, y2)), abs(x2-x1), abs(y2-y1),
                                        fill=False, edgecolor='r')
                ax.add_patch(rect)
                text = ax.text(min(x1, x2), min(y1, y2), temp_name, color='r')
                draggable_rects[temp_name] = DraggableRectangle(rect, text)
                fig.canvas.draw()
                print(f"Created {temp_name}. Select it and use 'Rename' to change its name.")

        def onclick(event):
            if event.inaxes != ax or mode[0] != 'select':
                return
            for aoi_name, (x1, y1, x2, y2) in list(aois.items()):
                if x1 <= event.xdata <= x2 and y1 <= event.ydata <= y2:
                    selected_aoi[0] = aoi_name
                    print(f"Selected {aoi_name}")
                    rename_textbox.set_val(aoi_name)
                    break
            else:
                selected_aoi[0] = None
                rename_textbox.set_val('')

        def rename_aoi(event):
            if selected_aoi[0]:
                new_name = rename_textbox.text
                if new_name and new_name != selected_aoi[0]:
                    aois[new_name] = aois.pop(selected_aoi[0])
                    draggable_rects[new_name] = draggable_rects.pop(selected_aoi[0])
                    draggable_rects[new_name].name_text.set_text(new_name)
                    fig.canvas.draw()
                    print(f"Renamed {selected_aoi[0]} to {new_name}")
                    selected_aoi[0] = new_name

        def delete_aoi(event):
            if selected_aoi[0]:
                del aois[selected_aoi[0]]
                draggable_rects[selected_aoi[0]].rect.remove()
                draggable_rects[selected_aoi[0]].name_text.remove()
                del draggable_rects[selected_aoi[0]]
                fig.canvas.draw()
                print(f"Deleted {selected_aoi[0]}")
                selected_aoi[0] = None
                rename_textbox.set_val('')

        def toggle_mode(event):
            mode[0] = 'select' if mode[0] == 'create' else 'create'
            mode_button.label.set_text(f"Mode: {mode[0].capitalize()}")
            fig.canvas.draw()
            print(f"Switched to {mode[0]} mode")

        def onkey(event):
            if event.key == 'd':
                print("Current AOIs:")
                for name, coords in aois.items():
                    print(f"{name}: {coords}")
            elif event.key == 'q':
                plt.close(fig)

        print_instructions()
        rs = RectangleSelector(ax, onselect, useblit=True,
                            button=[1], minspanx=5, minspany=5,
                            spancoords='pixels', interactive=True)
        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', onkey)

        # Adjust button and textbox positions
        fig.subplots_adjust(bottom=0.2)  # Increase bottom margin for controls

        button_width = 0.12
        button_height = 0.05
        button_bottom = 0.05
        spacing = 0.02
        textbox_label_width = 0.08  # Width for the TextBox label

        ax_mode = plt.axes([0.02, button_bottom, button_width, button_height])
        ax_textbox = plt.axes([0.02 + button_width + spacing + textbox_label_width, button_bottom, 0.25, button_height])
        ax_rename = plt.axes(
            [0.02 + button_width + spacing + textbox_label_width + 0.25 + spacing, button_bottom, button_width,
            button_height])
        ax_delete = plt.axes(
            [0.02 + button_width + spacing + textbox_label_width + 0.25 + spacing + button_width + spacing, button_bottom,
            button_width, button_height])
        ax_close = plt.axes([
                                0.02 + button_width + spacing + textbox_label_width + 0.25 + spacing + button_width + spacing + button_width + spacing,
                                button_bottom, button_width, button_height])

        mode_button = Button(ax_mode, 'Mode: Create')
        rename_textbox = TextBox(ax_textbox, 'New Name: ')
        btn_rename = Button(ax_rename, 'Rename')
        btn_delete = Button(ax_delete, 'Delete')
        btn_close = Button(ax_close, 'Close')

        # Add close button functionality
        def close_figure(event):
            plt.close(fig)

        btn_close.on_clicked(close_figure)

        mode_button.on_clicked(toggle_mode)
        btn_rename.on_clicked(rename_aoi)
        btn_delete.on_clicked(delete_aoi)

        plt.show()

        # Update aois with final positions of draggable rectangles
        for name, drect in draggable_rects.items():
            x, y = drect.rect.get_xy()
            w, h = drect.rect.get_width(), drect.rect.get_height()
            aois[name] = (x, y, x + w, y + h)

        return aois


    def plot_gaze_path(results: List[Dict[str, float]], aois: Dict[str, Tuple[float, float, float, float]],
                    image_path: str):
        """
        Visualizes the gaze path over the advertisement image.

        This function creates a plot showing the path of the viewer's gaze overlaid on the original image,
        along with the defined Areas of Interest (AOIs).

        Args:
        results (List[Dict[str, float]]): A list of dictionaries, each containing 'pred_x' and 'pred_y' keys
                                        representing the predicted x and y coordinates of the eye gaze.
        aois (Dict[str, Tuple[float, float, float, float]]): A dictionary where keys are AOI names and values
                                                            are tuples representing the bounding box of each AOI
                                                            in the format (x1, y1, x2, y2).
        image_path (str): Path to the image file used as the background for the visualization.

        The function will:
        1. Load and display the background image.
        2. Plot the gaze path as a continuous line.
        3. Overlay scatter points representing individual gaze positions.
        4. Draw rectangles representing the AOIs.

        Note:
        - The gaze path is plotted in blue with low opacity for clarity.
        - The scatter points are colored according to their temporal order using a 'cool' colormap.
        - AOIs are drawn as red rectangles with their names labeled.

        Raises:
        FileNotFoundError: If the specified image file is not found.
        Exception: For any other error occurring while reading the image file.
        """

        try:
            img = plt.imread(image_path)
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            return
        except Exception as e:
            logger.error(f"Error reading image file: {e}")
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img)

        x = [r['pred_x'] for r in results]
        y = [r['pred_y'] for r in results]
        ax.plot(x, y, 'b-', linewidth=0.5, alpha=0.7)
        ax.scatter(x, y, c=range(len(x)), cmap='cool', s=10, zorder=2)

        for aoi_name, (x1, y1, x2, y2) in aois.items():
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='r')
            ax.add_patch(rect)
            ax.text(x1, y1, aoi_name, color='r')

        plt.title("Gaze Path Visualization")
        plt.show()


    def generate_heatmap(results: List[Dict[str, float]], image_path: str, bins: int = 50):
        """
        Creates a heatmap of gaze intensity overlaid on the advertisement image.

        This function generates a heatmap visualization of the gaze data, showing areas of high and low
        gaze concentration overlaid on the original image.

        Args:
        results (List[Dict[str, float]]): A list of dictionaries, each containing 'pred_x' and 'pred_y' keys
                                        representing the predicted x and y coordinates of the eye gaze.
        image_path (str): Path to the image file used as the background for the heatmap.
        bins (int): Number of bins to use for the 2D histogram. Default is 50.

        The function will:
        1. Load and display the background image.
        2. Create a 2D histogram of the gaze data.
        3. Overlay the heatmap on the image using a 'hot' colormap with partial transparency.
        4. Add a colorbar to show the intensity scale.

        Note:
        - The function includes error checking for empty results, negative coordinates, and coordinates
        outside the image dimensions.
        - The heatmap uses a 'hot' colormap where red indicates areas of high gaze concentration.

        Raises:
        FileNotFoundError: If the specified image file is not found.
        Exception: For any other error occurring while reading the image file or processing the data.
        """

        if not results:
            logger.error("No gaze data provided for heatmap generation.")
            return

        if any(r['pred_x'] < 0 or r['pred_y'] < 0 for r in results):
            logger.warning("Negative coordinates found in gaze data. This may cause issues with heatmap generation.")

        img = plt.imread(image_path)
        if any(r['pred_x'] > img.shape[1] or r['pred_y'] > img.shape[0] for r in results):
            logger.warning(
                "Gaze coordinates found outside image dimensions. This may cause issues with heatmap generation.")

        try:
            img = plt.imread(image_path)
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            return
        except Exception as e:
            logger.error(f"Error reading image file: {e}")
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img)

        x = [r['pred_x'] for r in results]
        y = [r['pred_y'] for r in results]

        heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, img.shape[1]], [0, img.shape[0]]])
        extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]

        # Create a heatmap overlay
        heatmap_overlay = ax.imshow(heatmap.T, extent=extent, origin='upper', cmap='hot', alpha=0.5)

        plt.title("Gaze Intensity Heatmap")
        plt.colorbar(heatmap_overlay, label='Gaze Intensity')
        plt.show()


    def aoi_significance_test(group1_results: List[Dict[str, float]], group2_results: List[Dict[str, float]],
                            aois: Dict[str, Tuple[float, float, float, float]], test: str = 't-test'):
        """
        Performs statistical tests to compare AOI metrics between two groups.

        This function calculates and compares metrics for each Area of Interest (AOI) between two groups
        of gaze data, using either a t-test or Mann-Whitney U test.

        Args:
        group1_results (List[Dict[str, float]]): Gaze data for the first group. Each dict should contain
                                                'pred_x' and 'pred_y' keys for gaze coordinates.
        group2_results (List[Dict[str, float]]): Gaze data for the second group. Same format as group1_results.
        aois (Dict[str, Tuple[float, float, float, float]]): A dictionary where keys are AOI names and values
                                                            are tuples representing the bounding box of each AOI
                                                            in the format (x1, y1, x2, y2).
        test (str): Statistical test to use. Either 't-test' or 'mann-whitney'. Default is 't-test'.

        Returns:
        Dict: A dictionary containing the results of the statistical tests for each AOI. Each AOI entry includes:
            - 'group1_mean': Mean value for group 1
            - 'group2_mean': Mean value for group 2
            - 'statistic': The test statistic
            - 'p_value': The p-value of the test

        The function will:
        1. Calculate the proportion of gaze points within each AOI for both groups.
        2. Perform the specified statistical test to compare these proportions between the groups.
        3. Return the results including means, test statistic, and p-value for each AOI.

        Note:
        - The function assumes that the AOIs and gaze coordinates use the same coordinate system.
        - The choice of test should be based on the nature of your data and experimental design.

        Raises:
        ValueError: If an invalid test type is specified.
        """

        def get_aoi_metrics(results, aois):
            metrics = {aoi: [] for aoi in aois}
            for r in results:
                for aoi, (x1, y1, x2, y2) in aois.items():
                    if x1 <= r['pred_x'] <= x2 and y1 <= r['pred_y'] <= y2:
                        metrics[aoi].append(1)
                    else:
                        metrics[aoi].append(0)
            return metrics

        group1_metrics = get_aoi_metrics(group1_results, aois)
        group2_metrics = get_aoi_metrics(group2_results, aois)

        results = {}
        for aoi in aois:
            if test == 't-test':
                statistic, p_value = stats.ttest_ind(group1_metrics[aoi], group2_metrics[aoi])
            elif test == 'mann-whitney':
                statistic, p_value = stats.mannwhitneyu(group1_metrics[aoi], group2_metrics[aoi])
            else:
                raise ValueError("Invalid test type. Use 't-test' or 'mann-whitney'.")

            results[aoi] = {
                'group1_mean': np.mean(group1_metrics[aoi]),
                'group2_mean': np.mean(group2_metrics[aoi]),
                'statistic': statistic,
                'p_value': p_value
            }

        return results


    def export_metrics_to_csv(aoi_metrics: Dict[str, Dict[str, float]], general_metrics: Dict[str, float], filename: str):
        """
        Exports calculated metrics to a CSV file for further analysis in other software.

        This function takes the metrics calculated for Areas of Interest (AOIs) and general viewing behavior
        and writes them to a CSV file in a structured format.

        Args:
        aoi_metrics (Dict[str, Dict[str, float]]): A nested dictionary where the outer key is the AOI name,
                                                and the inner dictionary contains various metrics as key-value pairs.
        general_metrics (Dict[str, float]): A dictionary of general metrics that apply to the entire viewing session.
        filename (str): The name of the output CSV file, including path if necessary.

        The function will:
        1. Create a new CSV file with the specified filename.
        2. Write AOI metrics, with each row containing the AOI name, metric name, and value.
        3. Write general metrics, with each row containing the metric name and value.

        The CSV structure will be:
        AOI Metrics
        AOI, Metric, Value
        [AOI metrics data]

        General Metrics
        Metric, Value
        [General metrics data]

        Note:
        - If the file already exists, it will be overwritten.
        - The function uses the csv module to ensure proper CSV formatting.

        Raises:
        IOError: If there's an error writing to the file (e.g., permission denied, disk full).
        """

        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                writer.writerow(['AOI Metrics'])
                writer.writerow(['AOI', 'Metric', 'Value'])
                for aoi, metrics in aoi_metrics.items():
                    for metric, value in metrics.items():
                        writer.writerow([aoi, metric, value])

                writer.writerow([])
                writer.writerow(['General Metrics'])
                writer.writerow(['Metric', 'Value'])
                for metric, value in general_metrics.items():
                    writer.writerow([metric, value])

            logger.info(f"Metrics exported to {filename}")
        except IOError as e:
            logger.error(f"Error writing to CSV file: {e}")


    def main():
        # Example usage of the functions
        image_path = "demo_ad.jpg"

        # Step 1: Define AOIs dynamically on the picture
        print("Please define Areas of Interest (AOIs) on the advertisement image.")
        aois = define_aois(image_path)
        print("Defined AOIs:", aois)

        # Step 2: Record a short video for eye tracking (simulated here)
        print("\nSimulating video recording for eye tracking...")
        # In a real scenario, you would record actual video here
        video_duration = 10  # seconds
        fps = 30
        frame_count = video_duration * fps

        # Simulate eye-tracking data
        np.random.seed(42)  # for reproducibility
        simulated_results = [
            {'pred_x': np.random.uniform(0, 1920), 'pred_y': np.random.uniform(0, 1080)}
            for _ in range(frame_count)
        ]

        # Step 3: Analyze eye-tracking data
        print("\nAnalyzing eye-tracking data...")
        aoi_metrics, general_metrics = analyze_eye_tracking_data(simulated_results, aois, fps)

        print("\nAOI Metrics:")
        for aoi, metrics in aoi_metrics.items():
            print(f"{aoi}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")

        print("\nGeneral Metrics:")
        for metric, value in general_metrics.items():
            print(f"{metric}: {value}")

        # Step 4: Visualize gaze path
        print("\nGenerating gaze path visualization...")
        plot_gaze_path(simulated_results, aois, image_path)

        # Step 5: Generate heatmap
        print("\nGenerating gaze intensity heatmap...")
        generate_heatmap(simulated_results, image_path)

        # Step 6: Perform significance test (simulating two groups)
        print("\nPerforming significance test between two simulated groups...")
        group1_results = simulated_results[:len(simulated_results) // 2]
        group2_results = simulated_results[len(simulated_results) // 2:]

        significance_results = aoi_significance_test(group1_results, group2_results, aois)

        print("\nSignificance Test Results:")
        for aoi, results in significance_results.items():
            print(f"{aoi}:")
            for metric, value in results.items():
                print(f"  {metric}: {value}")

        # Step 7: Export metrics to CSV
        print("\nExporting metrics to CSV...")
        export_metrics_to_csv(aoi_metrics, general_metrics, "eye_tracking_metrics.csv")
        print("Metrics exported to eye_tracking_metrics.csv")

        print("\nEye-tracking analysis pipeline completed!")

        # # Step 2: Display the ad and record a 10-second video
        # print("\nPreparing to record a 10-second video. Please look at the displayed advertisement.")
        # screen = cv2.imread(image_path)
        # screen_height, screen_width = screen.shape[:2]
        #
        # # Prepare video recording
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter('gaze_recording.mp4', fourcc, 30.0, (screen_width, screen_height))
        #
        # cv2.namedWindow("Advertisement", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("Advertisement", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        #
        # # Record video
        # start_time = time.time()
        # while time.time() - start_time < 10:  # Record for 10 seconds
        #     out.write(screen)
        #     cv2.imshow("Advertisement", screen)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        #
        # out.release()
        # cv2.destroyAllWindows()
        #
        # print("Video recording completed.")
        #
        # # Step 3: Use Vytal API to get gaze predictions
        # print("\nProcessing the video to obtain gaze predictions...")
        # predictor = vytal.Client()
        # results = predictor.predict_from_video('gaze_recording.mp4')
        #
        # # Convert Vytal results to our format
        # processed_results = []
        # for result in results:
        #     gaze_x = result.x * screen_width
        #     gaze_y = result.y * screen_height
        #     processed_results.append({
        #         'pred_x': gaze_x,
        #         'pred_y': gaze_y
        #     })
        #
        # # Step 4: Analyze eye-tracking data
        # print("\nAnalyzing eye-tracking data...")
        # fps = 30  # Assuming 30 fps video
        # aoi_metrics, general_metrics = analyze_eye_tracking_data(processed_results, aois, fps)
        #
        # print("\nAOI Metrics:")
        # for aoi, metrics in aoi_metrics.items():
        #     print(f"{aoi}:")
        #     for metric, value in metrics.items():
        #         print(f"  {metric}: {value}")
        #
        # print("\nGeneral Metrics:")
        # for metric, value in general_metrics.items():
        #     print(f"{metric}: {value}")
        #
        # # Step 5: Visualize gaze path
        # print("\nGenerating gaze path visualization...")
        # plot_gaze_path(processed_results, aois, image_path)
        #
        # # Step 6: Generate heatmap
        # print("\nGenerating gaze intensity heatmap...")
        # generate_heatmap(processed_results, image_path)
        #
        # # Step 7: Export metrics to CSV
        # print("\nExporting metrics to CSV...")
        # export_metrics_to_csv(aoi_metrics, general_metrics, "eye_tracking_metrics.csv")
        # print("Metrics exported to eye_tracking_metrics.csv")
        #
        # print("\nEye-tracking analysis pipeline completed!")


    # if __name__ == "__main__":
    #     main()