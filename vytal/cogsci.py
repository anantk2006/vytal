import torch
from typing import List, Dict, Tuple
import numpy as np
import scipy.signal

class EyeTrackingAnalyzer:
    def __init__(self, data: List[Dict], sampling_rate: float):
        self.data = self.preprocess_data(data)
        self.NEW_FPS = sampling_rate

    def preprocess_data(self, data: List[Dict]) -> Dict[str, torch.Tensor]:
        if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
            raise TypeError("Data must be a list of dictionaries.")

        if len(data) == 0:
            raise ValueError("Data list cannot be empty.")

        required_keys = ['time', 'left', 'right', 'face', 'left_PoG', 'right_PoG', 'blink']
        for entry in data:
            for key in required_keys:
                if key not in entry:
                    raise KeyError(f"Missing key '{key}' in one or more dictionaries in the data list.")

        # Convert the data to tensors
        processed_data = {
            'time': torch.tensor([d['time'] for d in data], dtype=torch.float32),
            'left': torch.tensor([d['left'] for d in data], dtype=torch.float32),
            'right': torch.tensor([d['right'] for d in data], dtype=torch.float32),
            'face': torch.tensor([d['face'] for d in data], dtype=torch.float32),
            'left_PoG': torch.tensor([d['left_PoG'] for d in data], dtype=torch.float32),
            'right_PoG': torch.tensor([d['right_PoG'] for d in data], dtype=torch.float32),
            'blink': torch.tensor([d['blink'] for d in data], dtype=torch.float32)
        }

        # Prepare the data in the required format for calc_velocities
        calc_velocities_input = [
            {key: d[key] for key in required_keys}
            for d in data
        ]

        # Calculate velocities
        velocities = self.calc_velocities(calc_velocities_input)

        # Add velocities to processed_data
        for key, value in velocities.items():
            processed_data[key + '_velocity'] = value

        return processed_data

    def calc_velocities(self, data: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Calculates the derivative of the given eye tracking data with respect to time.

        Args:
            data (List[Dict]): List of dictionaries containing eye tracking data and time.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the derivatives for each key in the original data.
        """
        if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
            raise TypeError("Data must be a list of dictionaries.")

        if len(data) == 0:
            raise ValueError("Data list cannot be empty.")

        required_keys = ['time', 'left', 'right', 'face', 'left_PoG', 'right_PoG', 'blink']
        for entry in data:
            for key in required_keys:
                if key not in entry:
                    raise KeyError(f"Missing key '{key}' in one or more dictionaries in the data list.")

        # Extract time and convert to tensor
        time = torch.tensor([entry['time'] for entry in data], dtype=torch.float32)

        # Convert time from milliseconds to seconds
        time = time / 1000.0

        # Initialize a dictionary to store the tensors for each key
        keys_to_process = ['left', 'right', 'face', 'left_PoG', 'right_PoG', 'blink']
        tensors = {key: [] for key in keys_to_process}

        # Convert the list of dicts to a dict of lists
        for entry in data:
            for key in keys_to_process:
                tensors[key].append(entry[key])

        # Ensure all lists have the same length
        lengths = [len(tensors[key]) for key in keys_to_process]
        if len(set(lengths)) != 1:
            raise ValueError("Inconsistent lengths found among keys in the data list.")

        # Convert lists to tensors
        for key in tensors:
            try:
                if isinstance(tensors[key][0], list):
                    tensors[key] = torch.tensor(tensors[key], dtype=torch.float32)
                else:
                    tensors[key] = torch.tensor(tensors[key], dtype=torch.float32).unsqueeze(-1)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Non-numeric value encountered in key '{key}': {e}")

        # Calculate derivatives
        derivatives = {}
        for key, tensor in tensors.items():
            delta_time = torch.diff(time)
            delta_data = torch.diff(tensor, dim=0)
            derivative = delta_data / delta_time.unsqueeze(-1)

            # Append torch.nan to the end of the derivative tensor to match the original length
            nan_tensor = torch.full((1, *derivative.shape[1:]), torch.nan, dtype=derivative.dtype, device=derivative.device)
            derivatives[key] = torch.cat((derivative, nan_tensor), dim=0)

        return derivatives

    def find_saccade_locs(self, velocity_threshold: float = 30, min_duration: float = 30) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
        """
        Detects saccades in velocity data for both left and right eyes.

        Args:
            velocity_threshold (float): The minimum peak height in deg/sec to consider a saccade.
            min_duration (float): The minimum duration of a saccade in ms.

        Returns:
            Tuple: A tuple containing two lists of tuples. Each list corresponds to left and right eyes respectively.
            Each tuple represents the start, end, and peak indices of a saccade.
        """
        width = int(self.NEW_FPS * min_duration / 1000)

        def find_saccades(velocities):
            abs_velocities = np.abs((180 / np.pi) * velocities.squeeze().numpy().flatten())
            peaks, _ = scipy.signal.find_peaks(abs_velocities, height=velocity_threshold, width=width)
            for rel_height in np.arange(0.75, 0.65, -0.01):
                _, _, left_ips, right_ips = scipy.signal.peak_widths(abs_velocities, peaks, rel_height=rel_height)
                if np.all(right_ips[:-1] < left_ips[1:]):
                    break

            if len(left_ips) == len(right_ips) == len(peaks):
                return list(zip(left_ips.astype(int) + 1, right_ips.astype(int) + 1, peaks.astype(int) + 1))
            else:
                raise ValueError("The iterables do not have the same length.")

        left_velocities = self.data['left_velocity']
        right_velocities = self.data['right_velocity']

        left_saccade_locs = find_saccades(left_velocities)
        right_saccade_locs = find_saccades(right_velocities)

        return left_saccade_locs, right_saccade_locs

    @staticmethod
    def remove_overlapping_saccades(saccades: List[Dict]) -> List[Dict]:
        """
        Removes overlapping saccades from the given list.

        Args:
            saccades (List[Dict]): List of saccades, each represented as a dictionary.

        Returns:
            List[Dict]: List of non-overlapping saccades.
        """
        saccades.sort(key=lambda x: x['start'])
        non_overlapping_saccades = []

        for saccade in saccades:
            if not non_overlapping_saccades or saccade['start'] > non_overlapping_saccades[-1]['end']:
                non_overlapping_saccades.append(saccade)
            else:
                if abs(saccade['peak_velocity']) > abs(non_overlapping_saccades[-1]['peak_velocity']):
                    non_overlapping_saccades[-1] = saccade
        return non_overlapping_saccades

    def get_saccade_data(self, angle_type: str = 'face', accel_threshold: float = 0) -> Tuple[List[Dict], List[Dict]]:
        """
        Extracts saccade data from the given data for both left and right eyes.

        Args:
            angle_type (str): The type of angle to use for saccade data. Can be 'face', 'left', or 'right'.
            accel_threshold (float): The minimum acceleration to consider for a saccade.

        Returns:
            Tuple: Each list contains dictionaries for left and right eye saccades respectively.
            Each dictionary contains information about a saccade, including its start and end indices,
            duration, peak index, peak velocity, and amplitude.
        """
        left_saccade_locs, right_saccade_locs = self.find_saccade_locs()
        assert angle_type in ['face', 'left', 'right']

        def extract_saccade_data(saccade_locs, velocity, angles):
            saccade_data = []
            acceleration = np.abs((180 / np.pi) * self.calc_velocities([{'time': self.data['time'].numpy().tolist(), 'data': velocity.numpy().tolist()}])['data_velocity'].squeeze().numpy().flatten())
            for start, stop, peak in saccade_locs:
                if acceleration[peak] > accel_threshold:
                    duration = self.data['time'][stop].item() - self.data['time'][start].item()
                    start_angle = 180 / np.pi * angles[start].item()
                    end_angle = 180 / np.pi * angles[stop].item()
                    amplitude = end_angle - start_angle
                    peak_velocity = 180 / np.pi * velocity[peak].item()

                    saccade_info = {
                        'start': start,
                        'end': stop,
                        'duration': duration,  # in ms
                        'peak': peak,
                        'peak_velocity': peak_velocity,
                        'amplitude': amplitude,
                    }

                    saccade_data.append(saccade_info)

            return self.remove_overlapping_saccades(saccade_data)

        # Left eye saccade data
        left_velocity = self.data['left_velocity']
        left_angles = self.data[angle_type] if angle_type == 'face' else self.data['left']
        left_saccade_data = extract_saccade_data(left_saccade_locs, left_velocity, left_angles)

        # Right eye saccade data
        right_velocity = self.data['right_velocity']
        right_angles = self.data[angle_type] if angle_type == 'face' else self.data['right']
        right_saccade_data = extract_saccade_data(right_saccade_locs, right_velocity, right_angles)

        return left_saccade_data, right_saccade_data

def detect_saccades(data: List[Dict], sampling_rate: float, velocity_threshold: float = 30, min_duration: float = 30,
                    accel_threshold: float = 0, angle_type: str = 'face') -> Dict[str, List[Dict]]:
    """
    Detects saccades in eye tracking data for both left and right eyes.

    This function processes eye tracking data to identify saccades based on velocity and acceleration thresholds.
    It calculates velocities and accelerations from the eye angle data, detects potential saccades,
    and then filters and refines these detections to produce a final list of saccades for each eye.

    Args:
        data (List[Dict]): A list of dictionaries containing eye tracking data.
            Each dictionary should have keys 'time', 'left', 'right', 'face' (angles in radians).
        sampling_rate (float): The sampling rate of the eye tracking data in Hz.
        velocity_threshold (float, optional): The minimum peak velocity in deg/sec to consider a saccade.
            Defaults to 30 deg/sec.
        min_duration (float, optional): The minimum duration of a saccade in milliseconds.
            Defaults to 30 ms.
        accel_threshold (float, optional): The minimum peak acceleration in deg/sec^2 to consider a saccade.
            Defaults to 0 deg/sec^2 (no acceleration filtering).
        angle_type (str, optional): The type of angle to use for saccade detection.
            Can be 'face', 'left', or 'right'. Defaults to 'face'.

    Returns:
        Dict[str, List[Dict]]: A dictionary with keys 'left' and 'right', each containing a list of
        dictionaries. Each dictionary represents a detected saccade with the following keys:
            - 'start': Index of saccade start in the original data list
            - 'end': Index of saccade end
            - 'duration': Duration of the saccade in milliseconds
            - 'peak': Index of peak velocity
            - 'peak_velocity': Maximum velocity reached during the saccade (deg/sec)
            - 'amplitude': Change in eye angle during the saccade (degrees)

    Raises:
        ValueError: If the input data is empty or doesn't contain the required keys.

    Note:
        This function uses a sophisticated algorithm to detect saccades, including peak detection,
        acceleration thresholding, and removal of overlapping saccades.
    """
    analyzer = EyeTrackingAnalyzer(data, sampling_rate)
    left_saccades, right_saccades = analyzer.get_saccade_data(angle_type, accel_threshold)

    return {
        'left': left_saccades,
        'right': right_saccades
    }



def detect_fixations(data, dispersion_threshold=1.0, duration_threshold=100, angle_type='face'):
    """
    Detects fixations in eye tracking data using a dispersion-based algorithm.
    Fixations are identified as periods where the gaze remains within a defined spatial threshold for a minimum time.

    Args:
        data (List[Dict]): Eye tracking data with each entry containing:
            'time' (float): Timestamp in milliseconds.
            'POG_x' (float): Gaze position X-coordinate.
            'POG_y' (float): Gaze position Y-coordinate.
            Additionally requires 'left', 'right', 'face' if 'angle_type' is specified.
        dispersion_threshold (float): Maximum allowed dispersion in gaze position units to qualify as a fixation.
        duration_threshold (float): Minimum duration in milliseconds for a valid fixation.
        angle_type (str): Specifies which angle data to use for additional fixation info ('face', 'left', 'right').

    Returns:
        List[Dict]: Each dictionary in the list represents a detected fixation, containing:
            'start_index' (int): Start index of fixation in data.
            'end_index' (int): End index of fixation.
            'duration' (float): Duration of fixation in milliseconds.
            'centroid_x' (float): Average X-coordinate of fixation.
            'centroid_y' (float): Average Y-coordinate of fixation.
            'dispersion' (float): Calculated dispersion of fixation.
            'mean_angle' (float): Mean angle during the fixation according to 'angle_type'.

    Raises:
        ValueError: If data is empty or missing required keys.
    """
    required_keys = {'time', 'POG_x', 'POG_y', angle_type}
    if not data:
        raise ValueError("Input data must be a non-empty list.")
    if not all(key in data[0] for key in required_keys):
        raise ValueError("Data items must contain all required keys: 'time', 'POG_x', 'POG_y', and the specified 'angle_type'.")

    fixations = []
    window_start = 0

    while window_start < len(data):
        window_end = window_start
        window_x, window_y = [], []

        while window_end < len(data):
            window_x.append(data[window_end]['POG_x'])
            window_y.append(data[window_end]['POG_y'])

            dispersion_x = max(window_x) - min(window_x)
            dispersion_y = max(window_y) - min(window_y)
            dispersion = np.sqrt(dispersion_x**2 + dispersion_y**2)

            if dispersion > dispersion_threshold and window_end > window_start:
                duration = data[window_end]['time'] - data[window_start]['time']
                if duration >= duration_threshold:
                    # Collect angles with safety checks
                    angles = [d.get(angle_type, 0) for d in data[window_start:window_end]]
                    mean_angle = np.mean(angles) if angles else 0  # Calculate mean or default to 0

                    fixation = {
                        'start_index': window_start,
                        'end_index': window_end - 1,
                        'duration': duration,
                        'centroid_x': np.mean(window_x[:-1]),
                        'centroid_y': np.mean(window_y[:-1]),
                        'dispersion': dispersion,
                        'mean_angle': mean_angle
                    }
                    fixations.append(fixation)
                break
            window_end += 1

        if window_end == len(data) and dispersion <= dispersion_threshold:
            duration = data[window_end - 1]['time'] - data[window_start]['time']
            if duration >= duration_threshold:
                angles = [d.get(angle_type, 0) for d in data[window_start:window_end]]
                mean_angle = np.mean(angles) if angles else 0  # Calculate mean or default to 0

                fixation = {
                    'start_index': window_start,
                    'end_index': window_end - 1,
                    'duration': duration,
                    'centroid_x': np.mean(window_x),
                    'centroid_y': np.mean(window_y),
                    'dispersion': dispersion,
                    'mean_angle': mean_angle
                }
                fixations.append(fixation)

        window_start = window_end

    return fixations


