from math import sqrt
from statistics import mean
import numpy as np

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