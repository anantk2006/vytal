from math import sqrt
from statistics import mean
import numpy as np

def fixation_detection(gaze_points, distance_threshold=30, time_threshold_ms=1500):
    """
    Detects fixations from a list of gaze points.

    A fixation is defined as a period where the gaze points are within a certain distance
    threshold from a central point (centroid) for a minimum duration of time.

    Parameters:
    gaze_points (list of tuples): A list of tuples where each tuple contains (x, y, timestamp).
                                  - x (float): The x-coordinate of the gaze point.
                                  - y (float): The y-coordinate of the gaze point.
                                  - timestamp (float): The timestamp of the gaze point.
    distance_threshold (float): The maximum allowable distance from the centroid for points
                                to be considered part of the same fixation. Must be a positive number.
    time_threshold_ms (float): The minimum duration (in milliseconds) that points must be within
                               the distance threshold to be considered a fixation. Must be a positive number.

    Returns:
    list of tuples: A list of detected fixations, where each fixation is represented as a tuple:
                    - (centroid_x, centroid_y): The centroid coordinates of the fixation.
                    - duration (float): The duration of the fixation in seconds.

    Raises:
    ValueError: If the gaze points list is empty.
    ValueError: If any gaze point is not a valid (x, y, timestamp) tuple with numeric values.
    ValueError: If distance_threshold or time_threshold_ms is not a positive number.

    Example:
    >>> gaze_points = [(100, 100, 1000), (102, 98, 1500), (101, 99, 2000), (150, 150, 3000)]
    >>> fixations = fixation_detection(gaze_points, 30, 1500)
    >>> print(fixations)
    [((100.33333333333333, 99.0), 1.0), ((150.0, 150.0), 1.0)]
    """
    if not gaze_points:
        raise ValueError("Gaze points list is empty.")
    
    if not isinstance(distance_threshold, (int, float)) or distance_threshold <= 0:
        raise ValueError("Distance threshold must be a positive number.")
    
    if not isinstance(time_threshold_ms, (int, float)) or time_threshold_ms <= 0:
        raise ValueError("Time threshold must be a positive number.")
    
    # Convert time threshold from milliseconds to seconds
    time_threshold = time_threshold_ms / 1000
    
    for i, point in enumerate(gaze_points):
        if not isinstance(point, (tuple, list)) or len(point) != 3:
            raise ValueError(f"Gaze point {i} is not a valid (x, y, timestamp) tuple.")
        
        x, y, timestamp = point
        
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float)) and isinstance(timestamp, (int, float))):
            raise ValueError(f"Gaze point {i} contains non-numeric values.")
    
    # Sort gaze points by timestamp
    gaze_points.sort(key=lambda p: p[2])

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
            fixation_duration_ms = current_fixation[-1][2] - current_fixation[0][2]
            fixation_duration = fixation_duration_ms / 1000  # Convert milliseconds to seconds
            if fixation_duration >= time_threshold:
                fixation_centroid = (centroid_x, centroid_y)
                fixations.append((fixation_centroid, fixation_duration))
            
            # Start a new fixation with the current point
            current_fixation = [point]
    
    # Check if the last fixation meets the time threshold
    if current_fixation:
        fixation_duration_ms = current_fixation[-1][2] - current_fixation[0][2]
        fixation_duration = fixation_duration_ms / 1000  # Convert milliseconds to seconds
        if fixation_duration >= time_threshold:
            centroid_x = mean(p[0] for p in current_fixation)
            centroid_y = mean(p[1] for p in current_fixation)
            fixation_centroid = (centroid_x, centroid_y)
            fixations.append((fixation_centroid, fixation_duration))
    
    return fixations



def saccade_detection(gaze_points, velocity_threshold=1000):
    """
    Detects saccades from a list of gaze points.

    A saccade is defined as a rapid movement of the eye between two points that exceeds a certain velocity threshold.

    Parameters:
    gaze_points (list of tuples): A list of tuples where each tuple contains (x, y, timestamp).
                                  - x (float): The x-coordinate of the gaze point.
                                  - y (float): The y-coordinate of the gaze point.
                                  - timestamp (float): The timestamp of the gaze point in milliseconds.
    velocity_threshold (float): The minimum velocity (in pixels per second) required to consider a movement a saccade. Must be a positive number.

    Returns:
    list of dictionaries: A list of detected saccades, where each saccade is represented as a dictionary:
                          - start_point (tuple): The starting gaze point of the saccade.
                          - end_point (tuple): The ending gaze point of the saccade.
                          - duration (float): The duration of the saccade in milliseconds.
                          - amplitude (float): The total distance traveled during the saccade.
                          - velocities (list): A list of velocities calculated for each segment of the saccade.
                          - peak_velocity (float): The highest velocity recorded during the saccade.
                          - average_velocity (float): The average velocity recorded during the saccade.

    Raises:
    ValueError: If the gaze points list is empty.
    ValueError: If any gaze point is not a valid (x, y, timestamp) tuple with numeric values.
    ValueError: If velocity_threshold is not a positive number.
    """
    if not gaze_points:
        raise ValueError("Gaze points list is empty.")
    
    if not isinstance(velocity_threshold, (int, float)) or velocity_threshold <= 0:
        raise ValueError("Velocity threshold must be a positive number.")
    
    for i, point in enumerate(gaze_points):
        if not isinstance(point, (tuple, list)) or len(point) != 3:
            raise ValueError(f"Gaze point {i} is not a valid (x, y, timestamp) tuple.")
        
        x, y, timestamp = point
        
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float)) and isinstance(timestamp, (int, float))):
            raise ValueError(f"Gaze point {i} contains non-numeric values.")
    
    # Sort gaze points by timestamp
    gaze_points.sort(key=lambda p: p[2])
    
    for i in range(1, len(gaze_points)):
        if gaze_points[i][2] < gaze_points[i-1][2]:
            raise ValueError(f"Timestamp for gaze point {i} is earlier than the previous point.")

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
            velocity = float('inf')  # Handle zero time difference to avoid division by zero
        
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
    
    Parameters:
    gaze_points (list of tuples): A list of tuples where each tuple contains (x, y, timestamp).
                                  - x (float): The x-coordinate of the gaze point.
                                  - y (float): The y-coordinate of the gaze point.
                                  - timestamp (float): The timestamp of the gaze point in milliseconds.
    time_window (float): The time window in milliseconds to consider for smooth pursuit.
    velocity_threshold (float): The maximum velocity (pixels per second) to be considered smooth pursuit.
    direction_threshold (float): The maximum direction change (degrees) to be considered smooth pursuit.
    
    Returns:
    list of tuples: A list of smooth pursuit segments where each segment is represented as a tuple:
                    - start_index (int): The starting index of the smooth pursuit segment.
                    - end_index (int): The ending index of the smooth pursuit segment.
                    - duration (float): The duration of the smooth pursuit segment in milliseconds.

    Raises:
    ValueError: If the gaze points list is empty.
    ValueError: If any gaze point is not a valid (x, y, timestamp) tuple with numeric values.
    ValueError: If time_window, velocity_threshold, or direction_threshold is not a positive number.
    """
    if not gaze_points:
        raise ValueError("Gaze points list is empty.")
    
    if not isinstance(time_window, (int, float)) or time_window <= 0:
        raise ValueError("Time window must be a positive number.")
    
    if not isinstance(velocity_threshold, (int, float)) or velocity_threshold <= 0:
        raise ValueError("Velocity threshold must be a positive number.")
    
    if not isinstance(direction_threshold, (int, float)) or direction_threshold <= 0:
        raise ValueError("Direction threshold must be a positive number.")
    
    for i, point in enumerate(gaze_points):
        if not isinstance(point, (tuple, list)) or len(point) != 3:
            raise ValueError(f"Gaze point {i} is not a valid (x, y, timestamp) tuple.")
        
        x, y, timestamp = point
        
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float)) and isinstance(timestamp, (int, float))):
            raise ValueError(f"Gaze point {i} contains non-numeric values.")
    
    # Sort gaze points by timestamp
    gaze_points.sort(key=lambda p: p[2])
    
    smooth_pursuits = []
    n = len(gaze_points)
    
    def calculate_velocity(p1, p2):
        x1, y1, t1 = p1
        x2, y2, t2 = p2
        distance = sqrt((x2 - x1)**2 + (y2 - y1)**2)
        time_diff = (t2 - t1) / 1000  # Convert to seconds
        return distance / time_diff if time_diff > 0 else float('inf')  # Handle zero time difference
    
    def calculate_direction(p1, p2):
        x1, y1, _ = p1
        x2, y2, _ = p2
        direction = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if direction < 0:
            direction += 360
        return direction
    
    start_index = 0
    while start_index < n - 1:
        end_index = start_index + 1
        prev_direction = calculate_direction(gaze_points[start_index], gaze_points[end_index])
        
        while end_index < n:
            current_velocity = calculate_velocity(gaze_points[end_index-1], gaze_points[end_index])
            current_direction = calculate_direction(gaze_points[end_index-1], gaze_points[end_index])
            direction_change = abs(current_direction - prev_direction)
            
            if direction_change > 180:
                direction_change = 360 - direction_change  # Handle wrap-around at 360 degrees
            
            if current_velocity > velocity_threshold or direction_change > direction_threshold:
                break
            
            prev_direction = current_direction
            end_index += 1

        if end_index > start_index + 1 and gaze_points[end_index-1][2] - gaze_points[start_index][2] >= time_window:
            duration = gaze_points[end_index-1][2] - gaze_points[start_index][2]
            smooth_pursuits.append((start_index, end_index-1, duration))
        
        start_index = end_index
    
    return smooth_pursuits
