API
===

.. autosummary::
   :toctree: generated

   client
   adtech
   hci
   cogsci

Client
------------
Predicts gaze from webcam data.

.. py:module:: vytal.client
    
.. py:function:: Client(api_key: str, ipd: float = None)
   :noindex:

   :param api_key: (str) The API key for the Vytal API.
   :param ipd: (float) The inter-pupillary distance of the person in the video. Defaults to None.

.. py:function:: calibrate(save_directory: str = None) -> Union[scipy.interpolate._rbfinterp.RBFInterpolator, str, bytes]
   :module: vytal.client
   :noindex:

   Calibrates your personal gaze tracker with eye-tracking process.

   :param save_directory: (str) The directory to save calibration data to. Calibration data is formatted as '.pkl' file.
   :return: scipy.interpolate._rbfinterp.RBFInterpolator object, a string, or bytes containing calibration data.
   The user will be presented with a calibration task consisting of focusing on spinning triangles at different locations on the screen. For effective calibration, instructions should be followed closely.

.. py:function:: predict_from_video(video_path: str, calib: Union[scipy.interpolate._rbfinterp.RBFInterpolator, str, bytes] = None, eye_frames: bool = False) -> List[Dict[str, torch.Tensor]]
   :module: vytal.client
   :noindex:

   Predicts the gaze of a person in a video file.

   :param video_path: (str) The path to the video file.
   :param calib: (scipy.interpolate._rbfinterp.RBFInterpolator, str, bytes) A calibration object or data used for prediction.
   :param eye_frames: (bool) Whether to return the eye frames (128x128 images used for prediction)
   :return: A list of dictionaries containing the gaze predictions. Each dictionary represents a frame.
   The keys are 'left', 'right', 'le_3d', 're_3d', 'hr', 'ht', 'blinked', 
   and optionally "right_eye_frame" and "left_eye_frame" if eye_frames is True.
   Each key maps to a tensor containing the predictions.

.. py:function:: start_thread(cam_id: int = 0, calib: Union[scipy.interpolate._rbfinterp.RBFInterpolator, str, bytes] = None, verbose: bool = False, show_frame: bool = False, eye_frames: bool = False) -> threading.Thread
    :module: vytal.client
    :noindex:
    
    Starts a thread that continuously predicts the gaze of a person using a webcam in the background of your code's execution.
    
    :param cam_id: (int) The ID of the webcam to use. Defaults to 0 (normally used).
    :param calib: (scipy.interpolate._rbfinterp.RBFInterpolator, str, bytes) A calibration object or data used for prediction.
    :param verbose: (bool) Whether to print the predictions to the console.
    :param show_frame: (bool) Whether to show the webcam feed.
    :param eye_frames: (bool) Whether to return the eye frames (128x128 images used for prediction)
    :return: The thread that is running the prediction loop.

.. py:function:: end_thread(thread: threading.Thread)
    :module: vytal.client
    :noindex:
    
    Ends the thread that is running the prediction loop.
    
    :param thread: The thread to end.
    :return: None

.. py:function:: predict_from_websocket(cam_id: int = 0, calib: Union[scipy.interpolate._rbfinterp.RBFInterpolator, str, bytes] = None, verbose: bool = False, show_frame: bool = False)
    :module: vytal.client
    :noindex:
    
    Asynchronously predicts the gaze of a person using a webcam in real time and returns back the predictions once run is complete/interrupted. 
    
    :param cam_id: (int) The ID of the webcam to use. Defaults to 0 (normally used).
    :param calib: (scipy.interpolate._rbfinterp.RBFInterpolator, str, bytes) A calibration object or data used for prediction.
    :param verbose: (bool) Whether to print the predictions to the console.
    :param show_frame: (bool) Whether to show the webcam feed.
    :return: All predictions during the time running at the end of run.

.. py:function:: real_time_pred(cam_id: int = 0, calib: Union[scipy.interpolate._rbfinterp.RBFInterpolator, str, bytes] = None, verbose: bool = False, show_frame: bool = False)
    :module: vytal.client
    :noindex:
    
    Synchronously runs predict_from_websocket using asyncio. 
    
    :param cam_id: (int) The ID of the webcam to use. Defaults to 0 (normally used).
    :param calib: (scipy.interpolate._rbfinterp.RBFInterpolator, str, bytes) A calibration object or data used for prediction.
    :param verbose: (bool) Whether to print the predictions to the console.
    :param show_frame: (bool) Whether to show the webcam feed.
    :return: Real-time predictions during the time running.

Advertising Technology
------------

.. py:module:: vytal.adtech

    The module for advertisement testing.


    

.. py:function:: analyze_eye_tracking_data(results, aois, fps, fixation_threshold_sec=0.5, distance_threshold=50)
   :module: vytal.adtech
   :noindex:

   Analyze eye tracking data to calculate metrics for Areas of Interest (AOIs) and general viewing behavior.

   This function processes a series of eye gaze predictions and calculates various metrics
   for predefined Areas of Interest (AOIs) as well as general viewing metrics.

   :param results: A list of dictionaries, each containing a 'PoG' key
                            representing the predicted x and y coordinates of the eye gaze as a tensor.
   :type results: list of dict
   :param aois: A dictionary where keys are AOI names and values are tuples representing
                the bounding rectangle of each AOI in the format (x1, y1, x2, y2).
   :type aois: dict
   :param fps: The frames per second of the recorded eye tracking data.
   :type fps: int
   :param fixation_threshold_sec: Minimum duration in seconds for a gaze point to be considered a fixation.
   :type fixation_threshold_sec: float
   :param distance_threshold: Maximum distance in pixels between consecutive gaze points to be considered part of the same fixation.
   :type distance_threshold: float

   :return: A tuple containing two dictionaries:
            
            1. aoi_metrics: A dictionary with metrics for each AOI:
               
               - 'TFF' (Time to First Fixation): Time in seconds before the AOI was first looked at.
               - 'Fixation_Count': Number of fixations on the AOI.
               - 'Total_Fixation_Duration': Total time in seconds spent looking at the AOI.
               - 'Avg_Fixation_Duration': Average duration of fixations on the AOI in seconds.
               - 'Revisits': Number of times the gaze returned to the AOI after looking elsewhere.
            
            2. general_metrics: A dictionary with general viewing metrics:
               
               - 'Entry_Point': The coordinates (x, y) where the gaze first entered the stimulus.
               - 'Exit_Point': The coordinates (x, y) where the gaze last left the stimulus.
   :rtype: tuple

   .. note::
      - This function assumes that the eye tracking data points are equally spaced in time.
      - The fixation detection uses a simple distance-based threshold method.

   :raises ValueError:
      - If ``results`` or ``aois`` is empty.
      - If ``fps``, ``fixation_threshold``, or ``distance_threshold`` are non-positive.
      - The dictionaries in ``results`` or the ``aois`` are invalid.

.. py:function:: define_aois(image_path: str) -> Dict[str, Tuple[float, float, float, float]]
   :module: vytal.adtech
   :noindex:

   Provides an interactive interface for defining Areas of Interest (AOIs) on an image.

   This function opens a matplotlib window displaying the specified image and allows
   the user to create, select, rename, move, and delete AOIs using mouse interactions
   and GUI buttons.

   :param image_path: Path to the image file on which AOIs will be defined.
   :type image_path: str

   :return: A dictionary where keys are AOI names and values are tuples representing 
            the bounding box of each AOI in the format (x1, y1, x2, y2), where (x1, y1) 
            is the top-left corner and (x2, y2) is the bottom-right corner of the AOI.
   :rtype: Dict[str, Tuple[float, float, float, float]]

   Functionality:

   - Create Mode: Left-click and drag to create a new AOI.
   - Select Mode: Click on an existing AOI to select it.
   - Rename: Type a new name in the text box and click 'Rename' to rename the selected AOI.
   - Delete: Click 'Delete' to remove the selected AOI.
   - Move: Click and drag an existing AOI to move it.
   - Mode Toggle: Use the 'Mode' button to switch between 'Create' and 'Select' modes.
   - Display AOIs: Press 'd' key to display current AOIs in the console.
   - Quit: Press 'q' key or click 'Close' button to finish and close the window.

   .. note::
      - The function will return an empty dictionary if there's an error reading the image file.
      - AOIs are represented as rectangles on the image.
      - The function uses matplotlib for rendering and interaction.

   :raises FileNotFoundError: If the specified image file is not found.
   :raises Exception: For any other error occurring while reading the image file.

.. py:function:: plot_gaze_path(results: List[Dict[str, torch.Tensor]], aois: Dict[str, Tuple[float, float, float, float]], image_path: str)
   :module: vytal.adtech
   :noindex:

   Visualizes the gaze path over the advertisement image.

   This function creates a plot showing the path of the viewer's gaze overlaid on the original image,
   along with the defined Areas of Interest (AOIs).

   :param results: A list of dictionaries, each containing a 'PoG' key
                            representing the predicted x and y coordinates of the eye gaze as a tensor.
   :type results: List[Dict[str, float]]
   :param aois: A dictionary where keys are AOI names and values are tuples representing 
                the bounding box of each AOI in the format (x1, y1, x2, y2).
   :type aois: Dict[str, Tuple[float, float, float, float]]
   :param image_path: Path to the image file used as the background for the visualization.
   :type image_path: str

   The function will:

   1. Load and display the background image.
   2. Plot the gaze path as a continuous line.
   3. Overlay scatter points representing individual gaze positions.
   4. Draw rectangles representing the AOIs.

   .. note::
      - The gaze path is plotted in blue with low opacity for clarity.
      - The scatter points are colored according to their temporal order using a 'cool' colormap.
      - AOIs are drawn as red rectangles with their names labeled.

   :raises FileNotFoundError: If the specified image file is not found.
   :raises Exception: For any other error occurring while reading the image file.

.. py:function:: generate_heatmap(results: List[Dict[str, torch.Tensor]], image_path: str. bins: int = 50)
   :module: vytal.adtech
   :noindex:

   Creates a heatmap of gaze intensity overlaid on the advertisement image.

   This function generates a heatmap visualization of the gaze data, showing areas of high and low
   gaze concentration overlaid on the original image.

   :param results: A list of dictionaries, each containing a 'PoG' key
                            representing the predicted x and y coordinates of the eye gaze as a tensor.
   :type results: List[Dict[str, float]]
   :param image_path: Path to the image file used as the background for the heatmap.
   :type image_path: str
   :param bins: Number of bins to use for the 2D histogram. Default is 50.
   :type bins: int

   The function will:

   1. Load and display the background image.
   2. Create a 2D histogram of the gaze data.
   3. Overlay the heatmap on the image using a 'hot' colormap with partial transparency.
   4. Add a colorbar to show the intensity scale.

   .. note::
      - The function includes error checking for empty results, negative coordinates, and coordinates
        outside the image dimensions.
      - The heatmap uses a 'hot' colormap where red indicates areas of high gaze concentration.

   :raises FileNotFoundError: If the specified image file is not found.
   :raises Exception: For any other error occurring while reading the image file or processing the data.    

.. py:function:: aoi_significance_test(group1_results: List[Dict[str, torch.Tensor]], group2_results: List[Dict[str, torch.Tensor]], aois: Dict[str, Tuple[float, float, float, float]], test: str = 't-test')
   :module: vytal.adtech
   :noindex:

   Performs statistical tests to compare AOI metrics between two groups.

   This function calculates and compares metrics for each Area of Interest (AOI) between two groups
   of gaze data, using either a t-test or Mann-Whitney U test.

   :param group1_results: Gaze data for the first group. Each dict should contain
                                             the 'PoG' key for gaze coordinates.
   :type group1_results: List[Dict[str, float]]
   :param group2_results: Gaze data for the second group. Same format as group1_results.
   :type group2_results: List[Dict[str, float]]
   :param aois: A dictionary where keys are AOI names and values are tuples representing 
                the bounding box of each AOI in the format (x1, y1, x2, y2).
   :type aois: Dict[str, Tuple[float, float, float, float]]
   :param test: Statistical test to use. Either 't-test' or 'mann-whitney'. Default is 't-test'.
   :type test: str

   :return: A dictionary containing the results of the statistical tests for each AOI. Each AOI entry includes:
            
            - 'group1_mean': Mean value for group 1
            - 'group2_mean': Mean value for group 2
            - 'statistic': The test statistic
            - 'p_value': The p-value of the test
   :rtype: Dict

   The function will:

   1. Calculate the proportion of gaze points within each AOI for both groups.
   2. Perform the specified statistical test to compare these proportions between the groups.
   3. Return the results including means, test statistic, and p-value for each AOI.

   .. note::
      - The function assumes that the AOIs and gaze coordinates use the same coordinate system.
      - The choice of test should be based on the nature of your data and experimental design.

   :raises ValueError:
      - If ``group1_results``, ``group2_results``, or ``aois`` is empty.
      - If an invalid test type is used.

.. py:function:: export_metrics_to_csv(aoi_metrics: Dict[str, Dict[str, float]], general_metrics: Dict[str, float], filename: str)
   :module: vytal.adtech
   :noindex:

   Exports calculated metrics to a CSV file for further analysis in other software.

   This function takes the metrics calculated for Areas of Interest (AOIs) and general viewing behavior
   and writes them to a CSV file in a structured format.

   :param aoi_metrics: A nested dictionary where the outer key is the AOI name,
                       and the inner dictionary contains various metrics as key-value pairs.
   :type aoi_metrics: Dict[str, Dict[str, float]]
   :param general_metrics: A dictionary of general metrics that apply to the entire viewing session.
   :type general_metrics: Dict[str, float]
   :param filename: The name of the output CSV file, including path if necessary.
   :type filename: str

   The function will:

   1. Create a new CSV file with the specified filename.
   2. Write AOI metrics, with each row containing the AOI name, metric name, and value.
   3. Write general metrics, with each row containing the metric name and value.

   The CSV structure will be::

       AOI Metrics
       AOI, Metric, Value
       [AOI metrics data]

       General Metrics
       Metric, Value
       [General metrics data]

   .. note::
      - If the file already exists, it will be overwritten.
      - The function uses the csv module to ensure proper CSV formatting.

   :raises IOError: If there's an error writing to the file (e.g., permission denied, disk full).
   :raises ValueError:
      - If ``aoi_metrics`` or ``general_metrics`` is not a dictionary.
      - Filename is not a csv.

HCI
---------

.. py:module:: vytal.hci
    
        The module for Human-Computer Interaction (HCI) testing.

.. py:function:: fixation_detection(gaze_points: List[Tuple[float, float, float], distance_threshold: float=30, time_threshold_ms: float=1500)
   :module: vytal.hci
   :noindex:

   Detects fixations in a series of gaze points using a dispersion-based algorithm.

   This function processes a list of gaze points and identifies fixations based on spatial proximity 
   and temporal duration.

   :param gaze_points: A list of tuples, each containing (x, y, timestamp) of a gaze point.
   :type gaze_points: List[Tuple[float, float, float]]
   :param distance_threshold: Maximum distance (in pixels) between a gaze point and the centroid 
                              of the current fixation to be considered part of that fixation. 
                              Default is 30 pixels.
   :type distance_threshold: float
   :param time_threshold_ms: Minimum duration (in milliseconds) for a group of gaze points to be 
                          considered a fixation. Default is 1500 milliseconds.
   :type time_threshold_ms: float

   :return: A list of detected fixations, where each fixation is represented as a tuple 
            containing ((centroid_x, centroid_y), duration).
   :rtype: List[Tuple[Tuple[float, float], float]]

   The function works as follows:

   1. Iterates through the gaze points.
   2. Groups consecutive points that are within the `distance_threshold` of the current fixation's centroid.
   3. When a point exceeds the distance threshold, it checks if the current group of points meets the `time_threshold_ms`.
   4. If the time threshold is met, it records the fixation and starts a new potential fixation group.
   5. After processing all points, it checks if the last group qualifies as a fixation.

   .. note::
      - This implementation uses a simple dispersion-based algorithm and may not account for more complex eye movement patterns.
      - The choice of `distance_threshold` and `time_threshold_ms` can significantly affect the results and should be tuned based on the specific use case and recording setup.

   :raises ValueError:
      - If ``distance_threshold`` or ``time_threshold_ms`` is non-positive.
      - If ``gaze_points`` is empty or contains invalid data.



.. py:function:: saccade_detection(gaze_points: List[Tuple[float, float, float]], velocity_threshold: float=1000)
   :module: vytal.hci
   :noindex:

   Detects saccades in a series of gaze points using a velocity-based algorithm.

   This function processes a list of gaze points and identifies saccades based on the velocity 
   of eye movement between consecutive points.

   :param gaze_points: A list of tuples, each containing (x, y, timestamp) of a gaze point. 
                       Timestamp is expected to be in milliseconds.
   :type gaze_points: List[Tuple[float, float, float]]
   :param velocity_threshold: Minimum velocity (in pixels per second) for an eye movement 
                              to be considered a saccade. Default is 1000 pixels/second.
   :type velocity_threshold: float

   :return: A list of detected saccades, where each saccade is represented as a dictionary 
            containing start_point, end_point, duration, amplitude, peak_velocity, and average_velocity.
   :rtype: List[Dict[str, Union[Tuple[float, float, float], float]]]

   The function works as follows:

   1. Iterates through the gaze points, calculating the velocity between consecutive points.
   2. When the velocity exceeds the threshold, it starts or continues a saccade.
   3. When the velocity drops below the threshold, it ends the current saccade (if any).
   4. For each saccade, it calculates:
      - Start and end points
      - Duration (in milliseconds)
      - Amplitude (total distance traveled)
      - Peak velocity
      - Average velocity

   .. note::
      - This implementation uses a simple velocity-based algorithm and may not account for more complex eye movement patterns.
      - The choice of `velocity_threshold` can significantly affect the results and should be tuned based on the specific use case and recording setup.
      - The function assumes that timestamps are in milliseconds and converts them to seconds for velocity calculations.

   :raises ValueError
      - If ``velocity_threshold`` is non-positive.
      - If ``gaze_points`` is empty or contains invalid data.

.. py:function:: detect_smooth_pursuit(gaze_points: List[Tuple[float, float, float]], time_window: int=100, velocity_threshold: float=30, direction_threshold: float=30)
   :module: vytal.hci
   :noindex:

   Detect smooth pursuit movements in a sequence of gaze points.

   This function analyzes a series of gaze points to identify segments that represent smooth pursuit eye movements,
   based on velocity and direction consistency over a specified time window.

   :param gaze_points: A list of tuples, each containing (x, y, timestamp) of a gaze point.
                       Timestamp is expected to be in milliseconds.
   :type gaze_points: List[Tuple[float, float, float]]
   :param time_window: Minimum duration (in milliseconds) for a segment to be considered smooth pursuit.
                       Default is 100 ms.
   :type time_window: int
   :param velocity_threshold: Maximum velocity (in pixels per second) for an eye movement 
                              to be considered smooth pursuit. Default is 30 pixels/second.
   :type velocity_threshold: float
   :param direction_threshold: Maximum change in direction (in degrees) allowed between consecutive
                               gaze points to be considered part of the same smooth pursuit.
                               Default is 30 degrees.
   :type direction_threshold: float

   :return: A list of detected smooth pursuit segments, where each segment is represented 
            as a tuple containing (start_index, end_index, duration).
   :rtype: List[Tuple[int, int, float]]

   The function works as follows:

   1. Iterates through the gaze points, calculating velocity and direction between consecutive points.
   2. Identifies continuous segments where:
      - The velocity remains below the `velocity_threshold`
      - The change in direction remains below the `direction_threshold`
      - The duration of the segment is at least `time_window`
   3. Records each qualifying segment as a smooth pursuit movement.

   .. note::
      - This implementation uses a simple algorithm based on velocity and direction consistency.
      - The choice of `velocity_threshold`, `direction_threshold`, and `time_window` can significantly 
        affect the results and should be tuned based on the specific use case and recording setup.
      - The function assumes that timestamps in `gaze_points` are in milliseconds.

   :raises ValueError:
      - If ``time_window``, ``velocity_threshold``, or ``direction_threshold`` is non-positive.
      - If ``gaze_points`` is empty or contains invalid data.

Cognitive Science
---------

.. py:module:: vytal.cogsci
    
        The module for Cognitive Science testing.
    
.. py:function:: EyeTrackingAnalyzer(data: List[Dict], sampling_rate: float)
   :noindex:

   :param data: (List[Dict]) A list of dictionaries containing eye tracking data.
   :param sampling_rate: (float) The sampling rate of the eye tracking data in Hz.

.. py:function:: detect_saccades(data: List[Dict], sampling_rate: float, velocity_threshold: float = 30, min_duration: float = 30, accel_threshold: float = 0, angle_type: str = 'face') -> Dict[str, List[Dict]]
   :module: vytal.cogsci
   :noindex:

   Detects saccades in eye tracking data for both left and right eyes.

    This function processes eye tracking data to identify saccades based on velocity and acceleration thresholds. It calculates velocities and accelerations from the eye angle data, detects potential saccades, and then filters and refines these detections to produce a final list of saccades for each eye.

   :param data: (List[Dict]) A list of dictionaries containing eye tracking data. Each dictionary should have keys 'time', 'left', 'right', 'face' (angles in radians).
   :param sampling_rate: (float) The sampling rate of the eye tracking data in Hz.
   :param velocity_threshold: (float, optional) The minimum peak velocity in deg/sec to consider a saccade. Defaults to 30 deg/sec.
   :param min_duration: (float, optional) The minimum duration of a saccade in milliseconds. Defaults to 30 ms.
   :param accel_threshold: (float, optional) The minimum peak acceleration in deg/sec^2 to consider a saccade. Defaults to 0 deg/sec^2 (no acceleration filtering).
   :param angle_type: (str, optional) The type of angle to use for saccade detection. Can be 'face', 'left', or 'right'. Defaults to 'face'.

   :return: A dictionary with keys 'left' and 'right', each containing a list of
        dictionaries. Each dictionary represents a detected saccade with the following keys:
            - 'start': Index of saccade start in the original data list
            - 'end': Index of saccade end
            - 'duration': Duration of the saccade in milliseconds
            - 'peak': Index of peak velocity
            - 'peak_velocity': Maximum velocity reached during the saccade (deg/sec)
            - 'amplitude': Change in eye angle during the saccade (degrees)
   :rtype: Dict[str, List[Dict]]

   .. note::
      - This function uses a sophisticated algorithm to detect saccades, including peak detection,
        acceleration thresholding, and removal of overlapping saccades.

   :raises ValueError: If the input data is empty or doesn't contain the required keys.

.. py:function:: detect_fixations(data, dispersion_threshold=1.0, duration_threshold=100, angle_type='face')
   :module: vytal.cogsci
   :noindex:

   Detects fixations in eye tracking data using a dispersion-based algorithm. Fixations are identified as periods where the gaze remains within a defined spatial threshold for a minimum time.

   :param data: (List[Dict]) Eye tracking data with each entry containing
         - 'time' (float): Timestamp in milliseconds.
         - 'POG_x' (float): Gaze position X-coordinate.
         - 'POG_y' (float): Gaze position Y-coordinate.
      Additionally requires 'left', 'right', 'face' if 'angle_type' is specified.
   :param dispersion_threshold: (float) Maximum allowed dispersion in gaze position units to qualify as a fixation.
   :param duration_threshold: (float) Minimum duration in milliseconds for a valid fixation.
   :param angle_type: (str) Specifies which angle data to use for additional fixation info ('face', 'left', 'right').

   :return: A list of dictionaries, each of which represents a detected fixation
        containing
            - 'start_index' (int): Start index of fixation in data.
            - 'end_index' (int): End index of fixation.
            - 'duration' (float): Duration of fixation in milliseconds.
            - 'centroid_x' (float): Average X-coordinate of fixation.
            - 'centroid_y' (float): Average Y-coordinate of fixation.
            - 'dispersion' (float): Calculated dispersion of fixation.
            - 'mean_angle' (float): Mean angle during the fixation according to 'angle_type'.
   :rtype: List[Dict]

   :raises ValueError: If data is empty or missing required keys.
    


