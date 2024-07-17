API
===

.. autosummary::
   :toctree: generated

   vytal

Vytal.client
------------

.. py:module:: vytal.client(api_key: str, ipd: float = None)
    
   The main class for the Vytal API client.

   :param api_key: (str) The API key for the Vytal API.
   :param ipd: (float) The inter-pupillary distance of the person in the video. Defaults to None.

.. py:function:: predict_from_video(video_path: str, calib_mat: torch.Tensor = None, eye_frames: bool = False) -> Dict[str, Any]
   :module: vytal.client
   :noindex:

   Predicts the gaze of a person in a video file.

   :param video_path: (str) The path to the video file.
   :param calib_mat: (3x3 np.array) The calibration matrix for the camera. 
   :param eye_frames: (bool) Whether to return the eye frames (128x128 images used for prediction)
   :return: A dictionary containing the gaze predictions. 
   The keys are 'left', 'right', 'le_3d', 're_3d', 'hr', 'ht', 'blinked', 
   and optionally "right_eye_frame" and "left_eye_frame" if eye_frames is True.
   Each key maps to a tensor containing the predictions for each frame in the video.

.. py:function:: start_thread(cam_id: int = 0, calib_mat: np.array = None, verbose: bool = False, show_frame: bool = False, eye_frames: bool = False) -> threading.Thread
    :module: vytal.client
    :noindex:
    
    Starts a thread that continuously predicts the gaze of a person using a webcam in the background of your code's execution.
    
    :param cam_id: (int) The ID of the webcam to use. Defaults to 0 (normally used).
    :param calib_mat: (3x3 np.array) The calibration matrix for the camera.
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

.. py:function:: predict_from_websocket(cam_id: int = 0, calib_mat: np.array = None, verbose: bool = False, show_frame: bool = False, eye_frames: bool = False)
    :module: vytal.client
    :noindex:
    
    Asynchronously predicts the gaze of a person using a webcam in real time and returns back the predictions once run is complete/interrupted. 
    
    :param cam_id: (int) The ID of the webcam to use. Defaults to 0 (normally used).
    :param calib_mat: (3x3 np.array) The calibration matrix for the camera.
    :param verbose: (bool) Whether to print the predictions to the console.
    :param show_frame: (bool) Whether to show the webcam feed.
    :param eye_frames: (bool) Whether to return the eye frames (128x128 images used for prediction)
    :return: All predictions during time running at the end of run.

.. py:function:: real_time_pred(cam_id: int = 0, calib_mat: np.array = None, verbose: bool = False, show_frame: bool = False, eye_frames: bool = False)
    :module: vytal.client
    :noindex:
    
    Synchronously runs predict_from_websocket using asyncio. 
    
    :param cam_id: (int) The ID of the webcam to use. Defaults to 0 (normally used).
    :param calib_mat: (3x3 np.array) The calibration matrix for the camera.
    :param verbose: (bool) Whether to print the predictions to the console.
    :param show_frame: (bool) Whether to show the webcam feed.
    :param eye_frames: (bool) Whether to return the eye frames (128x128 images used for prediction)
    :return: All predictions during time running at the end of run. 

Vytal.adtech
------------

.. py:module:: vytal.adtech

    The module for the adtech helper functions.

.. py:function:: analyze_eye_tracking_data(results, aois, fps, fixation_threshold_sec=0.5, distance_threshold=50)
    :module: vytal.adtech
    :noindex:
   Analyze eye tracking data to calculate metrics for Areas of Interest (AOIs) and general viewing behavior.

   This function processes a series of eye gaze predictions and calculates various metrics
   for predefined Areas of Interest (AOIs) as well as general viewing metrics.

   :param results: A list of dictionaries, each containing 'pred_x' and 'pred_y' keys
                   representing the predicted x and y coordinates of the eye gaze.
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

.. py:function:: define_aois(image_path: str) -> Dict[str, Tuple[float, float, float, float]]:
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

.. py:function:: plot_gaze_path(results: List[Dict[str, float]], aois: Dict[str, Tuple[float, float, float, float]],
                   image_path: str):
    :module: vytal.adtech
    :noindex:
    Visualizes the gaze path over the advertisement image.

   This function creates a plot showing the path of the viewer's gaze overlaid on the original image,
   along with the defined Areas of Interest (AOIs).

   :param results: A list of dictionaries, each containing 'pred_x' and 'pred_y' keys
                   representing the predicted x and y coordinates of the eye gaze.
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

.. py:function:: generate_heatmap(results: List[Dict[str, float]], image_path: str. bins: int = 50):
    :module: vytal.adtech
    :noindex:
    Creates a heatmap of gaze intensity overlaid on the advertisement image.

   This function generates a heatmap visualization of the gaze data, showing areas of high and low
   gaze concentration overlaid on the original image.

   :param results: A list of dictionaries, each containing 'pred_x' and 'pred_y' keys
                   representing the predicted x and y coordinates of the eye gaze.
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

.. py:function:: aoi_significance_test(group1_results: List[Dict[str, float]], group2_results: List[Dict[str, float]],
                          aois: Dict[str, Tuple[float, float, float, float]], test: str = 't-test'):
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

.. py:function:: export_metrics_to_csv(aoi_metrics, general_metrics, filename)
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