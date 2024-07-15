API
===

.. autosummary::
   :toctree: generated

   vytal

..py:module:: vytal.client(api_key: str, ipd: float = None)
    
   The main class for the Vytal API client.

   :param api_key: The API key for the Vytal API.
   :param ipd: The inter-pupillary distance of the person in the video. Defaults to None.

.. py:function:: predict_from_video(video_path: str, calib_mat: torch.Tensor = None, eye_frames: bool = False) -> Dict[str, Any]
   :module: vytal.client
   :noindex:

   Predicts the gaze of a person in a video file.

   :param video_path: The path to the video file.
   :param calib_mat: The calibration matrix for the camera.
   :param eye_frames: Whether to return the eye frames (128x128 images used for prediction)
   :return: A dictionary containing the gaze predictions. 
   The keys are 'left', 'right', 'le_3d', 're_3d', 'hr', 'ht', 'blinked', 
   and optionally "right_eye_frame" and "left_eye_frame" if eye_frames is True.
   Each key maps to a tensor containing the predictions for each frame in the video.

.. py:function:: start_thread(cam_id: int = 0, calib_mat: np.array = None, verbose: bool = False, show_frame: bool = False, eye_frames: bool = False) -> threading.Thread
    :module: vytal.client
    :noindex:
    
    Starts a thread that continuously predicts the gaze of a person using a webcam in the background of your code's execution.
    
    :param cam_id: The ID of the webcam to use. Defaults to 0 (normally used).
    :param calib_mat: The calibration matrix for the camera.
    :param verbose: Whether to print the predictions to the console.
    :param show_frame: Whether to show the webcam feed.
    :param eye_frames: Whether to return the eye frames (128x128 images used for prediction)
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
    
    :param cam_id: The ID of the webcam to use. Defaults to 0 (normally used).
    :param calib_mat: The calibration matrix for the camera.
    :param verbose: Whether to print the predictions to the console.
    :param show_frame: Whether to show the webcam feed.
    :param eye_frames: Whether to return the eye frames (128x128 images used for prediction)
    :return: All predictions during time running at the end of run.

.. py:function:: real_time_pred(cam_id: int = 0, calib_mat: np.array = None, verbose: bool = False, show_frame: bool = False, eye_frames: bool = False)
    :module: vytal.client
    :noindex:
    
    Synchronously runs predict_from_websocket using asyncio. 
    
    :param cam_id: The ID of the webcam to use. Defaults to 0 (normally used).
    :param calib_mat: The calibration matrix for the camera.
    :param verbose: Whether to print the predictions to the console.
    :param show_frame: Whether to show the webcam feed.
    :param eye_frames: Whether to return the eye frames (128x128 images used for prediction)
    :return: All predictions during time running at the end of run. 