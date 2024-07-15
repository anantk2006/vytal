Usage
=====

.. _installation:

Installation
------------

To use Vytal, first install it using pip:

.. code-block:: console

   (.venv) $ pip install vytal

Getting gaze predictions in background
----------------

To start recording on the webcam in live time in the background of your own code execution
, you can use the ``start_thread()`` function:

.. autofunction:: vytal.client.start_thread

For example:

>>> from vytal.client import Client
>>> key = "<your-key-here>"
>>> api_client = Client(key, ipd=65)
>>> #The API Client is now initialized.
>>> vytal_api_loop = api_client.start_thread()
>>> # The API Client is now running in the background, viewing and predicting based on the webcam.
>>> while True:
>>>     print(api_client.preds[-1])
{'left': tensor([[[-0.1950], [-0.0949]]]), 'right': tensor([[[ 0.0131], [-0.2413]]]), 
'le_3d': tensor([[[ 61.3681], [ 15.6595], [541.7325]]]), 're_3d': tensor([[[  1.2703], [ 13.9736], [542.0076]]]),
 'hr': tensor([[[ 0.9996, -0.0229,  0.0156], [ 0.0273,  0.9126, -0.4079], [-0.0049,  0.4081,  0.9129]]]), 
 'ht': tensor([[[ 29.7278], [ 59.1505], [522.4246]]]), 'blinked': tensor([0.])}
``api_client.preds`` will contain the predictions. Whenever you want the thread to end:
>>> api_client.end_thread(vytal_api_loop)





This code will output the gaze predictions based on webcam footage. 


Getting gaze predictions from a video file
----------------

To get gaze predictions from a video file, you can use the ``predict_from_video()`` function:

.. autofunction:: vytal.client.predict_from_video

For example:

>>> from vytal.client import Client
>>> key = "<your-key-here>"
>>> api_client = Client(key, ipd=65)
>>> preds = api_client.predict_from_video("path/to/video.mp4")
>>> print(preds)
>>> # ``preds`` will contain the predictions in a dictionary with each key mapping to that metric to all frames.





