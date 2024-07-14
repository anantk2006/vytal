Usage
=====

.. _installation:

Installation
------------

To use Vytal, first install it using pip:

.. code-block:: console

   (.venv) $ pip install vytal

Getting gaze predictions
----------------

To start recording on the webcam in live time in the background of your own code execution
, you can use the ``vytal.start_thread()`` function:

.. autofunction:: vytal.start_thread

For example:

>>> from vytal.client import Client
>>> key = "<your-key-here>"
>>> api_client = Client(key, ipd=65)

>>> vytal_api_loop = api_client.start_thread()

>>> while True:
    >>> print(api_client.preds[-1])

This code will output the gaze predictions based on webcam footage. 


