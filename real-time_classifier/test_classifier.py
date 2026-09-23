import logging
import time
import threading
import numpy as np
import pyrtma
from joblib import load
from sklearn.base import BaseEstimator
import climber_core_utilities.load_config as load_config
import climber_message as md

# Initialize logger
logging.basicConfig(
    format="%(asctime)s;%(levelname)s;%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def predict(new_data: np.ndarray, model: BaseEstimator):
    # Use the classifier's standard class prediction
    pred = model.predict(new_data)

    if np.any(pred):
        print("TRUE")
        return 1.0

    print("FALSE")
    return 0.0


def run_classifier(model: BaseEstimator, lock):
    global last_prediction
    global serial_no
    global prediction_serial_nr
    global time_since_last_prediction
    global bufferData
    global done

    last_prediction_time = time.time()

    while not done:
        with lock:
            latest_data = bufferData  # local copy of buffer data
            prediction_serial_nr = (
                serial_no  # last SPM_SPIKECOUNT number represented in this buffer
            )

        prediction = 0
        if latest_data.shape[0] >= bufferLength:
            # Make sure the incoming stream has the expected shape
            assert latest_data.T[::5, :].shape == (
                nFeatures,
                bufferLength,
            ), "Data stream does not have the expected shape"

            # Sum spike counts for each channel across the last 10 x 20 ms bins
            batch = np.sum(latest_data.T[::5, :], axis=1)[np.newaxis]
            assert batch.shape == (
                1,
                nFeatures,
            ), "Batch does not have the expected shape"

            prediction = predict(batch, model)

        with lock:
            last_prediction = prediction
            time_since_last_prediction = time.time() - last_prediction_time

        last_prediction_time = time.time()


def send_extraction_response(serial_no, mod):
    out_msg = md.MDF_EXTRACTION_RESPONSE()
    out_msg.header.serial_no = serial_no
    out_msg.decodertype = "ChapmanClassifier"
    out_msg.decoderloc = "robot.joblib"
    mod.send_message(out_msg)


# Load classifier and set variables for real-time processing
logging.info("Loading classifier...")
robot = load("robot.joblib")

logging.info("Setting variables for real-time processing...")
bufferLength = 10  # 10 x 20 ms samples = 200 ms
nFeatures = 256
logging.info("Length of buffer: %0.f" % bufferLength)
logging.info("Number of features: %0.f" % nFeatures)

empty_list = [0.0] * 30
command = empty_list
predIdx = 6  # classifier prediction
serialIdx = 7  # SPM_SPIKECOUNT serial number at prediction time
predTimeIdx = 8  # elapsed time since the previous classifier prediction (seconds)

# Connect to MessageManager
MID = md.MID_STIM_THRESH_GAME
mod = pyrtma.Client(MID, 0)
MMM_IP = load_config.get_mm_server()
# "192.168.1.40:7111" (Chicago)
# "192.168.110.40:7111" (Pittsburgh)
# "localhost:7111" (DEBUG)
mod.connect(MMM_IP)
mod.subscribe(
    [
        md.MT_SPM_SPIKECOUNT,
        md.MT_EXTRACTION_REQUEST,
        md.MT_EXIT,
    ]
)
mod.send_module_ready()

# Keep a running buffer and predict over the last 200 ms
bufferData = np.array([])
done = False
last_prediction = 0
serial_no = 0
time_since_last_prediction = 0
prediction_serial_nr = 0

# Create shared lock
lock = threading.Lock()

# Run classifier thread
x = threading.Thread(target=run_classifier, args=(robot, lock))
logging.info("Starting prediction thread...")
x.start()

logging.info("Starting continuous loop, reading messages...")
while not done:
    # Read latest RTMA messages
    msg = mod.read_message(0.001)
    if msg is None:
        continue

    if msg.type_id == md.MT_SPM_SPIKECOUNT:  # new spike data every 20 ms
        SPM_data = msg.data

        with lock:
            serial_no = SPM_data.header.serial_no

        command = empty_list

        # Append new data to the spike buffer
        with lock:
            if bufferData.shape[0] >= bufferLength:
                bufferData = np.delete(bufferData, 0, 0)

            bufferData = np.append(bufferData, msg.data.counts)
            bufferData = np.reshape(bufferData, [-1, 1280])

        with lock:
            command[predIdx] = last_prediction
            command[serialIdx] = prediction_serial_nr
            command[predTimeIdx] = time_since_last_prediction

        # Send classifier prediction
        out_msg = md.MDF_CONTROL_SPACE_COMMAND()
        out_msg.command = command
        out_msg.header.serial_no = serial_no
        mod.send_message(out_msg)

    elif msg.type_id == md.MT_EXTRACTION_REQUEST:
        send_extraction_response(serial_no, mod)

    elif msg.type_id == md.MT_EXIT:
        done = True

mod.disconnect()

