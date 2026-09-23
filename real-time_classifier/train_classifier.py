import os
import h5py
import logging
import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier

# Initialize logger
logging.basicConfig(
    format="%(asctime)s;%(levelname)s;%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

# Initialize colors
WHITE = "\033[1m\033[97m"
GREEN = "\033[1m\033[92m"
RESET = "\033[0m"


def extract_data_from_mat(directory: str, run: str):
    # Set location of run
    file = os.path.join(directory, run)

    # Import the .mat data into numpy arrays
    with h5py.File(file, "r") as f:
        # Each value is worth 20 ms of data
        trial_num = f["data"]["trial_num"][:].reshape(-1)
        spikes = f["data"]["SpikeCount"][:]
        who_move = f["data"]["Kinematics"]["ActualPos"][18, :]
        is_move = f["data"]["Kinematics"]["ActualPos"][15, :]

    return trial_num, spikes, who_move, is_move


def determine_movement_onset(
    ntrial: np.ndarray, is_move: np.ndarray, who_move: np.ndarray
):
    # Extract movement onset and determine who flipped the bucket
    mov_onset = []
    player = []
    trial_ids = np.unique(ntrial)

    # Iterate over all trials
    for exp in trial_ids:
        # Get the time step at which movement is initiated
        mov_onset.append(np.argmax(is_move[ntrial == exp] == 2))
        try:
            # If any of the players moved first, store the player ID
            player.append(who_move[ntrial == exp][is_move[ntrial == exp] == 2][0])
        except IndexError:
            # Otherwise add a flag indicating no player moved and the trial ended
            player.append(-1)

    return mov_onset, player


def create_2d_sequences(
    spikes: np.ndarray,
    mov_onset: int = 100,
    seq_size: int = 10,
    sampling: int = 20,
    jump: int = 10,
):
    seqs = []
    onset_horizon = []
    arr_size = spikes.shape[1]

    for i in range(0, arr_size - seq_size + 1, jump):
        # Calculate and store ms to movement onset
        onset_horizon.append(sampling * (mov_onset - (i + seq_size)))

        # Store the sequence
        seqs.append(spikes[:, i : i + seq_size])

    return np.array(seqs), np.array(onset_horizon)


# Set directory and files to train on
directory = r"D:\git\climber\src\FlipThatBucket"
train = ["trainData.mat"]

# Training data
X_train = []
y_train = []
train_trials = 0

logging.info("Extracting training data...")
for run in train:
    # Extract data
    trial_num, spikes, who_move, is_move = extract_data_from_mat(directory, run)

    # Extract movement onset and determine who flipped the bucket
    mov_onset, player = determine_movement_onset(trial_num, is_move, who_move)

    # Trials in which the scientist moved first are used to train the classifier
    scientist_moves = np.where(np.array(player) == 1)[0]

    obs = []
    for trial in scientist_moves:
        # Get spikes from the individual trial
        data = spikes[::5, :][:, trial_num == trial + 1]

        # Extract movement onset
        onset = mov_onset[trial]

        # Make sure we have enough pre-movement data
        if onset > 100:
            # Keep 2 seconds before onset and 1 second after onset
            chunk = data[:, onset - (50 * 2) : onset + 50]

            if len(chunk) and not np.isnan(chunk).any():
                obs.append(chunk)

    # Convert trials with the same length to a numpy array
    obs = np.array(obs)

    # Create 200 ms sliding windows and labels
    X = []
    y = []
    for ob in obs:
        train_trials += 1

        twodseqs, horizon = create_2d_sequences(ob, jump=1)

        # Keep windows up to movement onset
        index_onset = np.where(horizon == 0)[0][0] + 1
        twodseqs = twodseqs[:index_onset]

        X.append(twodseqs)
        y.append((horizon[:index_onset] <= 600) * 1)

    if len(X) > 0:
        X_train.append(np.concatenate(X))
        y_train.append(np.array(y))

# Build final training dataset by summing spike counts within each 200 ms window
X_train = np.sum(np.concatenate(X_train), axis=2)
y_train = np.concatenate(y_train).reshape(-1)

logging.info(f"{GREEN}Training data successfully extracted...{RESET}")
logging.info(f"{WHITE}=> Training Summary <={RESET}")
logging.info("Training trials: %0.f" % train_trials)
logging.info("Training instances: %0.f" % X_train.shape[0])
logging.info("Training features: %0.f" % X_train.shape[1])
logging.info(
    "Percentage of negative classes: %0.2f"
    % (np.sum(y_train == 0) * 100 / y_train.size)
)
logging.info(
    "Percentage of positive classes: %0.2f"
    % (np.sum(y_train == 1) * 100 / y_train.size)
)

# Train the Random Forest classifier
logging.info("Training classifier...")
robot = RandomForestClassifier(
    n_estimators=200,
    random_state=0,
    max_depth=15,
    n_jobs=-1,
    class_weight="balanced",
)
robot.fit(X_train, y_train)

# Save the classifier for real-time use
dump(robot, "robot.joblib")
logging.info(f"{GREEN}Classifier successfully trained and saved...{RESET}")
