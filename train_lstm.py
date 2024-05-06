# -*- coding: utf-8 -*-

# mount Google Drive
from google.colab import drive
drive.mount("/content/drive/")

# extract training data ZIP files
from typing import List
import tqdm
import os
def extract_data(dst_folder_name : str, src_folder_name : str) -> None:
    assert isinstance(dst_folder_name, str)
    assert isinstance(src_folder_name, str)
    filenames : List[ str ] = [
        filename for x in os.listdir(src_folder_name)
            for filename in (os.path.join(src_folder_name, x),)
                if os.path.splitext(filename)[1] == ".zip"
    ]
    os.makedirs(dst_folder_name, exist_ok = True)
    for filename in tqdm.tqdm(filenames, total = len(filenames), desc = "Extracting zip files"):
        os.system(f"""unzip -o -d '{dst_folder_name}' '{filename}'""")
if (__name__ == "__main__"):
    src_folder_name : str = "/content/drive/MyDrive/multimedia_hw4/"
    dst_folder_name : str = "/content/data/"
    extract_data(dst_folder_name, src_folder_name)

# preprocess audio files
from typing import Optional, List
# import librosa.display
import numpy as np
import librosa
import shutil
import tqdm
import os
def compute_mfcc(audio_file_path : str,
                 n_mfcc          : Optional[ int ] = 20, #13,
                 sr              : Optional[ int ] = 22050,
                 hop_length      : Optional[ int ] = 512,
                 n_fft           : Optional[ int ] = 2048) -> np.ndarray:
    assert isinstance(audio_file_path, str)
    assert isinstance(n_mfcc, int)
    assert isinstance(sr, int)
    assert isinstance(hop_length, int)
    assert isinstance(n_fft, int)
    assert os.path.isfile(audio_file_path)
    y, sr = librosa.load(audio_file_path, sr = sr)
    mfccs = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = n_mfcc, n_fft = n_fft, hop_length = hop_length)
    return mfccs
def compute_mel_spectrogram(audio_file_path : str,
                            n_mels          : Optional[ int ] = 50,
                            sr              : Optional[ int ] = 22050,
                            hop_length      : Optional[ int ] = 512,
                            n_fft           : Optional[ int ] = 2048) -> np.ndarray:
    assert isinstance(audio_file_path, str)
    assert isinstance(n_mels, int)
    assert isinstance(sr, int)
    assert isinstance(hop_length, int)
    assert isinstance(n_fft, int)
    assert os.path.isfile(audio_file_path)
    y, sr = librosa.load(audio_file_path, sr = sr)
    S = librosa.feature.melspectrogram(y = y, sr = sr, n_fft = n_fft, hop_length = hop_length, n_mels = n_mels)
    log_S = librosa.power_to_db(S, ref = np.max)
    return log_S
def preprocess_audios(src_folder : str,
                      dst_folder : str) -> None:
    assert isinstance(src_folder, str)
    assert isinstance(dst_folder, str)
    assert os.path.exists(src_folder)
    class_names : List[ str ] = [
        class_name for class_name in os.listdir(src_folder)
            if not class_name.startswith("_")
    ]
    class_folders : List[ str ] = [
        class_folder for class_name in class_names
            for class_folder in (os.path.join(src_folder, class_name),)
                if os.path.exists(class_folder) and not class_name.startswith("_")
    ]
    def files_in_folder(folder_name : str, extension : Optional[ str ] = ".wav") -> List[ str ]:
        assert isinstance(folder_name, str)
        assert os.path.exists(folder_name)
        return [
            filename for fn in os.listdir(folder_name)
                for filename in (os.path.join(folder_name, fn),)
                    if os.path.isfile(filename) and not fn.startswith("_") and fn.endswith(extension)
        ]
    if os.path.exists(dst_folder):
        shutil.rmtree(dst_folder)
    for class_idx, class_name in tqdm.tqdm(enumerate(class_names), total = len(class_names), desc = "Preprocessing audio files"):
        class_folder : str = class_folders[class_idx]
        audio_files : List[ str ] = files_in_folder(class_folder, extension = ".wav")
        for src_filename in audio_files:
            # MFCC
            dst_filename_1 : str = os.path.join(dst_folder, "mfcc", class_name, os.path.splitext(os.path.basename(src_filename))[0] + ".npy")
            os.makedirs(os.path.dirname(dst_filename_1), exist_ok = True)
            np.save(dst_filename_1, compute_mfcc(src_filename))
            # melcepstrum
            dst_filename_2 : str = os.path.join(dst_folder, "melc", class_name, os.path.splitext(os.path.basename(src_filename))[0] + ".npy")
            os.makedirs(os.path.dirname(dst_filename_2), exist_ok = True)
            np.save(dst_filename_2, compute_mel_spectrogram(src_filename))
if (__name__ == "__main__"):
    # res = compute_mfcc("/content/data/blues/blues.00000.wav")
    # print(res.shape) # (13, 1293)
    # res = compute_mel_spectrogram("/content/data/blues/blues.00000.wav")
    # print(res.shape) # (128, 1293)
    src_folder : str = "/content/data/"
    dst_folder : str = "/content/data_pre/"
    preprocess_audios(src_folder, dst_folder)

# folding training data
from typing import List
import numpy as np
import shutil
import tqdm
import os
def split_fold_data(src_folder : str,
                    dst_folder : str,
                    fold_ratio : float) -> None:
    assert isinstance(src_folder, str)
    assert isinstance(dst_folder, str)
    assert isinstance(fold_ratio, float)
    assert os.path.exists(src_folder)
    assert 0 < fold_ratio <= 1
    np.random.seed(1207)
    def find_class_names(folder_name : str) -> List[ str ]:
        assert isinstance(folder_name, str)
        assert os.path.exists(folder_name)
        return [
            subfolder for subfolder in os.listdir(folder_name)
                if not subfolder.startswith("_")
        ]
    def find_class_folders(folder_name : str, class_names : List[ str ]) -> List[ str ]:
        assert isinstance(folder_name, str)
        assert isinstance(class_names, list)
        return [
            os.path.join(folder_name, class_name)
                for class_name in class_names
        ]
    def find_class_files(folder_name : str) -> List[ str ]:
        assert isinstance(folder_name, str)
        # print(folder_name)
        assert os.path.exists(folder_name)
        return [
            filename for x in os.listdir(folder_name)
                for filename in (os.path.join(folder_name, x),)
                    if os.path.isfile(filename) and not x.startswith("_")
        ]
    class_names : List[ str ] = find_class_names(src_folder)
    if os.path.exists(dst_folder):
        shutil.rmtree(dst_folder)
    class_folders : List[ str ] = find_class_folders(src_folder, class_names)
    for class_idx, class_name in tqdm.tqdm(enumerate(class_names), total = len(class_names), desc = "Folding training data"):
        files_in_class : List[ str ] = find_class_files(class_folders[class_idx])
        indices : np.ndarray = np.arange(len(files_in_class))
        np.random.shuffle(indices)
        index_folds : List[ np.ndarray ] = np.array_split(indices, np.ceil(1 / fold_ratio))
        for fold_idx, index_fold in enumerate(index_folds, start = 1):
            for file_index in index_fold:
                src_filename : str = files_in_class[file_index]
                dst_filename : str = os.path.join(dst_folder, str(fold_idx), class_name, os.path.basename(src_filename))
                os.makedirs(os.path.dirname(dst_filename), exist_ok = True)
                shutil.copy(src_filename, dst_filename)
if (__name__ == "__main__"):
    src_folder_name : str = "/content/data_pre/melc/"
    dst_folder_name : str = "/content/data_melc/"
    split_fold_data(src_folder_name, dst_folder_name, fold_ratio = 0.2)
    src_folder_name : str = "/content/data_pre/mfcc/"
    dst_folder_name : str = "/content/data_mfcc/"
    split_fold_data(src_folder_name, dst_folder_name, fold_ratio = 0.2)

from tensorflow.keras.utils import to_categorical
from typing import Optional, Tuple, List, Dict
import tensorflow as tf
import numpy as np
import os
class NpyDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory  : str,
                       batch_size : Optional[ int ]  = 32,
                       shuffle    : Optional[ bool ] = True) -> None:
        assert isinstance(directory, str)
        assert isinstance(batch_size, int)
        assert isinstance(shuffle, bool)
        assert os.path.exists(directory)
        self.directory : str = directory
        self.batch_size : int = batch_size
        self.shuffle : bool = shuffle
        self.classes : List[ str ] = sorted(os.listdir(directory))
        self.num_classes : int = len(self.classes)
        self.indices : List[ Tuple[ int, str ] ] = self._get_indices()
    def _get_indices(self) -> List[ Tuple[ int, str ] ]:
        indices = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.directory, class_name)
            for filename in os.listdir(class_dir):
                indices.append((class_idx, os.path.join(class_dir, filename)))
        if self.shuffle:
            np.random.shuffle(indices)
        return indices
    def __len__(self) -> int:
        return len(self.indices) // self.batch_size
    def __getitem__(self, index : int) -> Tuple[ np.ndarray, np.ndarray ]:
        assert isinstance(index, int)
        def pad_or_truncate(array : np.ndarray, length : Optional[ int ] = 1300) -> np.ndarray:
            if (array.shape[0] == length):
                return array
            if (array.shape[0] > length):
                return array[:length]
            return np.pad(array, ((0, length - array.shape[0]), (0, 0)))
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X = []
        y = []
        for class_idx, filepath in batch_indices:
            data = np.load(filepath)
            X.append(pad_or_truncate(np.transpose(data)))
            y.append(class_idx)
        X = np.array(X)
        y = np.array(y)
        y = to_categorical(y, num_classes=self.num_classes)
        return X, y
def create_data_loaders(folder_name : str,
                        batch_size  : int) -> Dict[ str, NpyDataGenerator ]:
    assert isinstance(folder_name, str)
    assert os.path.exists(folder_name)
    assert isinstance(batch_size, int)
    assert batch_size > 0
    fold_names : List[ str ] = [
        fold_name for fold_name in os.listdir(folder_name)
            if not fold_name.startswith("_")
    ]
    return {
        fold_name : NpyDataGenerator(directory = os.path.join(folder_name, fold_name), batch_size = batch_size)
            for fold_name in fold_names
    }
if (__name__ == "__main__"):
    melc_folder_name : str = "/content/data_melc/"
    mfcc_folder_name : str = "/content/data_mfcc/"
    batch_size : int = 32
    melc_loaders : Dict[ str, NpyDataGenerator ] = create_data_loaders(melc_folder_name, batch_size)
    mfcc_loaders : Dict[ str, NpyDataGenerator ] = create_data_loaders(mfcc_folder_name, batch_size)
    print(len(melc_loaders))
    print(len(mfcc_loaders))

from tensorflow.keras.layers import Bidirectional, LeakyReLU, Dense, Input, LSTM
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
def build_lstm_classifier(timesteps : int, vector_size : int) -> Model:
    assert isinstance(timesteps, int)
    assert isinstance(vector_size, int)
    assert timesteps > 0
    assert vector_size > 0
    return Sequential([
        Input(shape = (timesteps, vector_size)),
        LSTM(64, return_sequences = True),
        LSTM(64),
        Dense(10, activation = "softmax")
    ])
if (__name__ == "__main__"):
    lstm_parameters = {
        "lstm": {
            "mfcc": {
                "timesteps": 1300,
                "vector_size": 20
            },
            "melc": {
                "timesteps": 1300,
                "vector_size": 50
            }
        }
    }
    num_folds : int = 5
    lstm_classifiers = {
        "lstm": {
            "mfcc": [
                build_lstm_classifier(**lstm_parameters["lstm"]["mfcc"]) for _ in range(num_folds)
            ],
            "melc": [
                build_lstm_classifier(**lstm_parameters["lstm"]["melc"]) for _ in range(num_folds)
            ]
        }
    }
    for model_type in [ "lstm" ]:
        for feature_type in [ "mfcc", "melc" ]:
            for model in lstm_classifiers[model_type][feature_type]:
                model.compile(optimizer = Adam(learning_rate = 1e-3), loss = "categorical_crossentropy", metrics = [ "Accuracy" ])
    print(lstm_classifiers)
    lstm_classifiers["lstm"]["mfcc"][0].summary()

from matplotlib import pyplot as plt

from difflib import SequenceMatcher

from typing import *

import numpy as np

import json

import cv2

import re

class TrainingHistory(object):

    @staticmethod
    def longest_common_substring(*strings) -> str:

        assert all(isinstance(s, str) for s in strings)

        assert len(strings) >= 2

        common_substring = strings[0]

        for i in range(1, len(strings)):

            matcher = SequenceMatcher(None, common_substring, strings[i])

            match = matcher.find_longest_match(0, len(common_substring), 0, len(strings[i]))

            common_substring = (("") if (match.size <= 0) else (common_substring[ match.a : match.a + match.size ]))

        return common_substring

    def __init__(self) -> None:
        self._num_epochs = 0
        self._history = {}

    def update(self, **metrics) -> None:
        for metric, value in metrics.items():
            self._history[metric] = self._history.get(metric, []) + [ value ]

    def clear(self) -> None:
        self._num_epochs = 0
        self._history = {}

    @classmethod
    def load(cls, filename : str) -> "TrainingHistory":

        assert isinstance(filename, str)

        training_history = TrainingHistory()

        training_history._history = json.load(open(filename, mode = "r", encoding = "utf-8"))

        return training_history

    def __contains__(self, key : str) -> bool:

        assert isinstance(key, str)

        return self._history.__contains__(key)

    def __getitem__(self, key : str) -> list:

        assert self.__contains__(key)

        return self._history[key]

    def export(self, metrics : List[ str ]) -> np.ndarray:

        assert isinstance(metrics, list)

        assert len(metrics) > 0

        assert all(self._history.__contains__(metric) for metric in metrics)

        y_label = " ".join(map(lambda x : x.capitalize(), re.split("[^a-zA-Z0-9]+", metrics[0])))

        if (len(metrics) > 1):
            y_label = "".join(filter(lambda x : x.isalpha(), self.longest_common_substring(*metrics))).capitalize()

        min_length = min([  len(self._history[metric]) for metric in metrics  ])

        for metric in metrics:
            # print(metric)
            plt.plot(self._history[metric][0:min_length])

        plt.xlabel("Epochs")

        plt.ylabel(y_label)

        plt.title(f"{y_label} History")

        plt.xticks(ticks = range(0, min_length, 5), labels = range(1, min_length + 1, 5))

        plt.legend(metrics)

        canvas = plt.gcf().canvas;  canvas.draw();  (w, h) = plt.gcf().get_size_inches() * plt.gcf().get_dpi()

        image = cv2.cvtColor(np.frombuffer(canvas.tostring_rgb(), dtype = np.uint8).reshape(int(h), int(w), 3), cv2.COLOR_RGB2BGR)

        plt.clf()

        return image

from typing import *

class EarlyStopping(object):

    def __init__(self, training_history : TrainingHistory,
                       monitor_metric   : str,
                       patience         : int, *,
                       increase_improve : Optional[ bool ] = True
            ) -> None:

        assert isinstance(training_history, TrainingHistory)

        assert isinstance(monitor_metric, str)

        assert isinstance(patience, int)

        assert patience > 0

        assert isinstance(increase_improve, bool)

        self._training_history : TrainingHistory = training_history

        self._monitor_metric   : str             = monitor_metric

        self._patience         : int             = patience

        self._patience_counter : int             = patience

        self._increase_improve : bool            = increase_improve

        self._best_value       : float           = float(f"{('-') if (increase_improve) else ('')}inf")

    def __is_improvement(self, new_value : float) -> bool:
        return ((new_value > self._best_value) if (self._increase_improve) else (new_value < self._best_value))

    def should_stop(self) -> bool:

        assert self._training_history.__contains__(self._monitor_metric)

        latest_value : float = self._training_history[self._monitor_metric][-1]

        if (self.__is_improvement(latest_value)):
            self._best_value       = latest_value
            self._patience_counter = self._patience
            return False

        self._patience_counter -= 1

        if (self._patience_counter <= 0):
            return True

from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Model
from collections import defaultdict
from typing import Union, Dict
from datetime import datetime
import json
import sys
import gc
import os
def train_classifier(model            : Model,
                     dataloaders      : Dict[ str, NpyDataGenerator ],
                     validation_label : str,
                     epochs           : int,
                     patience         : int) -> Dict[ str, List[ Union[ float, int ] ] ]:
    assert isinstance(model, Model)
    assert isinstance(dataloaders, dict)
    assert isinstance(validation_label, str)
    assert isinstance(epochs, int)
    assert epochs > 0
    assert validation_label in dataloaders
    history = TrainingHistory()
    early_stopping = EarlyStopping(history, monitor_metric = "val_loss", patience = patience, increase_improve = False)
    for epoch in range(epochs):
        _history : dict = {
            "train_loss": -1,
            "train_Accuracy": -1,
            "val_loss": -1,
            "val_Accuracy": -1
        }
        train_accuracy : List[ float ] = []
        train_loss : float = 0
        gen_counter : int = 0
        for gen_cls, data_generator in dataloaders.items():
            if (gen_cls != validation_label):
                gen_counter += 1
                sys.stdout.write("\rEPOCH {} ({}%)".format(epoch + 1, round(100 * gen_counter / (len(dataloaders) - 1))))
                sys.stdout.flush()
                hist = model.fit(data_generator, verbose = False)
                train_accuracy.append(hist.history["Accuracy"][0])
                train_loss += hist.history["loss"][0]
        _history["train_loss"] = train_loss
        _history["train_Accuracy"] = np.mean(train_accuracy)
        valid_loss, valid_accuracy = model.evaluate(dataloaders[validation_label], verbose = False)
        _history["val_loss"] = valid_loss
        _history["val_Accuracy"] = valid_accuracy
        history.update(**_history)
        print(" [ train_acc: {}% ] [ train_loss: {} ] [ val_acc: {}% ] [ val_loss: {} ]".format(
            round(_history["train_Accuracy"] * 100, 2),
            round(_history["train_loss"], 4),
            round(_history["val_Accuracy"] * 100, 2),
            round(_history["val_loss"], 4),
        ))
        # print(history._history)
        if (early_stopping.should_stop()):
            break
    return history._history
if (__name__ == "__main__"):
    history_name : str = "/content/drive/MyDrive/multimedia_hw4_train_history/lstm/{}.json".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(os.path.dirname(history_name), exist_ok = True)
    lstm_train_parameters = {
        "epochs": 500,
        "patience": 10
    }
    lstm_train_history = {
        "lstm": {
            "mfcc": [

            ],
            "melc": [

            ]
        }
    }
    # for val_idx, validation_label in enumerate([ "1", "2", "3", "4", "5" ]):
    #     lstm_train_history["lstm"]["mfcc"].append(train_classifier(lstm_classifiers["lstm"]["mfcc"][val_idx], mfcc_loaders, validation_label = validation_label, **lstm_train_parameters))
    #     json.dump(lstm_train_history, open(history_name, mode = "w"))
    #     gc.collect()
    #     clear_session()
    for val_idx, validation_label in enumerate([ "1", "2", "3", "4", "5" ]):
        lstm_train_history["lstm"]["melc"].append(train_classifier(lstm_classifiers["lstm"]["melc"][val_idx], melc_loaders, validation_label = validation_label, **lstm_train_parameters))
        json.dump(lstm_train_history, open(history_name, mode = "w"))
        gc.collect()
        clear_session()

