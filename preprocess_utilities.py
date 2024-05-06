# -*- coding: utf-8 -*-
from google.colab import drive
drive.mount("/content/drive/")

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

from typing import List
import numpy as np
import shutil
import tqdm
import os
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
    assert os.path.exists(folder_name)
    return [
        filename for x in os.listdir(folder_name)
            for filename in (os.path.join(folder_name, x),)
                if os.path.isfile(filename) and not x.startswith("_")
    ]
def split_validation_data(dst_folder_name : str,
                          src_folder_name : str,
                          validation_rate : float) -> None:
    assert isinstance(dst_folder_name, str)
    assert isinstance(src_folder_name, str)
    assert isinstance(validation_rate, float)
    assert os.path.exists(src_folder_name)
    assert validation_rate >= 0.0
    if os.path.exists(dst_folder_name):
        shutil.rmtree(dst_folder_name)
    os.makedirs(dst_folder_name, exist_ok = True)
    class_names : List[ str ] = find_class_names(src_folder_name)
    class_folders : List[ str ] = find_class_folders(src_folder_name, class_names)
    for class_idx, class_name in tqdm.tqdm(enumerate(class_names), total = len(class_names), desc = "Splitting data"):
        class_folder : str = class_folders[class_idx]
        class_files : List[ str ] = find_class_files(class_folder)
        num_files : int = len(class_files)
        num_train : int = int(np.ceil(num_files * (1 - validation_rate)))
        indices : List[ int ] = list(range(num_files))
        train_indices : List[ int ] = np.random.choice(indices, size = num_train, replace = False)
        valid_indices : List[ int ] = list(set(indices).difference(train_indices))
        for index in train_indices:
            src_file : str = class_files[index]
            dst_file : str = os.path.join(dst_folder_name, "train", class_name, os.path.basename(src_file))
            os.makedirs(os.path.dirname(dst_file), exist_ok = True)
            shutil.copy(src_file, dst_file)
        for index in valid_indices:
            src_file : str = class_files[index]
            dst_file : str = os.path.join(dst_folder_name, "valid", class_name, os.path.basename(src_file))
            os.makedirs(os.path.dirname(dst_file), exist_ok = True)
            shutil.copy(src_file, dst_file)
if (__name__ == "__main__"):
    src_folder_name : str = "/content/data/"
    dst_folder_name : str = "/content/data_split/"
    split_validation_data(dst_folder_name, src_folder_name, validation_rate = 0.2)

from typing import Optional, Tuple, List
import soundfile as sf
import librosa
import tqdm
def change_speed_and_pitch(input_file   : str,
                           output_file  : str,
                           speed_factor : Optional[ float ] = 1.0,
                           pitch_shift  : Optional[ float ] = 0) -> None:
    assert isinstance(input_file, str)
    assert isinstance(output_file, str)
    assert isinstance(speed_factor, float)
    assert isinstance(pitch_shift, float)
    assert os.path.isfile(input_file)
    assert speed_factor > 0
    assert pitch_shift >= 0
    y, sr = librosa.load(input_file)
    y_speed = librosa.effects.time_stretch(y, rate = speed_factor)
    y_pitch = librosa.effects.pitch_shift(y_speed, sr = sr, n_steps=pitch_shift)
    sf.write(output_file, y_pitch, sr)
def augment_audio_files(folder_name : str,
                        quantity    : int,
                        speed_range : Tuple[ int, int ],
                        pitch_range : Tuple[ int, int ]) -> None:
    assert isinstance(folder_name, str)
    assert isinstance(quantity, int)
    assert isinstance(speed_range, tuple)
    assert isinstance(pitch_range, tuple)
    assert os.path.exists(folder_name)
    assert quantity > 0
    filenames : List[ str ] = [
        filename for x in os.listdir(folder_name)
            for filename in (os.path.join(folder_name, x), )
                if os.path.splitext(filename)[1] == ".wav" and not filename.startswith("_")
    ]
    for filename in tqdm.tqdm(filenames, total = len(filenames), desc = "Augmenting data"):
        for aug_index in range(quantity):
            speed_factor : float = float(np.random.uniform(*speed_range))
            pitch_shift : float = float(np.random.uniform(*pitch_range))
            dst_filename : str = os.path.join(os.path.dirname(filename), os.path.splitext(os.path.basename(filename))[0] + f"_{aug_index}.wav")
            change_speed_and_pitch(filename, dst_filename, speed_factor, pitch_shift)
def augment_folder_audio_files(folder_name : str,
                                 quantity    : int,
                                 speed_range : Tuple[ int, int ],
                                 pitch_range : Tuple[ int, int ]) -> None:
    assert isinstance(folder_name, str)
    assert isinstance(quantity, int)
    assert isinstance(speed_range, tuple)
    assert isinstance(pitch_range, tuple)
    assert os.path.exists(folder_name)
    assert quantity > 0
    folder_names : List[ str ] = [
        _folder_name for x in os.listdir(folder_name)
            for _folder_name in (os.path.join(folder_name, x),)
                if os.path.isdir(_folder_name) and not _folder_name.startswith("_")
    ]
    for _folder_name in folder_names:
        augment_audio_files(_folder_name, quantity, speed_range, pitch_range)
if (__name__ == "__main__"):
    folder_name : str = "/content/data_split/train/"
    quantity : int = 5
    speed_range : tuple = (0.7, 1.3)
    pitch_range : tuple = (0.0, 2.0)
    augment_folder_audio_files(folder_name, quantity, speed_range, pitch_range)

from typing import Optional, List
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import librosa
import shutil
import tqdm
import os
import re
def save_mel_spectrogram(filename   : str,
                         wav_file   : str,
                         hop_length : Optional[ int ] = 512,
                         n_fft      : Optional[ int ] = 2048,
                         n_mels     : Optional[ int ] = 128) -> None:
    assert isinstance(filename, str)
    assert isinstance(wav_file, str)
    assert isinstance(hop_length, int)
    assert isinstance(n_fft, int)
    assert isinstance(n_mels, int)
    assert os.path.isfile(wav_file)
    assert hop_length > 0
    assert n_fft > 0
    assert n_mels > 0
    y, sr = librosa.load(wav_file, sr = None)
    mel_spectrogram = librosa.feature.melspectrogram(y = y, sr = sr, n_fft = n_fft, hop_length = hop_length, n_mels = n_mels)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref = np.max)
    plt.figure(figsize=(10, 10))
    librosa.display.specshow(mel_spectrogram_db, sr = sr, hop_length = hop_length)
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok = True)
    plt.savefig(filename)
    plt.clf()
def convert_to_melcepstrum(folder_name : str) -> None:
    assert isinstance(folder_name, str)
    assert os.path.exists(folder_name)
    def audio_files_in_folder(folder_name : str) -> List[ str ]:
        # print(folder_name)
        return [
            os.path.join(folder_name, x)
                for x in os.listdir(folder_name)
                    if os.path.splitext(x)[1] == ".wav"
        ]
    split_folders : List[ str ] = [
        os.path.join(folder_path, x)
            for folder_path in (folder_name,)
                for x in os.listdir(folder_path)
                    if not x.startswith("_")
    ]
    dst_folder : str = (folder_name if folder_name[-1] != "/" else folder_name[:-1]) + "_mel"
    if os.path.exists(dst_folder):
        shutil.rmtree(dst_folder)
    os.makedirs(dst_folder, exist_ok = True)
    for split_folder in split_folders:
        class_folders : List[ str ] = [
            os.path.join(split_folder, x)
                for x in os.listdir(split_folder)
                    if not x.startswith("_")
        ]
        split_name : str = list(filter("".__ne__, re.split(r"[/\\]+", split_folder)))[-1]
        os.makedirs(os.path.dirname(split_folder), exist_ok = True)
        for class_folder in class_folders:
            for audio_file in tqdm.tqdm(audio_files_in_folder(class_folder), total = len(class_folders), desc = "Converting to Melcepstrum"):
                basename : str = os.path.splitext(os.path.basename(audio_file))[0]
                image_name : str = "{}/{}/{}/{}.png".format(dst_folder, split_name, os.path.splitext(basename)[0], basename)
                save_mel_spectrogram(image_name, audio_file)
if (__name__ == "__main__"):
    folder_name : str = "/content/data_split/"
    convert_to_melcepstrum(folder_name)

!zip -r /content/drive/MyDrive/multimedia_hw4_mel_M022024.zip /content/data_split_mel/

import gc
gc.collect()

from typing import List
import numpy as np
import shutil
import tqdm
import os
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
    print(folder_name)
    assert os.path.exists(folder_name)
    return [
        filename for x in os.listdir(folder_name)
            for filename in (os.path.join(folder_name, x),)
                if os.path.isfile(filename) and not x.startswith("_")
    ]
def split_fold_data(src_folder : str,
                    dst_folder : str,
                    fold_ratio : float) -> None:
    assert isinstance(src_folder, str)
    assert isinstance(dst_folder, str)
    assert isinstance(fold_ratio, float)
    assert os.path.exists(src_folder)
    assert 0 < fold_ratio <= 1
    class_names : List[ str ] = find_class_names(src_folder)
    if os.path.exists(dst_folder):
        shutil.rmtree(dst_folder)
    class_folders : List[ str ] = find_class_folders(src_folder, class_names)
    for class_idx, class_name in enumerate(class_names):
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
    src_folder_name : str = "/content/data_split_mel/train/"
    dst_folder_name : str = "/content/data_split_mel_fold/"
    split_fold_data(src_folder_name, dst_folder_name, fold_ratio = 0.2)

