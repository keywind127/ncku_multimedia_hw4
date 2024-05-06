# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount("/content/drive/")

from matplotlib import pyplot as plt
from typing import List, Dict
import numpy as np
import json

def plot_training_history(history : Dict[ str, Dict[ str, List[ Dict[ str, List[ float ] ] ] ] ], feature_type : str, should_limit : bool = True) -> None:

    history : List[ Dict[ str, List[ float ] ] ] = history[tuple(history.keys())[0]][feature_type]

    scale_factor : dict = {
        "Accuracy": 100,
        "loss": 1
    }

    for metric in [ "loss", "Accuracy" ]:

        legends : List[ str ] = []

        for fold_label, fold_history in enumerate(history, start = 1):
            plt.plot(np.array(fold_history[f"train_{metric}"]) * scale_factor[metric])
            plt.plot(np.array(fold_history[f"val_{metric}"]) * scale_factor[metric])
            legends.extend([ f"train_{fold_label}", f"valid_{fold_label}"])

        plt.title(f"Folds {metric.capitalize()} History")
        plt.xlabel("Epochs")
        plt.ylabel(metric.capitalize())
        plt.legend(legends, bbox_to_anchor = (1, 1))
        plt.tight_layout()
        if (scale_factor[metric] != 1) and (should_limit):
            plt.ylim(0, scale_factor[metric])
        plt.show()
        plt.clf()

if (__name__ == "__main__"):

    filename : str = input("> ")
    history : Dict[ str, Dict[ str, List[ Dict[ str, List[ float ] ] ] ] ] = json.load(open(filename, "r"))
    print(history)
    plot_training_history(history, "mfcc", False)

