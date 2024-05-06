from collections import defaultdict
from typing import Dict, List
import numpy as np
import json
def calc_average_scores(history : dict) -> Dict[ str, float ]:
    calculated_scores : Dict[ str, List[ float ] ] = defaultdict(list)
    for __history in history:
        for metric, scores in __history.items():
            if ("loss" in metric):
                calculated_scores[metric].append(np.min(scores))
            else:
                calculated_scores[metric].append(np.max(scores))
    for metric in tuple(calculated_scores.keys()):
        calculated_scores[metric] = np.mean(calculated_scores[metric])
    return calculated_scores
if (__name__ == "__main__"):
    filename : str = input("> ")
    history_obj : dict = json.load(open(filename))
    print(calc_average_scores(history_obj["cnn"]["mfcc"]))
    print(calc_average_scores(history_obj["cnn"]["melc"]))