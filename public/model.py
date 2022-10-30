from contextlib import nullcontext
from tkinter.messagebox import NO
import numpy as np
from vit import pred

class BrainTumorDetector:
    def __init__(self) -> None:
        self.cancer = None
        self.predictions = []

    def evaluate(self, image):
        random = np.random.rand(1)
        print(random)
        self.cancer = random >= 0.5
        self.predictions.append(*random)
        print(pred(image))

    def get_cancer(self):
        return self.cancer

    def get_probability(self) -> str:
        prob = np.mean(self.predictions) * 100 
        return "{:.2f}".format(prob)

