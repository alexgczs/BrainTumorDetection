from contextlib import nullcontext
from tkinter.messagebox import NO
import numpy as np
from vit import pred
from torchvision.models import VisionTransformer
import torchvision
import torch.nn as nn
import torch

class BrainTumorDetector():
    def __init__(self) -> None:
        super().__init__()
        self.cancer = None
        self.predictions = []
        self.model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
        self.model.heads = nn.Linear(768, 1)

    def random_eval(self, image):
        random = np.random.rand(1)
        print(random)
        self.cancer = random >= 0.5
        self.predictions.append(*random)
        print("pred", pred(image))

    def evaluate(self, imagen):
        input=np.array([imagen])
        input.resize((1,3,224,224))
        input = torch.from_numpy(input)
        input = input.type(torch.float32)
        output=self.model(input)
        print("out",output)
        _,pred=torch.max(output, 1)
        return pred

    def get_cancer(self):
        return self.cancer

    def get_probability(self) -> str:
        prob = np.mean(self.predictions) * 100 
        return "{:.2f}".format(prob)


if __name__ == '__main__':
    model = BrainTumorDetector()

    model.save_pretrained("PleaseWork", push_to_hub=True)
