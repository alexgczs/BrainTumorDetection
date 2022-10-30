from torchvision.models import vit_b_16, ViT_B_16_Weights
import torchvision.transforms as transforms
from PIL import Image
from torch import unsqueeze, max, from_numpy, load, round, sigmoid
from torch.nn import Linear
from numpy import array

class BrainTumorDetector:
    def __init__(self) -> None:
        self.model = load("model_1.pt")

    def __preprare_img(self, image):
        input = Image.open(image).convert("RGB")
        input = transforms.Resize((224, 224))(input)
        tensor = transforms.ToTensor()(input)
        tensor = unsqueeze(tensor, dim= 0)
        return tensor

    def evaluate(self, images):
        preds = from_numpy(array([[0.0]]))
        num_preds = len(images)

        for image in images:
            tensor = self.__preprare_img(image)

            output = self.model(tensor)
            print("out", output)
            pred = round(sigmoid(output))
            print("pred", pred)

            preds += pred

        global_prediction = preds / num_preds
        print(global_prediction)
        if global_prediction >= 0.5: return "The pacient does not have cancer"
        else: return "The pacient has cancer"
