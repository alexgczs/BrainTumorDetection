from torchvision.models import vit_b_16, ViT_B_16_Weights
import torchvision.transforms as transforms
from PIL import Image
from torch import unsqueeze, max, from_numpy
from torch.nn import Linear
from numpy import array

class BrainTumorDetector:
    def __init__(self) -> None:
        self.__init_model()

    def __init_model(self):
        self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.model.heads = Linear( self.model.heads.head.in_features, 1)

    def __preprare_img(self, image):
        input = Image.open(image).convert("RGB")
        input = transforms.Resize((224, 224))(input)
        tensor = transforms.ToTensor()(input)
        tensor = unsqueeze(tensor, dim= 0)
        return tensor

    def evaluate(self, images):
        preds = from_numpy(array([[0.0]]))
        for image in images:
            tensor = self.__preprare_img(image)

            output = self.model(tensor)

            print("pred", output)
            # _,pred= max(output, 1)

            preds += output

        _, global_prediction = max(preds, 1)
        print(global_prediction)
        if global_prediction == 0: return "The pacient does not have cancer"
        elif global_prediction == 1: return "The pacient has cancer"
        return preds

