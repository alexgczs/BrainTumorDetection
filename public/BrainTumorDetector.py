from torchvision.models import vit_b_16, ViT_B_16_Weights
import torchvision.transforms as T
from PIL import Image

class BrainTumorDetector():
    def __init__(self) -> None:
        self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

    def evaluate(self, images):
        for image in images:
            input = Image.open(image)
            input = T.Resize((224,224))(input)

            output = self.model(input)
            print("out", output)

    def evaluate_v0(self, imagen):
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