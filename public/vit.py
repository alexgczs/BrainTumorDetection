from torch import nn, optim
from torch.optim import lr_scheduler
import torch
import torchvision
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Load VIT (Maybe a big vit model)
model_transf = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)

for param in model_transf.parameters():
    param.requires_grad = False

#Number of classes: 2, positive or negative. Access to the last layer and change output size
model_transf.heads = nn.Linear(1000, 1)

#loss function
criterion = nn.CrossEntropyLoss()
#To be faster, change all params to only last layer param --->  model_transf.heads.head.in_features
optimizer= optim.SGD(model_transf.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def pred(imagen):
    input=np.array([imagen])
    input.resize((1,3,224,224))
    print(input.shape)
    output=model_transf(input)
    _,pred=torch.max(output, 1)
    print(pred)