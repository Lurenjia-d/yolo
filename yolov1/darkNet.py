import torch
import torch.nn as nn
def GetNet():
    model=torch.nn.modules.Sequential(
        nn.Conv2d(3,64,7,2,3),
        nn.LeakyReLU(0.1, inplace=True),
        nn.MaxPool2d(2,2),

        nn.Conv2d(64,192,3,1,1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.MaxPool2d(2,2),

        nn.Conv2d(192,128,1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(128,256,3,1,1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(256,256,1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(256,512,3,1,1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.MaxPool2d(2,2),

        nn.Conv2d(512, 256, 1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(256, 512, 3, 1, 1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(512, 256, 1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(256, 512, 3, 1, 1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(512, 256, 1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(256, 512, 3, 1, 1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(512, 256, 1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(256, 512, 3, 1, 1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(512, 512, 1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(512, 1024, 3, 1, 1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.MaxPool2d(2,2),

        nn.Conv2d(1024, 512, 1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(512, 1024, 3, 1, 1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(1024, 512, 1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(512, 1024, 3, 1, 1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(1024, 1024, 3, 1, 1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(1024, 1024, 3, 2, 1),
        nn.LeakyReLU(0.1, inplace=True),

        nn.Conv2d(1024, 1024, 3, 1, 1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(1024, 1024, 3, 1, 1),
        nn.LeakyReLU(0.1, inplace=True),

        nn.Flatten(),
        nn.Linear(7*7*1024,4096),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Linear(4096,7*7*30),
        nn.Sigmoid()
    )
    return model