from PIL import Image
import telepot
import os
import numpy as np
import torch
from torchvision import transforms
from  torch import nn
import matplotlib.image


API_TOKEN = '1212978916:AAErcJf4hW3mTAq9dkLF1ZwZ7ns_RXMrjzg'
SIZE = 256

def blocks(input_res = 256, num_features = 256):

    result = nn.Sequential(
        nn.Conv2d(input_res, num_features,(3,3),stride = (1,1), padding = (1,1)),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(input_res, num_features,(3,3),stride = (1,1), padding = (1,1)),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(input_res, num_features,(3,3),stride = (1,1), padding = (1,1)),
        nn.LeakyReLU(0.2, inplace=True),
        nn.InstanceNorm2d(input_res),
        )
    return result


class Generator(torch.nn.Module):
    def __init__(self, ):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super().__init__()
        channels = 3

        # Initial convolution block
        out_features = 64
        # encoder
        self.input = nn.Sequential(
            nn.ReflectionPad2d(channels),
            nn.Conv2d(3, out_features, (7, 7)),
            nn.InstanceNorm2d(out_features),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_features, out_features * 2, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(out_features * 2, out_features * 4, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.in0 = nn.InstanceNorm2d(256)
        self.block0 = blocks()
        self.block1 = blocks()
        self.block2 = blocks()
        self.block3 = blocks()
        self.block4 = blocks()
        self.block5 = blocks()
        self.block6 = blocks()
        self.block7 = blocks()

        self.out = nn.Sequential(

            nn.Upsample(scale_factor=2),
            nn.Conv2d(out_features * 4, out_features * 2, 3, stride=1, padding=1),
            nn.InstanceNorm2d(out_features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(out_features * 2, out_features, 3, stride=1, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ReflectionPad2d(channels),
            nn.Conv2d(out_features, channels, 7),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = self.input(x)
        x = self.in0(x)
        x = self.block0(x) + x
        x = self.block1(x) + x
        x = self.block2(x) + x
        x = self.block3(x) + x
        x = self.block4(x) + x
        x = self.in0(x)

        out = self.out(x)

        return out

def preper(photo):
    transform_test = transforms.Compose([
        transforms.Resize(SIZE),
        transforms.RandomCrop((SIZE, SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5, ],
            [0.5, 0.5, 0.5, ]),
    ])

    result = transform_test(photo)
    result = result.view(1, 3, SIZE, SIZE)

    return result

def jobs(order, count_last):
    for line in order[count_last:]:
        line = line.split()

        chat_id = line[0]

        idd = line[2]
        # bot.download_file(idd, 'test3.jpg')

        img = Image.open(idd)
        img_anime = model(preper(img)).detach()
        img_anime = np.rollaxis(img_anime[0].numpy() / 2 + 0.5, 0, 3)
        matplotlib.image.imsave('photo.jpg', img_anime)
        bot.sendPhoto(chat_id=chat_id, photo=open('photo.jpg', 'rb'))
        os.remove(idd)


model = Generator()
model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))

bot = telepot.Bot(API_TOKEN)

order = [1]
count = 0
with open(f'order_anime.txt', 'r') as f:
    order = f.readlines()
while count != len(order):
    count_last = count
    jobs(order, count_last)
    count = len(order)
    with open(f'order_anime.txt', 'r') as f:
        order = f.readlines()

os.remove('order_anime.txt')
