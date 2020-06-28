import matplotlib.image
from PIL import Image
import telepot
import numpy as np
import os
import torchvision.transforms as transforms
from nst_model import *

API_TOKEN = '<>'
bot = telepot.Bot(API_TOKEN)

def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)      # функция для отрисовки изображения
    image = transforms.ToPILImage()(image)
    return image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 256
loader = transforms.Compose([
    transforms.Resize(imsize),  # нормируем размер изображения
    transforms.CenterCrop(imsize),
    transforms.ToTensor()])  # превращаем в удобный формат

def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


with open(f'order_nst.txt', 'r') as f:
    order = f.readlines()

def jobs(order, count_last):
    for line in order[count_last:]:
        line = line.split()

        chat_id = line[0]

        idd1 = line[2]
        idd2 = line[3]
        # bot.download_file(idd, 'test3.jpg')

        content_img = image_loader(idd1)
        style_img = image_loader(idd2)

        model = STYLE_TRANSFER_vgg16()
        output = model(content_img, style_img, content_layers_default, style_layers_default)
        output = np.rollaxis(output.detach()[0].numpy(), 0, 3)
        matplotlib.image.imsave('photo.jpg', output , )
        bot.sendPhoto(chat_id=chat_id, photo=open('photo.jpg', 'rb'))

        os.remove(idd1)
        os.remove(idd2)


bot = telepot.Bot(API_TOKEN)

order = [1]
count = 0
with open(f'order_nst.txt', 'r') as f:
    order = f.readlines()
while count != len(order):
    count_last = count
    jobs(order, count_last)
    count = len(order)
    with open(f'order_nst.txt', 'r') as f:
        order = f.readlines()
os.remove('order_nst.txt')






