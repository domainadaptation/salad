import os.path as osp

from torchvision import transforms, datasets

def load_dataset(path='./data', im_size = 224):
    T = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(.1, .8, .75, 0),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    print(osp.join(path, 'train'))
    train      = datasets.ImageFolder(osp.join(path, 'train'), transform=T)
    validation = datasets.ImageFolder(osp.join(path,'validation'), transform=T)

    return {'source' : train, 'target' : validation}