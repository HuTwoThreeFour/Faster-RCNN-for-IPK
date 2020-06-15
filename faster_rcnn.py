import re
import utils
import Dataset
import torch
import torch.utils.data
import torchvision
from engine import train_one_epoch, evaluate
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 13  # 12 class + background
background = 'with_paper'  # can choose from 'blank' , 'with_paper' , 'with_coin'
object = 'Mutter' # can choose from 'all' , 'Schraube' , 'Mutter'
# set epochs
num_epochs = 15 

def faster_rcnn(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

   # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

import transforms as T

def get_transform(train):
    transforms = []
    if train:
        # transforms.append(T.random_affine(degrees=1.98, translate=0.05, scale=0.05, shear=0.641))
        transforms.append(T.ColorJitter(brightness=0.5, saturation=0.5))
        transforms.append(T.RandomRotation())
        transforms.append(T.ToTensor())
        transforms.append(T.RandomHorizontalFlip(0.5))
    else:
        transforms.append(T.ToTensor())
    return T.Compose(transforms)


if __name__ == '__main__':
#def main():
    writer = SummaryWriter(comment=background+'_'+object+'_'+str(num_epochs))
    dataset = Dataset.Schraubenerkennung(root = 'Datensatz_512',
                                         background= background, which_object = object, transforms = get_transform(train=True))
    dataset_test = Dataset.Schraubenerkennung(root='Datensatz_512',
                                         background= background, which_object= object, transforms=get_transform(train=False))
    #split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    #set ratio between dataset_train and dataset_test
    indices_ratio = 0.1
    len_indices = len(indices)
    num_test = int(indices_ratio*len_indices)
    dataset = torch.utils.data.Subset(dataset, indices[:-num_test])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-num_test:])

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # device = torch.device('cpu')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    #choose the model
    model = faster_rcnn(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0002,
                                 weight_decay=0.0005)
                            # momentum=0.9,    # when SGD
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)


    titels = ['AP IOU=0.5:0.95', 'AP IOU=0.5', 'AP IOU=0.75',
             'AP IOU=0.5:0.95 small', 'AP IOU=0.5:0.95 mudium', 'AP IOU=0.5:0.95 large']

    for epoch in range(num_epochs):
        # For Training
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        results, maps = evaluate(model, data_loader_test, device=device)
        for titel, map in zip(titels, maps):
            # print('map is: ' ,map)
            writer.add_scalar(titel, map, epoch)
    writer.close()
    # torch.save(model.state_dict(), 'model')
    print("That's it!")
