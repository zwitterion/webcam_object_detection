# Detects people in a webcam stream
# See also webcam.ipynb notebook on this repo
# and https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb#scrollTo=bX0rqK-A3Nbl
#


import torch
import torchvision
import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import cv2

from PIL import Image, ImageDraw, ImageTk
import PIL

from tkinter import *


model_checkpoint_file = "./pedestrian.pt"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 2

def get_instance_segmentation_model(num_classes):    
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def analyze_image(cv2image):
    
    img = None
    with torch.no_grad():
        img, _ = trx(cv2image, None)
        prediction = model([img.to(device)])
        if (len(prediction[0]['boxes']) > 0):
            new_img = PIL.Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
            draw = ImageDraw.Draw(new_img)
            for box in prediction[0]['boxes']:
                draw.rectangle((box[0], box[1], box[2], box[3]), outline=128, width=4)
            del draw
            img = new_img
        else:
            # nothing detected 
            img = PIL.Image.fromarray(cv2image)
    
        return img


# load base model 
model = get_instance_segmentation_model(num_classes)

checkpoint = torch.load(model_checkpoint_file)
model.load_state_dict(checkpoint['model_state_dict'])

epoch = checkpoint['epoch']
loss = checkpoint['loss']

print("Using model: {0}, epoch: {1}, loss: {2}".format(model_checkpoint_file, epoch, loss))

model.to(device)
model.eval()

trx = get_transform(False)

   
width, height = 800, 600
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

root = Tk()
root.bind('<Escape>', lambda e: root.quit())
lmain = Label(root)
lmain.pack()

def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # COLOR_BGR2RGBA

    img = analyze_image(cv2image)

    #img = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)

show_frame()
root.mainloop()