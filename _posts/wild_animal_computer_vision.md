---
title: An Algorithm for Wildlife Animals Identification
image: /assets/images/animals.png
author: Alan Gewerc
date: November 2018
categories:
  - Data Science
  - Finance
  - Machine Learning
  - Deep learning
layout: post
---


In notebook we will be working with the Oregon Wildlife [dataset](https://www.kaggle.com/virtualdvid/oregon-wildlife/kernels) created by [David Molina](https://www.kaggle.com/virtualdvid) with a google scrapper.It constains about 14.000 pictures of 19 different wildlife species such as Deers, Cougars, Grey Wolfs and so on. 

Wouldn't it be fun to have a app that tells you what animals are you observing in the nature when you are camping or hiking? This project may be a first step to create this app. 

We aim to generate a model that scans an image and identify what is the animal species on the screen. To achieve this goal we will make use of **Convolutional Neural Networks.** <br><br>

## How does a Convolutional Neural Network function ?  

[deep.ai](https://deepai.org/machine-learning-glossary-and-terms/convolutional-neural-network): CNNs process images as volumes, receiving a color image as a rectangular box where the width and height are measure by the number of pixels associated with each dimension, and the depth is three layers deep for each color (RGB). These layers are called channels. Within each pixel of the image, the intensity of the R, G, or B is expressed by a number. That number is part of three, stacked two-dimensional matrices that make up the image volume and form the initial data that is fed to into the convolutional network. The network then begins to filter the image by grouping squares of pixels together and looking for patterns, performing what is known as a convolution. This process of pattern analysis is the foundation of CNN functions.<br><br>


![CNN](https://miro.medium.com/max/2510/1*vkQ0hXDaQv57sALXAJquxA.jpeg)


### Libraries
- numpy (Linear Algebra)
- pandas (Data Manipulation and Analysis)
- glob (File Manipulation)
- os (File Manipulation)
- regex (text patterns)
- random (sampling data)
- PIL (image processing) 
- Sklearn (Evaluation Metrics)
- Pytorch (Deep Learning)


### The Road Ahead

We break the notebook into separate steps.  Feel free to use the links below to navigate the notebook.

* [Step 1](#step1): Import Libraries and Load the Dataset 
* [Step 2](#step2): Create a CNN to Classify Wild Animals (from Scratch)
* [Step 3](#step3): Create a CNN to Classify Wild Animals (using Transfer Learning) 
* [Step 4](#step3): Test the model and create an Algorithm

We first mount the folder on the google drive with the dataset. 


```python
from google.colab import drive
drive.mount('/content/gdrive')
```    

<a id='step1'></a>
## Step 1: Import Dataset and Libraries

Our fist step is to import all the libraries used in this project.


```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
import pandas as pd
import os
import re
import random 
from PIL import Image 
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd
from skimage import io
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchvision
from torch.utils.data import (Dataset, DataLoader)  # Gives easier dataset managment and creates mini batches
from torchvision.datasets import ImageFolder
import torchvision.models as models
```

## Visualize the data

First lets examine the files. 
- Which are the species?
- How many pictures do we have from each animal? 




```python
animals_list = os.listdir("/content/gdrive/My Drive/oregon_wildlife")
animals_file_list = []

for i in range(len(animals_list)):

  animals_file_list.append(os.listdir(str("/content/gdrive/My Drive/oregon_wildlife/" + animals_list[i])))  
  n = len(animals_file_list[i])
  print('There are', n , animals_list[i] , 'images.')
```

    There are 660 elk images.
    There are 696 bobcat images.
    There are 686 cougar images.
    There are 748 bald_eagle images.
    There are 717 canada_lynx images.
    There are 668 gray_fox images.
    There are 736 coyote images.
    There are 735 columbian_black-tailed_deer images.
    There are 718 black_bear images.
    There are 764 deer images.
    There are 577 mountain_beaver images.
    There are 728 virginia_opossum images.
    There are 726 sea_lions images.
    There are 701 nutria images.
    There are 759 red_fox images.
    There are 728 raccoon images.
    There are 656 raven images.
    There are 698 seals images.
    There are 730 gray_wolf images.
    There are 588 ringtail images.
    

Apparently we have a balanced dataset,  which means we have a similar 9but not equal) proportion of images of animals from each species. 
To have a more real notion let's visualize the animals.







```python
w=10
h=10
fig=plt.figure(figsize=(16, 16))
columns = 4
rows = 5

for i in range(1, len(animals_list)+1):
    img = mpimg.imread(str("/content/gdrive/My Drive/oregon_wildlife/"+ animals_list[i-1] + "/"+ animals_file_list[i-1][0]))
    compose = transforms.Compose([transforms.ToPILImage(),transforms.Resize((256,256))])
    img = compose(img)
    fig.add_subplot(rows, columns, i)
    plt.axis('off')
    plt.title(animals_list[i-1])
    plt.imshow(img)
plt.show()
```


![png](/assets/images//wild_animal_computer_vision_7_0.png)


Very beautiful animals right?

## Load Data

Now we will make use of Pytorch elements `transform`, `ImageFolder`, `DataLoader` to load the data. 

The next steps will be the following. 

1. Create a dataframe with the name of each file, the animal and the absolute path. 
2. Select files that will further be in the train, test and validation sets. 
3. Perform transformation in the data such as reshaping, croping and rotation that will allow the images that are from different sizes to be analyzed together. 
4. Load the datasets using the DataLoader function, that will transform the images in tensors that will be analysed by the CNN.


```python
dir = '/content/gdrive/My Drive/oregon_wildlife'
files = [f for f in glob(dir + "**/**", recursive=True)] # create a list will allabsolute path of all files
```


```python
df_animals = pd.DataFrame({"file_path":files}) # transform in a dataframe
df_animals['animal'] = df_animals['file_path'].str.extract('/oregon_wildlife/(.+)/') # extract the name of the animal
df_animals['file'] = df_animals['file_path'].str.extract('oregon_wildlife/.+/(.+)') # extrat the file name
df_animals = df_animals.dropna() # drop nas  
```

Now we split the data in train, test and validation (inside the dataframe). 


```python
animal_set = set(df_animals['animal'])
train_val_test_list = [0,1,2]
train_val_weights = [70,15,15]
df_animals['train_val_test'] = 'NA' 

for an in animal_set:
  n = sum(df_animals['animal'] == an) # count the number of animals
  train_val_test = random.choices(train_val_test_list, weights= train_val_weights,  k=n)
  df_animals.loc[df_animals['animal'] == an, 'train_val_test'] = train_val_test 
```

Now we will create the dictonary `transform`. it will be used to transform the train, test and validation datasets. 
We will apply different transformations on the train and test/validation datasets. Data augmentation is used in the trainning dataset to avoid overfitting, that means to avoid the a very good performance on the trainning set but a bed performane on the validation and testing datasets (bed generalization). The methods used were:
- Flipping the images horizontally 
- Random Cropping: Extract randomly a 224 × 224 pixels section from 256 × 256 pixels
- RandomRotation: Randomly rotate the image by 10 degrees.



```python
transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),    
}

```

We create an auxiliary function to make sure the data is correctly splited among train, test and validation.   


```python
def check_train(path):
    return (df_animals[df_animals['file_path'] == path].train_val_test == 0).bool

def check_valid(path):
    return (df_animals[df_animals['file_path'] == path].train_val_test == 1).bool

def check_test(path):
    return (df_animals[df_animals['file_path'] == path].train_val_test == 2).bool
```

#### Load the dataset


```python
# Reading Dataset
image_datasets = {
    'train' : ImageFolder(root= dir, transform=transform['train'], is_valid_file=check_train),
    'valid' : ImageFolder(root=dir, transform=transform['valid'], is_valid_file=check_valid),
    'test' : ImageFolder(root=dir, transform=transform['test'], is_valid_file=check_test)
}
```


```python
num_workers = 0
batch_size = 20

loaders_scratch = {
    'train' : DataLoader(image_datasets['train'], shuffle = True, batch_size = batch_size),
    'valid' : DataLoader(image_datasets['valid'], shuffle = True, batch_size = batch_size),
    'test' : DataLoader(image_datasets['test'], shuffle = True, batch_size = batch_size)    
}
```

#### USE GPU


```python
# check if CUDA is available
use_cuda = torch.cuda.is_available()
```

<a id='step2'></a>
## Step 2: Create a CNN to Classify Wild Animals (from Scratch)

We create a CNN that reveives tensors of `224 x 224 x 3` dimensions (that's how we prepared the dataset). 

Some of the elements of our network built from scratch. 
- Five convolutional layers. The last will return a tensor of `128 x 256, x 3` dimensions. Padding equals 1. 
- A relu function applied after every convolutional iteration.
- A pooling function applied after every convolutional iteration.
- Two dropout layers to avoid overffiting.
- Two fully connected layers.  




 that will transform `128 x 256 x 3`
- relu



```python
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer (sees 224 x 224 x 3 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # convolutional layer (sees 122 x 122 x 16 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # convolutional layer (sees 56 x 56 x 32 tensor)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # convolutional layer (sees 28 x 28 x 64 tensor)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        # convolutional layer (sees 14 x 14 x 128 tensor)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

        self.conv_bn1 = nn.BatchNorm2d(224,3)
        self.conv_bn2 = nn.BatchNorm2d(16)
        self.conv_bn3 = nn.BatchNorm2d(32)
        self.conv_bn4 = nn.BatchNorm2d(64)
        self.conv_bn5 = nn.BatchNorm2d(128)
        self.conv_bn6 = nn.BatchNorm2d(256)

        # linear layer (64 * 4 * 4 -> 133)
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        # linear layer (133 -> 133)
        self.fc2 = nn.Linear(512, 20)
        

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.conv_bn2(self.pool(F.relu(self.conv1(x))))
        x = self.conv_bn3(self.pool(F.relu(self.conv2(x))))
        x = self.conv_bn4(self.pool(F.relu(self.conv3(x))))
        x = self.conv_bn5(self.pool(F.relu(self.conv4(x))))
        x = self.conv_bn6(self.pool(F.relu(self.conv5(x))))
        # flatten image input
        x = x.view(-1, 256 * 7 * 7)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x

#-#-# You so NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net()
print(model_scratch)

# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()
```

    Net(
      (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv5): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (dropout): Dropout(p=0.25, inplace=False)
      (conv_bn1): BatchNorm2d(224, eps=3, momentum=0.1, affine=True, track_running_stats=True)
      (conv_bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv_bn3): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv_bn4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv_bn5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv_bn6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc1): Linear(in_features=12544, out_features=512, bias=True)
      (fc2): Linear(in_features=512, out_features=20, bias=True)
    )
    

We define an optmizer and a loss function. 

- **loss function**: Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1. Cross-entropy loss increases as the predicted probability diverges from the actual label. So predicting a probability of .012 when the actual observation label is 1 would be bad and result in a high loss value. A perfect model would have a log loss of 0. From the [ml-cheatsheet](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)

- **optimizer**: Stochastic gradient descent (often abbreviated SGD) is an iterative method for optimizing an objective function with suitable smoothness properties (e.g. differentiable or subdifferentiable). It can be regarded as a stochastic approximation of gradient descent optimization, since it replaces the actual gradient (calculated from the entire data set) by an estimate thereof (calculated from a randomly selected subset of the data). Especially in high-dimensional optimization problems this reduces the computational burden, achieving faster iterations in trade for a lower convergence rate. From [wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)


```python

# specify loss function
criterion_scratch = nn.CrossEntropyLoss()

# specify optimizer
optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=0.001, momentum=0.9)
```


```python
def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):

    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            ## record the average training loss, using something like
            train_loss = train_loss + (1 / (batch_idx + 1)) * (loss.data - train_loss)

            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
            valid_loss = valid_loss + (1 / (batch_idx + 1)) * (loss.data - valid_loss)

            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), '/content/gdrive/My Drive/model_scratch.pt')
            valid_loss_min = valid_loss
            
    # return trained model
    return model

# train the model
model_scratch = train(25, loaders_scratch, model_scratch, optimizer_scratch, criterion_scratch, use_cuda, 'model_scratch.pt')

# load the model that got the best validation accuracy
model_scratch.load_state_dict(torch.load('/content/gdrive/My Drive/model_scratch.pt'))
```

    /usr/local/lib/python3.6/dist-packages/PIL/Image.py:932: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
      "Palette images with Transparency expressed in bytes should be "
    /usr/local/lib/python3.6/dist-packages/PIL/TiffImagePlugin.py:788: UserWarning: Corrupt EXIF data.  Expecting to read 2 bytes but only got 0. 
      warnings.warn(str(msg))
    

    Epoch: 1 	Training Loss: 2.607074 	Validation Loss: 2.116710
    Validation loss decreased (inf --> 2.116710).  Saving model ...
    Epoch: 2 	Training Loss: 2.354062 	Validation Loss: 1.838744
    Validation loss decreased (2.116710 --> 1.838744).  Saving model ...
    Epoch: 3 	Training Loss: 2.214730 	Validation Loss: 1.714859
    Validation loss decreased (1.838744 --> 1.714859).  Saving model ...
    Epoch: 4 	Training Loss: 2.126942 	Validation Loss: 1.554157
    Validation loss decreased (1.714859 --> 1.554157).  Saving model ...
    Epoch: 5 	Training Loss: 2.070691 	Validation Loss: 1.541050
    Validation loss decreased (1.554157 --> 1.541050).  Saving model ...
    Epoch: 6 	Training Loss: 1.990526 	Validation Loss: 1.407331
    Validation loss decreased (1.541050 --> 1.407331).  Saving model ...
    Epoch: 7 	Training Loss: 1.929990 	Validation Loss: 1.316320
    Validation loss decreased (1.407331 --> 1.316320).  Saving model ...
    Epoch: 8 	Training Loss: 1.853464 	Validation Loss: 1.214190
    Validation loss decreased (1.316320 --> 1.214190).  Saving model ...
    Epoch: 9 	Training Loss: 1.826795 	Validation Loss: 1.251694
    Epoch: 10 	Training Loss: 1.780122 	Validation Loss: 1.091102
    Validation loss decreased (1.214190 --> 1.091102).  Saving model ...
    Epoch: 11 	Training Loss: 1.744049 	Validation Loss: 1.061980
    Validation loss decreased (1.091102 --> 1.061980).  Saving model ...
    Epoch: 12 	Training Loss: 1.683120 	Validation Loss: 1.007441
    Validation loss decreased (1.061980 --> 1.007441).  Saving model ...
    Epoch: 13 	Training Loss: 1.638642 	Validation Loss: 0.962486
    Validation loss decreased (1.007441 --> 0.962486).  Saving model ...
    Epoch: 14 	Training Loss: 1.586386 	Validation Loss: 0.895636
    Validation loss decreased (0.962486 --> 0.895636).  Saving model ...
    Epoch: 15 	Training Loss: 1.577459 	Validation Loss: 0.875274
    Validation loss decreased (0.895636 --> 0.875274).  Saving model ...
    Epoch: 16 	Training Loss: 1.528924 	Validation Loss: 0.851284
    Validation loss decreased (0.875274 --> 0.851284).  Saving model ...
    Epoch: 17 	Training Loss: 1.514573 	Validation Loss: 0.792618
    Validation loss decreased (0.851284 --> 0.792618).  Saving model ...
    Epoch: 18 	Training Loss: 1.487057 	Validation Loss: 0.735587
    Validation loss decreased (0.792618 --> 0.735587).  Saving model ...
    Epoch: 19 	Training Loss: 1.449167 	Validation Loss: 0.702604
    Validation loss decreased (0.735587 --> 0.702604).  Saving model ...
    Epoch: 20 	Training Loss: 1.424645 	Validation Loss: 0.663261
    Validation loss decreased (0.702604 --> 0.663261).  Saving model ...
    Epoch: 21 	Training Loss: 1.391276 	Validation Loss: 0.634866
    Validation loss decreased (0.663261 --> 0.634866).  Saving model ...
    Epoch: 22 	Training Loss: 1.370575 	Validation Loss: 0.608783
    Validation loss decreased (0.634866 --> 0.608783).  Saving model ...
    Epoch: 23 	Training Loss: 1.331621 	Validation Loss: 0.564250
    Validation loss decreased (0.608783 --> 0.564250).  Saving model ...
    Epoch: 24 	Training Loss: 1.318738 	Validation Loss: 0.537431
    Validation loss decreased (0.564250 --> 0.537431).  Saving model ...
    Epoch: 25 	Training Loss: 1.305690 	Validation Loss: 0.526589
    Validation loss decreased (0.537431 --> 0.526589).  Saving model ...
    


### (IMPLEMENTATION) Test the Model

Try out your model on the test dataset of dog images.  Use the code cell below to calculate and print the test loss and accuracy.  Ensure that your test accuracy is greater than 10%.


```python
def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    if torch.cuda.is_available():
      model.cuda()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

# call test function    
test(loaders_scratch, model_scratch, criterion_scratch, use_cuda)
```


    Test Loss: 0.526578    
    Test Accuracy: 83% (11732/14019)
    

<a id='step3'></a>
## Step 3: Create a CNN to Classify Wild Animals (using Transfer Learning)

We will now use transfer learning to create a CNN that can identify the animals from the images. 

From [neurohive](https://neurohive.io/en/popular-networks/vgg16/):
*VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. The model achieves 92.7% top-5 test accuracy in ImageNet, which is a dataset of over 14 million images belonging to 1000 classes. It was one of the famous model submitted to ILSVRC-2014. It makes the improvement over AlexNet by replacing large kernel-sized filters (11 and 5 in the first and second convolutional layer, respectively) with multiple 3×3 kernel-sized filters one after another. VGG16 was trained for weeks and was using NVIDIA Titan Black GPU’s.*











```python
## TODO: Specify data loaders
loaders_transfer = loaders_scratch
```


```python
import torchvision.models as models
import torch.nn as nn

model_transfer = models.vgg16(pretrained=True)

for param in model_transfer.features.parameters():
    param.requires_grad = False

n_inputs = model_transfer.classifier[6].in_features
last_layer = nn.Linear(n_inputs, 133)
model_transfer.classifier[6] = last_layer


# if GPU is available, move the model to GPU
if use_cuda:
    model_transfer.cuda()
print(model_transfer)
```

    Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /root/.cache/torch/checkpoints/vgg16-397923af.pth
    


    HBox(children=(FloatProgress(value=0.0, max=553433881.0), HTML(value='')))


    
    VGG(
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU(inplace=True)
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): ReLU(inplace=True)
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): ReLU(inplace=True)
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU(inplace=True)
        (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (13): ReLU(inplace=True)
        (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): ReLU(inplace=True)
        (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (18): ReLU(inplace=True)
        (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (20): ReLU(inplace=True)
        (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (22): ReLU(inplace=True)
        (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (25): ReLU(inplace=True)
        (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (27): ReLU(inplace=True)
        (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (29): ReLU(inplace=True)
        (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
      (classifier): Sequential(
        (0): Linear(in_features=25088, out_features=4096, bias=True)
        (1): ReLU(inplace=True)
        (2): Dropout(p=0.5, inplace=False)
        (3): Linear(in_features=4096, out_features=4096, bias=True)
        (4): ReLU(inplace=True)
        (5): Dropout(p=0.5, inplace=False)
        (6): Linear(in_features=4096, out_features=133, bias=True)
      )
    )
    


```python
criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.SGD(model_transfer.classifier.parameters(), lr=0.001)
```


```python
# train the model
model_transfer = train(25, loaders_transfer, model_transfer, optimizer_transfer, criterion_transfer, use_cuda, 'model_transfer.pt')
```

    /usr/local/lib/python3.6/dist-packages/PIL/Image.py:932: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
      "Palette images with Transparency expressed in bytes should be "
    /usr/local/lib/python3.6/dist-packages/PIL/TiffImagePlugin.py:788: UserWarning: Corrupt EXIF data.  Expecting to read 2 bytes but only got 0. 
      warnings.warn(str(msg))
    

    Epoch: 1 	Training Loss: 1.620557 	Validation Loss: 0.565497
    Validation loss decreased (inf --> 0.565497).  Saving model ...
    Epoch: 2 	Training Loss: 0.963556 	Validation Loss: 0.474406
    Validation loss decreased (0.565497 --> 0.474406).  Saving model ...
    Epoch: 3 	Training Loss: 0.880449 	Validation Loss: 0.440184
    Validation loss decreased (0.474406 --> 0.440184).  Saving model ...
    Epoch: 4 	Training Loss: 0.852375 	Validation Loss: 0.417925
    Validation loss decreased (0.440184 --> 0.417925).  Saving model ...
    Epoch: 5 	Training Loss: 0.840897 	Validation Loss: 0.406296
    Validation loss decreased (0.417925 --> 0.406296).  Saving model ...
    Epoch: 6 	Training Loss: 0.800332 	Validation Loss: 0.393406
    Validation loss decreased (0.406296 --> 0.393406).  Saving model ...
    Epoch: 7 	Training Loss: 0.789050 	Validation Loss: 0.383758
    Validation loss decreased (0.393406 --> 0.383758).  Saving model ...
    Epoch: 8 	Training Loss: 0.772373 	Validation Loss: 0.368324
    Validation loss decreased (0.383758 --> 0.368324).  Saving model ...
    Epoch: 9 	Training Loss: 0.766794 	Validation Loss: 0.349522
    Validation loss decreased (0.368324 --> 0.349522).  Saving model ...
    Epoch: 10 	Training Loss: 0.738718 	Validation Loss: 0.344614
    Validation loss decreased (0.349522 --> 0.344614).  Saving model ...
    Epoch: 11 	Training Loss: 0.738962 	Validation Loss: 0.337593
    Validation loss decreased (0.344614 --> 0.337593).  Saving model ...
    Epoch: 12 	Training Loss: 0.735061 	Validation Loss: 0.327498
    Validation loss decreased (0.337593 --> 0.327498).  Saving model ...
    Epoch: 13 	Training Loss: 0.712622 	Validation Loss: 0.326299
    Validation loss decreased (0.327498 --> 0.326299).  Saving model ...
    Epoch: 14 	Training Loss: 0.704457 	Validation Loss: 0.318128
    Validation loss decreased (0.326299 --> 0.318128).  Saving model ...
    Epoch: 15 	Training Loss: 0.682098 	Validation Loss: 0.308181
    Validation loss decreased (0.318128 --> 0.308181).  Saving model ...
    Epoch: 16 	Training Loss: 0.692422 	Validation Loss: 0.298152
    Validation loss decreased (0.308181 --> 0.298152).  Saving model ...
    Epoch: 17 	Training Loss: 0.680175 	Validation Loss: 0.293996
    Validation loss decreased (0.298152 --> 0.293996).  Saving model ...
    Epoch: 18 	Training Loss: 0.680033 	Validation Loss: 0.287208
    Validation loss decreased (0.293996 --> 0.287208).  Saving model ...
    Epoch: 19 	Training Loss: 0.680246 	Validation Loss: 0.282499
    Validation loss decreased (0.287208 --> 0.282499).  Saving model ...
    Epoch: 20 	Training Loss: 0.681399 	Validation Loss: 0.280968
    Validation loss decreased (0.282499 --> 0.280968).  Saving model ...
    Epoch: 21 	Training Loss: 0.672516 	Validation Loss: 0.273934
    Validation loss decreased (0.280968 --> 0.273934).  Saving model ...
    Epoch: 22 	Training Loss: 0.667417 	Validation Loss: 0.272073
    Validation loss decreased (0.273934 --> 0.272073).  Saving model ...
    Epoch: 23 	Training Loss: 0.649626 	Validation Loss: 0.264014
    Validation loss decreased (0.272073 --> 0.264014).  Saving model ...
    Epoch: 24 	Training Loss: 0.648144 	Validation Loss: 0.258418
    Validation loss decreased (0.264014 --> 0.258418).  Saving model ...
    Epoch: 25 	Training Loss: 0.645692 	Validation Loss: 0.253728
    Validation loss decreased (0.258418 --> 0.253728).  Saving model ...
    


```python
model_transfer.load_state_dict(torch.load('/content/gdrive/My Drive/model_scratch.pt'))
test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)
```
    

    Test Loss: 0.253715
    
    
    Test Accuracy: 92% (12924/14019)
    


```python
### TODO: Write a function that takes a path to an image as input
### and returns the animal that is predicted by the model.

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.autograd import Variable
import random
import re

# create a list with a class names
class_names = image_datasets['train'].classes
class_names = [re.sub("\d{3}.", "", item) for item in class_names]
class_names = [re.sub("_", " ", item) for item in class_names]

def predict_breed_transfer(img_path):
  
    # load the image and return the predicted breed    
    img = Image.open(img_path) # Load the image from provided path
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize]
    )
    
    img_tensor = preprocess(img).float()
    img_tensor.unsqueeze_(0)  # Insert the new axis at index 0 i.e. in front of the other axes/dims.
    img_tensor = Variable(img_tensor) #The input to the network needs to be an autograd Variable
    
    if use_cuda:
        img_tensor = Variable(img_tensor.cuda())        
    
    model_transfer.eval()
    output = model_transfer(img_tensor) # Returns a Tensor of shape (batch, num class labels)
    output = output.cpu()
    
    # Our prediction will be the index of the class label with the largest value.
    predict_index = output.data.numpy().argmax() 
    
    predicted_breed = class_names[predict_index]
    true_breed = image_datasets['train'].classes[predict_index]
    
    return (predicted_breed, true_breed)

# Create list of test image paths
test_img_paths = list(df_animals[df_animals.train_val_test == 2].file_path)
np.random.shuffle(test_img_paths)

for img_path in test_img_paths[0:20]:
    predicted_breed, true_breed = predict_breed_transfer(img_path)
    print("Predicted Animal:" , predicted_breed, "\n", "True Animal:" , true_breed)
    img=mpimg.imread(img_path)
    imgplot = plt.imshow(img)
    plt.show()

```

    Predicted Animal: sea lions 
     True Animal: sea_lions
    


![png](/assets/images//wild_animal_computer_vision_37_1.png)


    Predicted Animal: bald eagle 
     True Animal: bald_eagle
    


![png](/assets/images//wild_animal_computer_vision_37_3.png)


    Predicted Animal: columbian black-tailed deer 
     True Animal: columbian_black-tailed_deer
    


![png](/assets/images//wild_animal_computer_vision_37_5.png)


    Predicted Animal: cougar 
     True Animal: cougar
    


![png](/assets/images//wild_animal_computer_vision_37_7.png)


    Predicted Animal: coyote 
     True Animal: coyote
    


![png](/assets/images//wild_animal_computer_vision_37_9.png)


    Predicted Animal: raven 
     True Animal: raven
    


![png](/assets/images//wild_animal_computer_vision_37_11.png)


    Predicted Animal: mountain beaver 
     True Animal: mountain_beaver
    


![png](/assets/images//wild_animal_computer_vision_37_13.png)


    Predicted Animal: sea lions 
     True Animal: sea_lions
    


![png](/assets/images//wild_animal_computer_vision_37_15.png)


    Predicted Animal: sea lions 
     True Animal: sea_lions
    


![png](/assets/images//wild_animal_computer_vision_37_17.png)


    Predicted Animal: coyote 
     True Animal: coyote
    


![png](/assets/images//wild_animal_computer_vision_37_19.png)


    Predicted Animal: bald eagle 
     True Animal: bald_eagle
    


![png](/assets/images//wild_animal_computer_vision_37_21.png)


    Predicted Animal: coyote 
     True Animal: coyote
    


![png](/assets/images//wild_animal_computer_vision_37_23.png)


    Predicted Animal: columbian black-tailed deer 
     True Animal: columbian_black-tailed_deer
    


![png](/assets/images//wild_animal_computer_vision_37_25.png)


    Predicted Animal: seals 
     True Animal: seals
    


![png](/assets/images//wild_animal_computer_vision_37_27.png)


    Predicted Animal: ringtail 
     True Animal: ringtail
    


![png](/assets/images//wild_animal_computer_vision_37_29.png)


    Predicted Animal: gray wolf 
     True Animal: gray_wolf
    


![png](/assets/images//wild_animal_computer_vision_37_31.png)


    Predicted Animal: gray fox 
     True Animal: gray_fox
    


![png](/assets/images//wild_animal_computer_vision_37_33.png)


    Predicted Animal: mountain beaver 
     True Animal: mountain_beaver
    


![png](/assets/images//wild_animal_computer_vision_37_35.png)


    Predicted Animal: deer 
     True Animal: deer
    


![png](/assets/images//wild_animal_computer_vision_37_37.png)


    Predicted Animal: bobcat 
     True Animal: bobcat
    


![png](/assets/images//wild_animal_computer_vision_37_39.png)

