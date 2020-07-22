---
title: An Algorithm for a Dog Identification App 
date: November 2018
layout: post
image: /assets/images/anthony-tori-102062.jpg
author: Alan Gewerc
categories:
    - work
    - projects

---




In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'(IMPLEMENTATION)'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section, and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully! 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the Jupyter Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.

>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut.  Markdown cells can be edited by double-clicking the cell to enter edit mode.

The rubric contains _optional_ "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. If you decide to pursue the "Stand Out Suggestions", you should include the code in this Jupyter notebook.




### Why We're Here 

In this notebook, you will make the first steps towards developing an algorithm that could be used as part of a mobile or web app.  At the end of this project, your code will accept any user-supplied image as input.  If a dog is detected in the image, it will provide an estimate of the dog's breed.  If a human is detected, it will provide an estimate of the dog breed that is most resembling.  The image below displays potential sample output of your finished project (... but we expect that each student's algorithm will behave differently!). 

![Sample Dog Output](images/sample_dog_output.png)

In this real-world setting, you will need to piece together a series of models to perform different tasks; for instance, the algorithm that detects humans in an image will be different from the CNN that infers dog breed.  There are many points of possible failure, and no perfect algorithm exists.  Your imperfect solution will nonetheless create a fun user experience!

### The Road Ahead

We break the notebook into separate steps.  Feel free to use the links below to navigate the notebook.

* [Step 0](#step0): Import Datasets
* [Step 1](#step1): Detect Humans
* [Step 2](#step2): Detect Dogs
* [Step 3](#step3): Create a CNN to Classify Dog Breeds (from Scratch)
* [Step 4](#step4): Create a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 5](#step5): Write your Algorithm
* [Step 6](#step6): Test Your Algorithm

<a id='step0'></a>
## Step 0: Import Datasets

Make sure that you've downloaded the required human and dog datasets:

**Note: if you are using the Udacity workspace, you *DO NOT* need to re-download these - they can be found in the `/data` folder as noted in the cell below.**

* Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in this project's home directory, at the location `/dog_images`. 

* Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the home directory, at location `/lfw`.  

*Note: If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder.*

In the code cell below, we save the file paths for both the human (LFW) dataset and dog dataset in the numpy arrays `human_files` and `dog_files`.


```python
import numpy as np
from glob import glob

# load filenames for human and dog images
human_files = np.array(glob("/data/lfw/*/*"))
dog_files = np.array(glob("/data/dog_images/*/*/*"))

# print number of images in each dataset
print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))
```

    There are 13233 total human images.
    There are 8351 total dog images.
    

<a id='step1'></a>
## Step 1: Detect Humans

In this section, we use OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images.  

OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades).  We have downloaded one of these detectors and stored it in the `haarcascades` directory.  In the next code cell, we demonstrate how to use this detector to find human faces in a sample image.


```python
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline                               

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[0])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()
```

    Number of faces detected: 1
    


![png](dog_files/dog_3_1.png)


Before using any of the face detectors, it is standard procedure to convert the images to grayscale.  The `detectMultiScale` function executes the classifier stored in `face_cascade` and takes the grayscale image as a parameter.  

In the above code, `faces` is a numpy array of detected faces, where each row corresponds to a detected face.  Each detected face is a 1D array with four entries that specifies the bounding box of the detected face.  The first two entries in the array (extracted in the above code as `x` and `y`) specify the horizontal and vertical positions of the top left corner of the bounding box.  The last two entries in the array (extracted here as `w` and `h`) specify the width and height of the box.

### Write a Human Face Detector

We can use this procedure to write a function that returns `True` if a human face is detected in an image and `False` otherwise.  This function, aptly named `face_detector`, takes a string-valued file path to an image as input and appears in the code block below.


```python
# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
```

### (IMPLEMENTATION) Assess the Human Face Detector

__Question 1:__ Use the code cell below to test the performance of the `face_detector` function.  
- What percentage of the first 100 images in `human_files` have a detected human face?  
- What percentage of the first 100 images in `dog_files` have a detected human face? 

Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face.  You will see that our algorithm falls short of this goal, but still gives acceptable performance.  We extract the file paths for the first 100 images from each of the datasets and store them in the numpy arrays `human_files_short` and `dog_files_short`.

__Answer:__ 
(You can print out your results and/or write your percentages in this cell)


```python
from tqdm import tqdm

human_files_short = human_files[:100]
dog_files_short = dog_files[:100]

#-#-# Do NOT modify the code above this line. #-#-#

## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.

n_range = range(100)

countHuman = sum([face_detector(human_files_short[i]) for i in n_range])
countDog = sum([face_detector(dog_files_short[i]) for i in n_range])

print("Predicted Human Faces on Human dataset:", countHuman)
print("Accuracy:", countHuman/100, "\n")

print("Predicted Human Faces on Dog dataset:", countDog)
print("Accuracy:", 1 - countDog/100)
```

    Predicted Human Faces on Human dataset: 98
    Accuracy: 0.98 
    
    Predicted Human Faces on Dog dataset: 17
    Accuracy: 0.83
    

We suggest the face detector from OpenCV as a potential way to detect human images in your algorithm, but you are free to explore other approaches, especially approaches that make use of deep learning :).  Please use the code cell below to design and test your own face detection algorithm.  If you decide to pursue this _optional_ task, report performance on `human_files_short` and `dog_files_short`.


```python
!pip install --upgrade pip
!pip3 install face_recognition
!pip3 install install cmake
```

    Requirement already up-to-date: pip in /opt/conda/lib/python3.6/site-packages (19.3.1)
    Requirement already satisfied: face_recognition in /opt/conda/lib/python3.6/site-packages (1.2.3)
    Requirement already satisfied: dlib>=19.7 in /opt/conda/lib/python3.6/site-packages (from face_recognition) (19.19.0)
    Requirement already satisfied: numpy in /opt/conda/lib/python3.6/site-packages (from face_recognition) (1.12.1)
    Requirement already satisfied: Pillow in /opt/conda/lib/python3.6/site-packages (from face_recognition) (5.2.0)
    Requirement already satisfied: Click>=6.0 in /opt/conda/lib/python3.6/site-packages (from face_recognition) (6.7)
    Requirement already satisfied: face-recognition-models>=0.3.0 in /opt/conda/lib/python3.6/site-packages (from face_recognition) (0.3.0)
    [31mERROR: Could not find a version that satisfies the requirement install (from versions: none)[0m
    [31mERROR: No matching distribution found for install[0m
    


```python
### (Optional) 
### TODO: Test performance of anotherface detection algorithm.
### Feel free to use as many code cells as needed.

import face_recognition

def face_recog(img_path):
    image = face_recognition.load_image_file(img_path) 
    face_locations = face_recognition.face_locations(image)
    return len(face_locations) > 0
```


```python
countHuman = sum([face_recog(human_files_short[i]) for i in n_range])
countDog = sum([face_recog(dog_files_short[i]) for i in n_range])

print("Predicted Human Faces on Human dataset:", countHuman)
print("Accuracy:", countHuman/100, "\n")

print("Predicted Human Faces on Dog dataset:", countDog)
print("Accuracy:", 1 - countDog/100)
```

    Predicted Human Faces on Human dataset: 100
    Accuracy: 1.0 
    
    Predicted Human Faces on Dog dataset: 8
    Accuracy: 0.92
    

---
<a id='step2'></a>
## Step 2: Detect Dogs

In this section, we use a [pre-trained model](http://pytorch.org/docs/master/torchvision/models.html) to detect dogs in images.  

### Obtain Pre-trained VGG-16 Model

The code cell below downloads the VGG-16 model, along with weights that have been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.  ImageNet contains over 10 million URLs, each linking to an image containing an object from one of [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).  


```python
import torch
import torchvision.models as models

# define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()
```

Given an image, this pre-trained VGG-16 model returns a prediction (derived from the 1000 possible categories in ImageNet) for the object that is contained in the image.

### (IMPLEMENTATION) Making Predictions with a Pre-trained Model

In the next code cell, you will write a function that accepts a path to an image (such as `'dogImages/train/001.Affenpinscher/Affenpinscher_00001.jpg'`) as input and returns the index corresponding to the ImageNet class that is predicted by the pre-trained VGG-16 model.  The output should always be an integer between 0 and 999, inclusive.

Before writing the function, make sure that you take the time to learn  how to appropriately pre-process tensors for pre-trained models in the [PyTorch documentation](http://pytorch.org/docs/stable/torchvision/models.html).


```python
from PIL import Image, ImageFile
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms


def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    image = Image.open(img_path).convert('RGB')
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )

    preprocess = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        normalize]
    )

    img_tensor = preprocess(image)
    img_tensor.unsqueeze_(0)

    # Load the pretrained model from pytorch   
    vgg16 = models.vgg16(pretrained=True)
    
    # get sample outputs
    output = vgg16(img_tensor)
    _, pred = torch.max(output, 1)    

    return pred.numpy()[0] # predicted class index
```

### (IMPLEMENTATION) Write a Dog Detector

While looking at the [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), you will notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all categories from `'Chihuahua'` to `'Mexican hairless'`.  Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained VGG-16 model, we need only check if the pre-trained model predicts an index between 151 and 268 (inclusive).

Use these ideas to complete the `dog_detector` function below, which returns `True` if a dog is detected in an image (and `False` if not).


```python
### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):

    ## TODO: Complete the function.
    output = VGG16_predict(img_path)
    
    return output >= 151 and output < 269  # true/false
```

### (IMPLEMENTATION) Assess the Dog Detector

__Question 2:__ Use the code cell below to test the performance of your `dog_detector` function.  
- What percentage of the images in `human_files_short` have a detected dog?  
- What percentage of the images in `dog_files_short` have a detected dog?

__Answer:__ 



```python
### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.

countHuman = sum([dog_detector(human_files_short[i]) for i in n_range])
countDog = sum([dog_detector(dog_files_short[i]) for i in n_range])

print("Predicted Human Faces on Human dataset:", countHuman)
print("Accuracy:", 1- countHuman/100, "\n")

print("Predicted Human Faces on Dog dataset:", countDog)
print("Accuracy:", countDog/100)
```

    Predicted Human Faces on Human dataset: 2
    Accuracy: 0.98 
    
    Predicted Human Faces on Dog dataset: 100
    Accuracy: 1.0
    

We suggest VGG-16 as a potential network to detect dog images in your algorithm, but you are free to explore other pre-trained networks (such as [Inception-v3](http://pytorch.org/docs/master/torchvision/models.html#inception-v3), [ResNet-50](http://pytorch.org/docs/master/torchvision/models.html#id3), etc).  Please use the code cell below to test other pre-trained PyTorch models.  If you decide to pursue this _optional_ task, report performance on `human_files_short` and `dog_files_short`.


```python
### (Optional) 
### TODO: Report the performance of another pre-trained network.
### Feel free to use as many code cells as needed.

def res_predict(img_path):

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    image = Image.open(img_path).convert('RGB')
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    preprocess = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        normalize])

    img_tensor = preprocess(image)
    img_tensor.unsqueeze_(0)

    # Load the pretrained model from pytorch   
    resnet50 = models.resnet50(pretrained=True)
    
    # get sample outputs
    output = resnet50(img_tensor)
    _, pred = torch.max(output, 1)    

    return pred.numpy()[0] # predicted class index

```


```python
### returns "True" if a dog is detected in the image stored at img_path
def resnetdog_detector(img_path):

    ## TODO: Complete the function.
    output = VGG16_predict(img_path)
    
    return output >= 151 and output < 269  # true/false
```


```python
### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.

countHuman = sum([resnetdog_detector(human_files_short[i]) for i in n_range])
countDog = sum([resnetdog_detector(dog_files_short[i]) for i in n_range])

print("Predicted Human Faces on Human dataset:", countHuman)
print("Accuracy:", 1- countHuman/100, "\n")

print("Predicted Human Faces on Dog dataset:", countDog)
print("Accuracy:", countDog/100)
```

    Predicted Human Faces on Human dataset: 1
    Accuracy: 0.99 
    
    Predicted Human Faces on Dog dataset: 100
    Accuracy: 1.0
    

---
<a id='step3'></a>
## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)

Now that we have functions for detecting humans and dogs in images, we need a way to predict breed from images.  In this step, you will create a CNN that classifies dog breeds.  You must create your CNN _from scratch_ (so, you can't use transfer learning _yet_!), and you must attain a test accuracy of at least 10%.  In Step 4 of this notebook, you will have the opportunity to use transfer learning to create a CNN that attains greatly improved accuracy.

We mention that the task of assigning breed to dogs from images is considered exceptionally challenging.  To see why, consider that *even a human* would have trouble distinguishing between a Brittany and a Welsh Springer Spaniel.  

Brittany | Welsh Springer Spaniel
- | - 
<img src="images/Brittany_02625.jpg" width="100"> | <img src="images/Welsh_springer_spaniel_08203.jpg" width="200">

It is not difficult to find other dog breed pairs with minimal inter-class variation (for instance, Curly-Coated Retrievers and American Water Spaniels).  

Curly-Coated Retriever | American Water Spaniel
- | -
<img src="images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="images/American_water_spaniel_00648.jpg" width="200">


Likewise, recall that labradors come in yellow, chocolate, and black.  Your vision-based algorithm will have to conquer this high intra-class variation to determine how to classify all of these different shades as the same breed.  

Yellow Labrador | Chocolate Labrador | Black Labrador
- | -
<img src="images/Labrador_retriever_06457.jpg" width="150"> | <img src="images/Labrador_retriever_06455.jpg" width="240"> | <img src="images/Labrador_retriever_06449.jpg" width="220">

We also mention that random chance presents an exceptionally low bar: setting aside the fact that the classes are slightly imabalanced, a random guess will provide a correct answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.  

Remember that the practice is far ahead of the theory in deep learning.  Experiment with many different architectures, and trust your intuition.  And, of course, have fun!

### (IMPLEMENTATION) Specify Data Loaders for the Dog Dataset

Use the code cell below to write three separate [data loaders](http://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for the training, validation, and test datasets of dog images (located at `dog_images/train`, `dog_images/valid`, and `dog_images/test`, respectively).  You may find [this documentation on custom datasets](http://pytorch.org/docs/stable/torchvision/datasets.html) to be a useful resource.  If you are interested in augmenting your training and/or validation data, check out the wide variety of [transforms](http://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transform)!


```python
from PIL import Image, ImageFile
import os
from torchvision import datasets
from torchvision.datasets import ImageFolder 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import PIL

### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes

ImageFile.LOAD_TRUNCATED_IMAGES = True

# number of subprocesses and samples per batch
num_workers = 0
batch_size = 20

# convert data to a normalized torch.FloatTensor
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

data_dir = '/data/dog_images'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# Reading Dataset
image_datasets = {
    'train' : ImageFolder(root=train_dir,transform=transform['train']),
    'valid' : ImageFolder(root=valid_dir,transform=transform['valid']),
    'test' : ImageFolder(root=test_dir,transform=transform['test'])
}

loaders_scratch = {
    'train' : DataLoader(image_datasets['train'], shuffle = True, batch_size = batch_size),
    'valid' : DataLoader(image_datasets['valid'], shuffle = True, batch_size = batch_size),
    'test' : DataLoader(image_datasets['test'], shuffle = True, batch_size = batch_size)    
}
```

**Question 3:** Describe your chosen procedure for preprocessing the data. 
- How does your code resize the images (by cropping, stretching, etc)?  What size did you pick for the input tensor, and why?
- Did you decide to augment the dataset?  If so, how (through translations, flips, rotations, etc)?  If not, why not?


**Answer**:

The code resizes images by cropping, while on the trainning data it was done randomly using `transforms.RandomResizedCrop`, in the validation and testing dataset it was done with `transforms.CenterCrop` and also `transforms.Resize`.  
The input tensor receives inputs of size (3 x 224 x 224), and this size was chosen after constant trial and error. In addition, the following text was extrected from the [Pytorch Documentation](https://pytorch.org/docs/stable/torchvision/models.html):<br>

**All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].** <br>

Data augmentation was used to avoid overfitting, that means to avoid the a very good performance on the trainning set but a bed performane on the validation and testing datasets (bed generalization). The methods used were:
- Flipping the images horizontally 
- Random Cropping: Extract randomly a 224 × 224 pixels section from 256 × 256 pixels
- RandomRotation: Randomly rotate the image by 10 degrees.

### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  Use the template in the code cell below.


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
        self.fc2 = nn.Linear(512, 133)
        

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
      (dropout): Dropout(p=0.25)
      (conv_bn1): BatchNorm2d(224, eps=3, momentum=0.1, affine=True, track_running_stats=True)
      (conv_bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv_bn3): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv_bn4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv_bn5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv_bn6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc1): Linear(in_features=12544, out_features=512, bias=True)
      (fc2): Linear(in_features=512, out_features=133, bias=True)
    )
    

__Question 4:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  

__Answer:__ 
- The CNN has five convolutional layers. 
- The first layer has 16 filters, doubling every layer until 256 filters on the last layer. 
- stride = 1 and padding = 1 in every layer.
- There are Relu activations after each layers.
- Each followed by a max pooling layer of size 2.
- There are two connected linear layers at the end are used, the first with 512 outputs, the second 113 (the number of classes).
- Dropout of 0.25
- Batch normalization was applied after each max pooling.

### (IMPLEMENTATION) Specify Loss Function and Optimizer

Use the next code cell to specify a [loss function](http://pytorch.org/docs/stable/nn.html#loss-functions) and [optimizer](http://pytorch.org/docs/stable/optim.html).  Save the chosen loss function as `criterion_scratch`, and the optimizer as `optimizer_scratch` below.


```python
import torch.optim as optim

# specify loss function
criterion_scratch = nn.CrossEntropyLoss()

# specify optimizer
optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=0.001, momentum=0.9)
```

### (IMPLEMENTATION) Train and Validate the Model

Train and validate your model in the code cell below.  [Save the final model parameters](http://pytorch.org/docs/master/notes/serialization.html) at filepath `'model_scratch.pt'`.


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
            torch.save(model.state_dict(), 'model_scratch.pt')
            valid_loss_min = valid_loss
            
    # return trained model
    return model

# train the model
model_scratch = train(20, loaders_scratch, model_scratch, optimizer_scratch, criterion_scratch, use_cuda, 'model_scratch.pt')

# load the model that got the best validation accuracy
model_scratch.load_state_dict(torch.load('model_scratch.pt'))
```

    Epoch: 1 	Training Loss: 4.769570 	Validation Loss: 4.535692
    Validation loss decreased (inf --> 4.535692).  Saving model ...
    Epoch: 2 	Training Loss: 4.510987 	Validation Loss: 4.340068
    Validation loss decreased (4.535692 --> 4.340068).  Saving model ...
    Epoch: 3 	Training Loss: 4.357699 	Validation Loss: 4.134140
    Validation loss decreased (4.340068 --> 4.134140).  Saving model ...
    Epoch: 4 	Training Loss: 4.231001 	Validation Loss: 4.016376
    Validation loss decreased (4.134140 --> 4.016376).  Saving model ...
    Epoch: 5 	Training Loss: 4.121954 	Validation Loss: 3.898394
    Validation loss decreased (4.016376 --> 3.898394).  Saving model ...
    Epoch: 6 	Training Loss: 4.052873 	Validation Loss: 3.803878
    Validation loss decreased (3.898394 --> 3.803878).  Saving model ...
    Epoch: 7 	Training Loss: 3.964024 	Validation Loss: 3.760672
    Validation loss decreased (3.803878 --> 3.760672).  Saving model ...
    Epoch: 8 	Training Loss: 3.898220 	Validation Loss: 3.681581
    Validation loss decreased (3.760672 --> 3.681581).  Saving model ...
    Epoch: 9 	Training Loss: 3.855025 	Validation Loss: 3.620196
    Validation loss decreased (3.681581 --> 3.620196).  Saving model ...
    Epoch: 10 	Training Loss: 3.787188 	Validation Loss: 3.721846
    Epoch: 11 	Training Loss: 3.722067 	Validation Loss: 3.609185
    Validation loss decreased (3.620196 --> 3.609185).  Saving model ...
    Epoch: 12 	Training Loss: 3.658974 	Validation Loss: 3.675768
    Epoch: 13 	Training Loss: 3.589201 	Validation Loss: 3.739745
    Epoch: 14 	Training Loss: 3.569790 	Validation Loss: 3.393636
    Validation loss decreased (3.609185 --> 3.393636).  Saving model ...
    Epoch: 15 	Training Loss: 3.509392 	Validation Loss: 3.373507
    Validation loss decreased (3.393636 --> 3.373507).  Saving model ...
    Epoch: 16 	Training Loss: 3.468454 	Validation Loss: 3.300366
    Validation loss decreased (3.373507 --> 3.300366).  Saving model ...
    Epoch: 17 	Training Loss: 3.384653 	Validation Loss: 3.246083
    Validation loss decreased (3.300366 --> 3.246083).  Saving model ...
    Epoch: 18 	Training Loss: 3.383477 	Validation Loss: 3.141809
    Validation loss decreased (3.246083 --> 3.141809).  Saving model ...
    Epoch: 19 	Training Loss: 3.303718 	Validation Loss: 3.371231
    Epoch: 20 	Training Loss: 3.280762 	Validation Loss: 3.131421
    Validation loss decreased (3.141809 --> 3.131421).  Saving model ...
    

### (IMPLEMENTATION) Test the Model

Try out your model on the test dataset of dog images.  Use the code cell below to calculate and print the test loss and accuracy.  Ensure that your test accuracy is greater than 10%.


```python
def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
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

    Test Loss: 3.095411
    
    
    Test Accuracy: 22% (190/836)
    

---
<a id='step4'></a>
## Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning)

You will now use transfer learning to create a CNN that can identify dog breed from images.  Your CNN must attain at least 60% accuracy on the test set.

### (IMPLEMENTATION) Specify Data Loaders for the Dog Dataset

Use the code cell below to write three separate [data loaders](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader) for the training, validation, and test datasets of dog images (located at `dogImages/train`, `dogImages/valid`, and `dogImages/test`, respectively). 

If you like, **you are welcome to use the same data loaders from the previous step**, when you created a CNN from scratch.


```python
## TODO: Specify data loaders
loaders_transfer = loaders_scratch
```

### (IMPLEMENTATION) Model Architecture

Use transfer learning to create a CNN to classify dog breed.  Use the code cell below, and save your initialized model as the variable `model_transfer`.


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

    VGG(
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace)
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU(inplace)
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): ReLU(inplace)
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): ReLU(inplace)
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU(inplace)
        (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (13): ReLU(inplace)
        (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): ReLU(inplace)
        (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (18): ReLU(inplace)
        (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (20): ReLU(inplace)
        (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (22): ReLU(inplace)
        (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (25): ReLU(inplace)
        (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (27): ReLU(inplace)
        (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (29): ReLU(inplace)
        (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (classifier): Sequential(
        (0): Linear(in_features=25088, out_features=4096, bias=True)
        (1): ReLU(inplace)
        (2): Dropout(p=0.5)
        (3): Linear(in_features=4096, out_features=4096, bias=True)
        (4): ReLU(inplace)
        (5): Dropout(p=0.5)
        (6): Linear(in_features=4096, out_features=133, bias=True)
      )
    )
    

__Question 5:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  Describe why you think the architecture is suitable for the current problem.

__Answer:__ 


In this project we will use the **End of ConvNet approach** for transfer learning. Since we have a small dataset to train our model, a model built from scratch would most likely perform poorly. The data used to train the VGG.16 is quite similar to our trainning data, added to the fact that our trainning data set is small makes it suited to this approach. The steps are the following:

1. Add one fully connected layer replacing the last layer of the pre-trained model. This new fully connected layer will have 133 categories as output, differently from the previous deleted layer. 
2. Randomize the weights of the new fully connected layer
3. Freeze all the weights from the pre-trained network
4. Train the network to update the weights of the new fully connected layer

After retrainning the model, the accuracy is ~ 85%, a satisfatory performance according to the asked threshold.

### (IMPLEMENTATION) Specify Loss Function and Optimizer

Use the next code cell to specify a [loss function](http://pytorch.org/docs/master/nn.html#loss-functions) and [optimizer](http://pytorch.org/docs/master/optim.html).  Save the chosen loss function as `criterion_transfer`, and the optimizer as `optimizer_transfer` below.


```python
criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.SGD(model_transfer.classifier.parameters(), lr=0.001)
```

### (IMPLEMENTATION) Train and Validate the Model

Train and validate your model in the code cell below.  [Save the final model parameters](http://pytorch.org/docs/master/notes/serialization.html) at filepath `'model_transfer.pt'`.


```python
# train the model
model_transfer = train(25, loaders_transfer, model_transfer, optimizer_transfer, criterion_transfer, use_cuda, 'model_transfer.pt')

# load the model that got the best validation accuracy (uncomment the line below)
# model_transfer.load_state_dict(torch.load('model_transfer.pt'))
```

    Epoch: 1 	Training Loss: 4.274225 	Validation Loss: 2.911767
    Validation loss decreased (inf --> 2.911767).  Saving model ...
    Epoch: 2 	Training Loss: 2.786192 	Validation Loss: 1.347715
    Validation loss decreased (2.911767 --> 1.347715).  Saving model ...
    Epoch: 3 	Training Loss: 1.949635 	Validation Loss: 0.859160
    Validation loss decreased (1.347715 --> 0.859160).  Saving model ...
    Epoch: 4 	Training Loss: 1.623275 	Validation Loss: 0.684027
    Validation loss decreased (0.859160 --> 0.684027).  Saving model ...
    Epoch: 5 	Training Loss: 1.468086 	Validation Loss: 0.592275
    Validation loss decreased (0.684027 --> 0.592275).  Saving model ...
    Epoch: 6 	Training Loss: 1.349954 	Validation Loss: 0.552001
    Validation loss decreased (0.592275 --> 0.552001).  Saving model ...
    Epoch: 7 	Training Loss: 1.295717 	Validation Loss: 0.510571
    Validation loss decreased (0.552001 --> 0.510571).  Saving model ...
    Epoch: 8 	Training Loss: 1.194560 	Validation Loss: 0.487547
    Validation loss decreased (0.510571 --> 0.487547).  Saving model ...
    Epoch: 9 	Training Loss: 1.190721 	Validation Loss: 0.462933
    Validation loss decreased (0.487547 --> 0.462933).  Saving model ...
    Epoch: 10 	Training Loss: 1.151106 	Validation Loss: 0.431540
    Validation loss decreased (0.462933 --> 0.431540).  Saving model ...
    Epoch: 11 	Training Loss: 1.131936 	Validation Loss: 0.437716
    Epoch: 12 	Training Loss: 1.114191 	Validation Loss: 0.431364
    Validation loss decreased (0.431540 --> 0.431364).  Saving model ...
    Epoch: 13 	Training Loss: 1.087310 	Validation Loss: 0.423709
    Validation loss decreased (0.431364 --> 0.423709).  Saving model ...
    Epoch: 14 	Training Loss: 1.071814 	Validation Loss: 0.423844
    Epoch: 15 	Training Loss: 1.033489 	Validation Loss: 0.413295
    Validation loss decreased (0.423709 --> 0.413295).  Saving model ...
    Epoch: 16 	Training Loss: 1.021853 	Validation Loss: 0.401993
    Validation loss decreased (0.413295 --> 0.401993).  Saving model ...
    Epoch: 17 	Training Loss: 1.025200 	Validation Loss: 0.394757
    Validation loss decreased (0.401993 --> 0.394757).  Saving model ...
    Epoch: 18 	Training Loss: 0.986601 	Validation Loss: 0.381682
    Validation loss decreased (0.394757 --> 0.381682).  Saving model ...
    Epoch: 19 	Training Loss: 0.987548 	Validation Loss: 0.394329
    Epoch: 20 	Training Loss: 0.990282 	Validation Loss: 0.391783
    Epoch: 21 	Training Loss: 0.978521 	Validation Loss: 0.394503
    Epoch: 22 	Training Loss: 0.953556 	Validation Loss: 0.380711
    Validation loss decreased (0.381682 --> 0.380711).  Saving model ...
    Epoch: 23 	Training Loss: 0.963119 	Validation Loss: 0.375479
    Validation loss decreased (0.380711 --> 0.375479).  Saving model ...
    Epoch: 24 	Training Loss: 0.921474 	Validation Loss: 0.380990
    Epoch: 25 	Training Loss: 0.931744 	Validation Loss: 0.384962
    

### (IMPLEMENTATION) Test the Model

Try out your model on the test dataset of dog images. Use the code cell below to calculate and print the test loss and accuracy.  Ensure that your test accuracy is greater than 60%.


```python
test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)
```

    Test Loss: 0.419376
    
    
    Test Accuracy: 86% (724/836)
    

### (IMPLEMENTATION) Predict Dog Breed with the Model

Write a function that takes an image path as input and returns the dog breed (`Affenpinscher`, `Afghan hound`, etc) that is predicted by your model.  


```python
### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.

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

test_img_paths = sorted(glob('/data/dog_images/test/*/*'))
np.random.shuffle(test_img_paths)

for img_path in test_img_paths[0:5]:
    predicted_breed, true_breed = predict_breed_transfer(img_path)
    print("Predicted Breed:" , predicted_breed, "\n", "True Breed:" , true_breed)
    img=mpimg.imread(img_path)
    imgplot = plt.imshow(img)
    plt.show()

```

    Predicted Breed: Dalmatian 
     True Breed: 057.Dalmatian
    


![png](dog_files/dog_54_1.png)


    Predicted Breed: Chow chow 
     True Breed: 051.Chow_chow
    


![png](dog_files/dog_54_3.png)


    Predicted Breed: Field spaniel 
     True Breed: 066.Field_spaniel
    


![png](dog_files/dog_54_5.png)


    Predicted Breed: Border terrier 
     True Breed: 030.Border_terrier
    


![png](dog_files/dog_54_7.png)


    Predicted Breed: Miniature schnauzer 
     True Breed: 104.Miniature_schnauzer
    


![png](dog_files/dog_54_9.png)


---
<a id='step5'></a>
## Step 5: Write your Algorithm

Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither.  Then,
- if a __dog__ is detected in the image, return the predicted breed.
- if a __human__ is detected in the image, return the resembling dog breed.
- if __neither__ is detected in the image, provide output that indicates an error.

You are welcome to write your own functions for detecting humans and dogs in images, but feel free to use the `face_detector` and `human_detector` functions developed above.  You are __required__ to use your CNN from Step 4 to predict dog breed.  

Some sample output for our algorithm is provided below, but feel free to design your own user experience!

![Sample Human Output](images/sample_human_output.png)


### (IMPLEMENTATION) Write your Algorithm


```python
### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.

def run_app(img_path):
    '''
    Use pre-trained model to to check if the image at the given path
    contains a human being or a dog or none. 
    
    Args:
        img_path: path to an image
        
    Returns:
        print if a human face is detected or not
        print the dog breed or show that neither human face nor a dog detected 
    '''            
    
    is_human = face_detector(img_path)
    is_dog = dog_detector(img_path)
    breed = predict_breed_transfer(img_path)[0]
    
    if(is_human) and (is_dog) :
        print("Hello Human!")
        img= mpimg.imread(img_path)
        imgplot = plt.imshow(img)
        plt.show()
        print("You look like ...", breed)
        print("\n"*3)
        
    elif (is_human):
        print("Hello Human!")
        img= mpimg.imread(img_path)
        imgplot = plt.imshow(img)
        plt.show()
        print("You look like ...", breed)
        print("\n"*3)

    elif (is_dog):
        print("Hello Doggy!")
        img= mpimg.imread(img_path)
        imgplot = plt.imshow(img)
        plt.show()
        print("You look like ...", breed)
        print("\n"*3)
        
    else:
        print('Error!... I can not determine what you are!')
        img= mpimg.imread(img_path)
        imgplot = plt.imshow(img)
        plt.show()
        return    

```

---
<a id='step6'></a>
## Step 6: Test Your Algorithm

In this section, you will take your new algorithm for a spin!  What kind of dog does the algorithm think that _you_ look like?  If you have a dog, does it predict your dog's breed accurately?  If you have a cat, does it mistakenly think that your cat is a dog?

### (IMPLEMENTATION) Test Your Algorithm on Sample Images!

Test your algorithm at least six images on your computer.  Feel free to use any images you like.  Use at least two human and two dog images.  

__Question 6:__ Is the output better than you expected :) ?  Or worse :( ?  Provide at least three possible points of improvement for your algorithm.

__Answer:__ (Three possible points for improvement)

1. Find different models to applly transfer learning (such as resnet...) and achieve better accuracy.
2. Work with a bigger dataset to train the model.
3. Improve pre-processing steps, with more data augmentation.  


```python
## TODO: Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as many code cells as needed.

## suggested code, below
for file in np.hstack((human_files[:3], dog_files[:3])):
    run_app(file)
```

    Hello Human!
    


![png](dog_files/dog_59_1.png)


    You look like ... Beagle
    
    
    
    
    Hello Human!
    


![png](dog_files/dog_59_3.png)


    You look like ... Dachshund
    
    
    
    
    Hello Human!
    


![png](dog_files/dog_59_5.png)


    You look like ... American water spaniel
    
    
    
    
    Hello Doggy!
    


![png](dog_files/dog_59_7.png)


    You look like ... Bullmastiff
    
    
    
    
    Hello Doggy!
    


![png](dog_files/dog_59_9.png)


    You look like ... Bullmastiff
    
    
    
    
    Hello Doggy!
    


![png](dog_files/dog_59_11.png)


    You look like ... Bullmastiff
    
    
    
    
    


```python
test_files = np.array(glob("test_images/*"))

# print number of images in each dataset
print('There are %d total test images' % len(test_files))

for file in np.hstack((test_files[:])):
    run_app(file)
```

    There are 6 total test images
    Error!... I can not determine what you are!
    


![png](dog_files/dog_60_1.png)


    Hello Human!
    


![png](dog_files/dog_60_3.png)


    You look like ... Dachshund
    
    
    
    
    Hello Human!
    


![png](dog_files/dog_60_5.png)


    You look like ... Dachshund
    
    
    
    
    Error!... I can not determine what you are!
    


![png](dog_files/dog_60_7.png)


    Hello Human!
    


![png](dog_files/dog_60_9.png)


    You look like ... American water spaniel
    
    
    
    
    Hello Human!
    


![png](dog_files/dog_60_11.png)


    You look like ... Dachshund
    
    
    
    
    
