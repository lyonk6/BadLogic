import torch
import torchvision as tv
import torchvision.transforms as tfs

# So pytorch is kinda dope in that it provides training data you can use to
# test your shit. Here is a dataset of 50,000 categoriezed images. We use
# the CIFAR10 constructor(?) to transform all these images as torch tensors
# for us :)

# The trainset and testset represent two different member sets in the CIFAR
# database. Pull both the testing set and the training set:
def make_training_sets():
    trainset = tv.datasets.CIFAR10(root='./datasets/',
        train=True,
        download=True,
        transform=tfs.ToTensor())

    testset = tv.datasets.CIFAR10(root='./datasets/',
        train=False,
        download=True,
        transform=tfs.ToTensor())
    return trainset, testset

# Create a tuple for the training categories:
labels = ('plane', 'car', 'bird', 'cat', 'deer',
          'dog',  'frog', 'horse', 'ship', 'truck')

NUM_WORKERS=2

##########################################################################
# Within torch.utils.data we have the DataLoader class that holds our data
# and is a python iterable too. For the record, there are many other torch
# utilities including:
# bottleneck, checkpoint, cpp_extension, dlpack, mobile_optimizer,
# model_zoo, and tensorboard.
def make_training_loaders(trainset, testset):
    trainloader = torch.utils.data.DataLoader(trainset,
        batch_size=8,
        shuffle=True,
        num_workers=NUM_WORKERS) # How many processor do we want?


    testloader = torch.utils.data.DataLoader(testset,
        batch_size=8,
        shuffle=True,
        num_workers=NUM_WORKERS) # How many processor do we want?
    return trainloader, testloader



# If you are looking to get your mind blown, we can even view these images
# using matplotlib:
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    trainset, testset = make_training_sets()
    trainloader, testloader = make_training_loaders(trainset, testset)

    # Use Python's iter() function to grab a trainloader-full:
    images_batch, labels_batch = iter(trainloader).next()

    print(images_batch.shape) #torch.Size([8, 3, 32, 32])

    # Ok that's 8 images, 3 'channels', 32 pixels by 32 pixels. Note that
    # channels are single monochrome layers of a given image. For our use
    # here we use simple images of only 3 channels. Probably green, blue
    # and red.

    # Before we can view the images we have to ensure the channels are in
    # the third dimension for matplotlib to understand:
    img = tv.utils.make_grid(images_batch)
    print("Image shape should be [3, 36, 274]:", img.shape)
    imgTrans = np.transpose(img, (1,2,0))
    print("Transposed should be  [36, 274, 3]:", imgTrans.shape)

    plt.imshow(imgTrans)
    plt.axis('off')
    plt.show()
