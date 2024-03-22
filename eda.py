import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as dsets
from torchvision import transforms
import torch

''' find_mean_img
    Take the average value of each pixel across all observations

    @params:
        - full_mat: the matrix of all images
        - title: the title of the plot
        - size: the desired size of the image
'''
def find_mean_img(data, title, size = (224,224), plot = False):
    # calculate the average
    mean_img = torch.mean(torch.stack(data), dim=0)
    # reshape it back to a matrix
    mean_img = mean_img.reshape(size)
    if plot:
        plt.imshow(mean_img, vmin=0, vmax=255, cmap='Greys_r')
        plt.title(f'Average {title}')
        plt.axis('off')
        plt.show()
    return mean_img

##########################################################################################

    
def full_eda(plot_bool=True):
    # Importing the dataset
    transform = transforms.Compose([
        transforms.Resize((128,128)), # Scaling the image to 128 x 128
        transforms.ToTensor(), # Converting the image to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalizing the image
    ])

    all_data = dsets.ImageFolder(root = './traindata', transform = transform)
    print(f'[INFO] Loaded all data')

    class_names = all_data.classes
    print(f'[INFO] Class names: \n{class_names}')
    class_indices = all_data.class_to_idx

    class_data ={}
    for cl in class_names:
        class_data[cl] = [all_data[i] for i in range(len(all_data)) if all_data[i][1] == class_indices[cl]]
    
    for cl in class_names:
        print(f'[INFO] Number of {cl} images: {len(class_data[cl])}')

full_eda(plot_bool=False)

def meanAndStd():
    # Define the transformations to apply to the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the images to 224x224 pixels
        transforms.ToTensor()  # Convert the images to PyTorch tensors
    ])

    # Load the images and apply the transformations
    dataset = dsets.ImageFolder(root="traindata", transform=transform)

    # Calculate the mean and standard deviation of the pixel values
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(dataset)
    std /= len(dataset)

    print('Mean:', mean)
    print('Std:', std)

meanAndStd()