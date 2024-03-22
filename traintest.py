import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.datasets as dsets
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def preprocessing(pca_components=None):
    # define a transformation to normalize the data
    transform = transforms.Compose([
        transforms.Resize((128,128)), # Scaling the image to 128 x 128
        transforms.ToTensor(), # Converting the image to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalizing the image
    ])

    # define the training and test datasets
    all_data = dsets.ImageFolder(root = './traindata', transform = transform)

    # setting split size
    train_size = int(len(all_data) * 0.8)
    test_size = len(all_data) - train_size

    # split the training data into training and test sets
    train_data, test_data = random_split(all_data, [train_size, test_size])

    # define the training, test, and test dataloaders
    train_loader = DataLoader(train_data, batch_size = 100, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = 100, shuffle = True)

    if pca_components is not None:
        flat_images = []
        targets = []
        for images, labels in train_loader:
            flat_images.append(images.view(images.size(0), -1))
            targets.append(labels)
        flat_images = torch.cat(flat_images, dim = 0)
        targets = torch.cat(targets, dim = 0)
        pca = PCA(n_components = pca_components)
        flat_images = pca.fit_transform(flat_images)

        # Convert NumPy array back to tensor
        flat_images = torch.from_numpy(flat_images)

        # Reshape back to the original shape if needed
        flat_images = flat_images.view(-1, pca_components)

        # Update the DataLoader
        train_data = torch.utils.data.TensorDataset(flat_images, targets)
        train_loader = DataLoader(train_data, batch_size=100, shuffle=True)

        # Update the test data with PCA
        flat_images = []
        for images, labels in test_loader:
            flat_images.append(images.view(images.size(0), -1))
        flat_images = torch.cat(flat_images, dim = 0)
        flat_images = pca.transform(flat_images)
    return train_loader, test_loader

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, out):
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out) # ReLU activation function
        out = self.fc2(out)
        return out
    
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0) # total number of labels
            correct += (predicted == labels).sum().item() # total correct predictions
    return (correct / total)

def train_MLP(data, test_loader, model, optimizer, num_epochs):
    # Training the model
    loss_function = nn.CrossEntropyLoss()
    loss_list = []
    acc_list = []
    model.train()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(data):
            # Convert torch tensor to Variable
            images = images.view(-1, 3*128*128).requires_grad_()

            # Forward + Backward + Optimize
            optimizer.zero_grad() # zero the gradient buffer
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data)}], Loss: {loss.item():.4f}')

        test_acc = evaluate_model(model, test_loader)
        loss_list.append(loss.item())
        acc_list.append(test_acc)
        print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {loss.item()}, Test accuracy: {test_acc:.4f}')
    return loss_list, acc_list

def run():
    train, test = preprocessing(pca_components=833 )
    model = MLP(3*128*128, 100, 3)
    optimizer = optim.SGD(model.parameters(), lr = 0.01)
    epochs = 10
    loss_list, acc_list = train_MLP(train, test, model, optimizer, epochs)

run()
