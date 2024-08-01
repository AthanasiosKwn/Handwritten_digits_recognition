import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


# Define a transform to tensor
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensors
])

# Download the training set and apply transform (If it is already downloaded in the specified folder, it will not download again)
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Download the test set and apply transform (If it is already downloaded in the specified folder, it will not download again)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)



class FCNN(nn.Module):
    def __init__(self):
        # Call the constructor of the Parent Class nn.Module
        super(FCNN, self).__init__()
        # Fully connected Layers. The input is of dim 50 for the first layer because that is the dimension of the histograms (feature vector) for each image
        self.fc1 = nn.Linear(50,128)
        self.fc2 = nn.Linear(128,10)
    
    def forward(self,x):
        # x is a mini batch, a set of images. forward method defines how is the mini batch passed through the network
        x = F.relu(self.fc1(x))
        # Cross entropy loss function will be utilized. It internally applies softmax so we do not have to incorporate it in the model architecture
        x = self.fc2(x) 
        return x



# Training function
def train_model(model, trainloader, valloader, criterion, optimizer, num_epochs=50):
    best_model_wts = None
    best_acc = 0.0
    epoch_train_loss_list = []
    epoch_eval_loss_list = []
    
    # How many times the model sees the whole dataset
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase. First phase is the train phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = trainloader   # use the train mini batches
            else:
                model.eval()  # Set model to evaluate mode
                dataloader = valloader     # use the validation mini batches
            
            # Initialize the 
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over the mini batches. Train or evaluation depending on the phase
            for inputs, labels in dataloader:
                # Move inputs and labels to the device (GPU or CPU)
                inputs, labels = inputs.to(device), labels.to(device)
              
                
                # Zero the parameter gradients. The gradients must be set to 0 for each mini batch to avoid gradient accumulation
                optimizer.zero_grad()
                
                # Forward pass. It is executed on both train and evaluation phase. Gradient computation is only executed during train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # Returns max_value, index along the columns(dim=1). Outputs is 2D matrix, where each row is the softmax output (10 values) for an image of the minibatch
                    _, preds = torch.max(outputs, 1)
                    
                    # Backward pass and optimize only if in training phase
                    if phase == 'train':
                        # Backpropagate
                        loss.backward()
                        # Update weights
                        optimizer.step()
                
                # loss.item() is the average loss for each mini batch. Multiplying with the size of the minibatch returns the total loss for the minibatch
                # We accumulate this total loss over all mini batches
                running_loss += loss.item() * inputs.size(0)
                # Accumulate the correct number of predictions across all mini batches
                running_corrects += torch.sum(preds == labels.data)
            
            # The epoch loss which is the average loss across all the mini batches. This is the average train loss or evaluation loss depending on the phase
            epoch_loss = running_loss / len(dataloader.dataset)
            if phase == 'train':
                epoch_train_loss_list.append(epoch_loss)
            elif phase == 'val':
                epoch_eval_loss_list.append(epoch_loss)
            # The epoch accuracy
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Deep copy the model if it has the best accuracy so far, based on evaluation error
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
    
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    

    # Return the best model found
    return model, epoch_train_loss_list, epoch_eval_loss_list


# Initialize SIFT detector
sift = cv.SIFT_create()

def extract_sift_descriptors(dataset):
    # Initialize lists containing sift descriptors and labels
    sift_descriptors = []
    labels = []
    # Iterate through images
    for img, label in dataset:
        # Convert image from Pytorch tensor to numpy array and to grayscale and scale back to [0,255]
        img_np = img.numpy().squeeze() * 255    #.squeeze() Removes all  dimensions of size 1 from the array
        img_np = img_np.astype(np.uint8)
        
        # Detect keypoints and compute descriptors. For an image, for each key point detected a descriptor is calculated. A descriptor is a 128 dimensional vector
        keypoints, descriptors = sift.detectAndCompute(img_np, None)
        
        if descriptors is not None:
            sift_descriptors.append(descriptors)
            labels.append(label)
    return sift_descriptors, labels

# Calculate SIFT descriptors for the images on the train and the test set
train_descriptors, train_labels = extract_sift_descriptors(trainset)
test_descriptors, test_labels = extract_sift_descriptors(testset)




def create_bow(descriptor_list, k=50):
    # Stack all descriptors vertically in a numpy array. Descriptor list contains list of descriptors. Each list corresponds to a single image, and contains the decriptors of each key points
    descriptors = np.vstack(descriptor_list)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k).fit(descriptors)
    
    return kmeans

def compute_histograms(descriptor_list, kmeans):
    # Initialize the list that contains the image descriptor histograms. The image descriptor histograms will be the final feature vectors for the images
    histograms = []
    for descriptors in descriptor_list:
        histogram = np.zeros(len(kmeans.cluster_centers_))
        if descriptors is not None:
            clusters = kmeans.predict(descriptors)
            for cluster in clusters:
                histogram[cluster] += 1
        histograms.append(histogram)
    return histograms

# Create a BoW model using the training descriptors
kmeans = create_bow(train_descriptors, k=50)

# Compute histograms for training and testing sets
train_histograms = compute_histograms(train_descriptors, kmeans)
test_histograms = compute_histograms(test_descriptors, kmeans)

# Convert to numpy arrays
train_histograms = np.array(train_histograms)
test_histograms = np.array(test_histograms)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Convert data to Pytorch Tensors
train_histograms = torch.tensor(train_histograms, dtype=torch.float32)
test_histograms = torch.tensor(test_histograms, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# Convert into Tensor Dataset (concatenate data to corresponding labels)
train_data = TensorDataset(train_histograms, train_labels)
test_data = TensorDataset(test_histograms, test_labels)

# Train and test model for different percentages of the original train set
for percentage in [0.05, 0.1, 0.5, 1]:
    # Create subset
    subset_size = int(percentage*len(train_data))
    null_size = len(train_data) - subset_size
    sub_set, _ = random_split(train_data, [subset_size,null_size])

    # Split subset into train and validation set
    train_size = int(0.8 * len(sub_set))
    val_size = len(sub_set) - train_size
    train_set, val_set = random_split(sub_set, [train_size,val_size])


    # Create data loaders to load data in batches
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True,drop_last = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False,drop_last = True)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False,drop_last = True) 

    # Initialize the model, loss function, and optimizer to be used
    model = FCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Move the model to the appropriate device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Train the model
    model, train_loss, eval_loss = train_model(model, trainloader, valloader, criterion, optimizer, num_epochs=150)
    # Save the best model state
    torch.save(model.state_dict(), f'FCNN_best_model_{percentage}.pth')

    # Plotting the train loss line
    plt.plot([x for x in range(1, len(train_loss)+1)], train_loss , label='train loss')

    # Plotting the validation loss line
    plt.plot([x for x in range(1, len(eval_loss)+1)], eval_loss, label='validation loss')

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Average Cross Entropy Loss')
    plt.title('Train vs Evaluation Loss')

    # Adding legend
    plt.legend()
    # Save fig
    plt.savefig(f"FCNN_train_vs_eval_loss_{percentage}.jpg")
    # Displaying the plot
    plt.show()

    # Testing on the test set

    # Set the model to evaluation mode
    model.eval() 

    # Initialize prediction and label lists
    predictions = []
    ground_truth = []

    with torch.no_grad():   # disables grad calculations
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            # Make predictions
            outputs = model(inputs)
            _, preds = torch.max(outputs,dim=1)
        
            # Convert predictions and labels from pytorch tensors to np arrays
            predictions.extend(preds.cpu().numpy())
            ground_truth.extend(labels.cpu().numpy())
   
    # Calculate accuracy on test set
    test_accuracy = accuracy_score(ground_truth,predictions)
    print(f"Accuracy on test set {percentage}: ", test_accuracy)

    # Calculate Confusion Matrix
    cm = ConfusionMatrixDisplay(confusion_matrix(ground_truth, predictions))
    cm.plot()
    plt.title('Confusion Matrix')
    plt.savefig(f"FCNN_confusion_matrix_{percentage}.jpg")
    plt.show() 