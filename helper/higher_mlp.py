import torch 
import os 
import numpy as np 
import cv2 
import torch
import torch.nn as nn
import torch.optim as optim


array2string = lambda x: ''.join(str(bit) for bit in x)

# Function to generate random binary numbers for each class
def generate_binary_numbers(num_classes):
    binary_numbers = []
    for i in range(num_classes):
        binary_number = np.random.randint(2, size=12)
        binary_numbers.append(binary_number)
    return np.array(binary_numbers)

# Function to generate random values for each binary number
def generate_random_values(binary_numbers, num_samples_per_class):
    data = []
    for binary_number in binary_numbers:
        class_data = []
        for _ in range(num_samples_per_class):
            random_values = np.random.rand(12)  # Generate 12 random values between 0 and 1
            class_data.append((random_values > 0.5).astype(int) if np.sum(binary_number) > 5 else (random_values < 0.5).astype(int))
        data.extend(class_data)
    return np.array(data)


def dataloader():
    random_array = np.random.rand(10000000, 12)
    binary_array = np.where(random_array > 0.5, 1, 0)
    binary_to_decimal_map = {}

    # Populate the dictionary
    for binary_value in range(2**12):
        binary_string = format(binary_value, '012b')  # Convert to binary string with leading zeros
        decimal_value = int(binary_string, 2)         # Convert binary string to decimal
        binary_to_decimal_map[binary_string] = decimal_value
    
    string_binary_array = [array2string(b) for b in  binary_array]
    decimal_class = [binary_to_decimal_map[i] for i in string_binary_array]
    
    return torch.from_numpy(random_array) , torch.from_numpy(binary_array) ,  torch.from_numpy(np.array(decimal_class))



class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Linear(12, 12)  # Single fully connected layer

    def forward(self, x):
        x = self.fc(x)
        return x

def triplet_loss(anchor, positive, negative, margin=1.0):
    distance_positive = torch.norm(anchor - positive, dim=-1)
    distance_negative = torch.norm(anchor - negative, dim=-1)
    loss = torch.relu(distance_positive - distance_negative + margin)
    return loss.mean()
def generate_triplets(data, labels, batch_size):
    """
    Generate anchor, positive, and negative data samples for triplet loss training.

    Args:
    - data (torch.Tensor): Tensor of shape [N, 12] containing the data samples.
    - labels (torch.Tensor): Tensor of shape [N] containing the class labels.
    - batch_size (int): Number of triplets to generate.

    Returns:
    - anchor (torch.Tensor): Tensor of shape [batch_size, 12] containing anchor data samples.
    - positive (torch.Tensor): Tensor of shape [batch_size, 12] containing positive data samples.
    - negative (torch.Tensor): Tensor of shape [batch_size, 12] containing negative data samples.
    """
    anchor = []
    positive = []
    negative = []

    unique_labels = labels.unique()
    for _ in range(batch_size):
        label = unique_labels[torch.randint(0, len(unique_labels), (1,))]
        # Get indices of data samples with this label
        label_indices = (labels == label).nonzero().view(-1)
        # If there's only one sample for this class, skip it
        if len(label_indices) < 2:
            continue
        # Randomly select an anchor and a positive sample
        anchor_index = torch.randint(0, len(label_indices), (1,))
        positive_index = torch.randint(0, len(label_indices), (1,))
        while positive_index == anchor_index:
            positive_index = torch.randint(0, len(label_indices), (1,))
        anchor.append(data[label_indices[anchor_index]])
        positive.append(data[label_indices[positive_index]])

        # Randomly select a different class for negative sample
        negative_label = label
        while negative_label == label:
            negative_label = unique_labels[torch.randint(0, len(unique_labels), (1,))]
        negative_indices = (labels == negative_label).nonzero().view(-1)
        negative_index = torch.randint(0, len(negative_indices), (1,))
        negative.append(data[negative_indices[negative_index]])

    anchor = torch.stack(anchor).to(torch.float32)
    positive = torch.stack(positive).to(torch.float32)
    negative = torch.stack(negative).to(torch.float32)

    return anchor, positive, negative
def train():
    # Generate dummy data
    random_array,bianry_array,binary_class = dataloader()
    
    # Initialize model and optimizer
    model = MLP()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 200
    batch_size = 3000

    for epoch in range(num_epochs):
        avg_loss = 0.
        for i in range(0, 100):
            # anchor = data[i:i+batch_size]
            # positive = labels[i:i+batch_size]
            # negative = torch.cat((labels[:i], labels[i+batch_size:]), dim=0)  # Assuming negative samples are randomly selected
            anchor, positive, negative = generate_triplets(random_array,binary_class,32)

            
            optimizer.zero_grad()
            anchor_outputs = model(anchor)
            positive_outputs = model(positive)
            negative_outputs = model(negative)
            loss = triplet_loss(anchor_outputs, positive_outputs, negative_outputs)
            
            # import pdb;pdb.set_trace()
            loss.backward()
            optimizer.step()
            avg_loss +=loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss/100}')
    import pdb;pdb.set_trace()


train()

