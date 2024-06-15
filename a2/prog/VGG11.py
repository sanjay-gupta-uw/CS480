import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time

# Need to resize images from 28x28 to 32x32
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize the images
    ]
)

# Load the MNIST dataset
trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)

testset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

print("Length of trainsset: ", len(trainset))
print("Length of testset: ", len(testset))


class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        # Conv2D (in_channels, out_channels, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv6 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv8 = nn.Conv2d(512, 512, 3, 1, 1)

        # Batch Normalization Layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(512)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(512)

        # FC Layers
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, input):
        # Convolutional Layers
        l1 = F.max_pool2d(F.relu(self.bn1(self.conv1(input))), 2, 2)
        l2 = F.max_pool2d(F.relu(self.bn2(self.conv2(l1))), 2, 2)
        l3 = F.relu(self.bn3(self.conv3(l2)))
        l4 = F.max_pool2d(F.relu(self.bn4(self.conv4(l3))), 2, 2)
        l5 = F.relu(self.bn5(self.conv5(l4)))
        l6 = F.max_pool2d(F.relu(self.bn6(self.conv6(l5))), 2, 2)
        l7 = F.relu(self.bn7(self.conv7(l6)))
        l8 = F.max_pool2d(F.relu(self.bn8(self.conv8(l7))), 2, 2)

        # Flatten
        flat = torch.flatten(l8, 1)

        # Fully Connected Layers
        fc1 = self.dropout(F.relu(self.fc1(flat)))
        fc2 = self.dropout(F.relu(self.fc2(fc1)))
        output = self.fc3(fc2)

        return output

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        for epoch in range(3):
            for i, data in enumerate(trainloader, 0):
                input, labels = data
                optimizer.zero_grad()

                output = net(input)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                if i % 10 == 0:
                    print(f"Input Shape: {input.shape}")
                    print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
        print("Finished Training")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = VGG11().to(device)
print("USING: ", device)

start = time.time()
net.train()
end = time.time()
print(f"Time to train: {end - start}")
#
