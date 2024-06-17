import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import random

# Need to resize images from 28x28 to 32x32
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
# Test transform
test_transform_h = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomHorizontalFlip(p=1),
    ]
)
test_transform_v = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomVerticalFlip(p=1),
    ]
)

test_transform_guassian_0dot01 = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
    ]
)

test_transform_guassian_0dot1 = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1),
    ]
)

test_transform_guassian_1 = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 1),
    ]
)


augmentation_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomHorizontalFlip(p=random.random()),
        transforms.RandomVerticalFlip(p=random.random()),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1),
    ]
)

# Load the MNIST dataset
######################################################################################
trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)

testset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

######################################################################################
testset_h = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=test_transform_h
)
testloader_h = torch.utils.data.DataLoader(testset_h, batch_size=256, shuffle=False)

######################################################################################
testset_v = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=test_transform_v
)
testloader_v = torch.utils.data.DataLoader(testset_v, batch_size=256, shuffle=False)

######################################################################################
testset_guassian_0dot01 = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=test_transform_guassian_0dot01
)
testloader_guassian_0dot01 = torch.utils.data.DataLoader(
    testset_guassian_0dot01, batch_size=256, shuffle=False
)

testset_guassian_0dot1 = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=test_transform_guassian_0dot1
)
testloader_guassian_0dot1 = torch.utils.data.DataLoader(
    testset_guassian_0dot1, batch_size=256, shuffle=False
)

testset_guassian_1 = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=test_transform_guassian_1
)
testloader_guassian_1 = torch.utils.data.DataLoader(
    testset_guassian_1, batch_size=256, shuffle=False
)
######################################################################################
train_augmented = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=augmentation_transform
)
trainloader_augmented = torch.utils.data.DataLoader(
    train_augmented, batch_size=256, shuffle=True
)

test_augmented = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=augmentation_transform
)
testloader_augmented = torch.utils.data.DataLoader(
    test_augmented, batch_size=256, shuffle=False
)


print("Length of trainsset: ", len(trainset))
print("Length of testset: ", len(testset))

NUM_EPOCHS = 5
EPOCHS = range(NUM_EPOCHS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def train(net, loader, plot_data=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train_accuracy = []
    train_loss = []

    test_accuracy = []
    test_loss = []

    for epoch in EPOCHS:
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(loader, 0):
            input, labels = data
            input, labels = input.to(device), labels.to(device)
            optimizer.zero_grad()

            output = net(input)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = torch.max(output, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

            if i % 100 == 0:
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
        if plot_data:
            print(f"Training Accuracy for epoch {epoch}: {correct / total}")
            train_accuracy.append(correct / total)
            print(f"Loss for epoch {epoch}: {running_loss / trainloader.__len__()}")
            train_loss.append(running_loss / trainloader.__len__())
            # test after each epoch
            t_acc, t_loss = test(net, loader, track_loss=True)
            test_accuracy.append(t_acc)
            test_loss.append(t_loss)
    if plot_data:
        # plot test accuracy vs epoch
        plt.plot(EPOCHS, test_accuracy, label="Test Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Test Accuracy vs Epoch")
        plt.legend()
        plt.savefig("test_accuracy_b.png")
        plt.clf()
        plt.close()

        # plot training accuracy vs epoch
        plt.plot(EPOCHS, train_accuracy, label="Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy vs Epoch")
        plt.legend()
        plt.savefig("train_accuracy_b.png")
        plt.clf()
        plt.close()

        # plot test loss vs epoch
        plt.plot(EPOCHS, test_loss, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Test Loss vs Epoch")
        plt.legend()
        plt.savefig("test_loss_b.png")
        plt.clf()
        plt.close()

        # plot training loss vs epoch
        plt.plot(EPOCHS, train_loss, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss vs Epoch")
        plt.legend()
        plt.savefig("train_loss_b.png")
        plt.clf()
        plt.close()


def test(net, loader, track_loss=False):
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    running_loss = 0
    with torch.no_grad():
        for data in loader:
            input, labels = data
            input, labels = input.to(device), labels.to(device)
            output = net(input)
            # training loss
            loss = criterion(output, labels)
            running_loss += loss.item()
            _, pred = torch.max(output, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    # print(f"Test Accuracy: {correct / total}")
    # print(f"Test Loss: {running_loss / loader.__len__()}")
    if track_loss:
        return correct / total, running_loss / loader.__len__()
    return correct / total


def train_netBC(B=False):
    net = VGG11().to(device)
    train(net, trainloader, plot_data=B)
    if not B:
        return net


def B():
    train_netBC(True)
    print("Successfully executed B")


def C():
    net = train_netBC(False)
    # Produces the answers for the C)
    h_acc = test(net, testloader_h)
    v_acc = test(net, testloader_v)
    # plot accuracy vs flip type
    plt.bar(["Horizontal", "Vertical"], [h_acc, v_acc])
    plt.xlabel("Flip Type")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Flip Type")
    plt.legend()
    plt.savefig("flip_type_c.png")
    plt.clf()
    plt.close()

    # Gaussian noise
    g001_acc = test(net, testloader_guassian_0dot01)
    g01_acc = test(net, testloader_guassian_0dot1)
    g1_acc = test(net, testloader_guassian_1)
    # plot accuracy vs noise variance
    plt.bar(["0.01", "0.1", "1"], [g001_acc, g01_acc, g1_acc])
    plt.xlabel("Noise Variance")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Noise Variance")
    plt.legend()
    plt.savefig("noise_variance_c.png")
    plt.clf()
    plt.close()


def D():
    # Augmented data
    aug_net = VGG11().to(device)
    train(aug_net, trainloader_augmented)
    h_acc_aug = test(aug_net, testloader_v)
    v_acc_aug = test(aug_net, testloader_h)
    # plot accuracy vs flip type
    plt.bar(["Horizontal", "Vertical"], [h_acc_aug, v_acc_aug])
    plt.xlabel("Flip Type")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Flip Type")
    plt.legend()
    plt.savefig("flip_type_augmented.png")
    plt.clf()
    plt.close()

    g001_acc_aug = test(aug_net, testloader_guassian_0dot01)
    g01_acc_aug = test(aug_net, testloader_guassian_0dot1)
    g1_acc_aug = test(aug_net, testloader_guassian_1)
    # plot accuracy vs noise variance
    plt.bar(["0.01", "0.1", "1"], [g001_acc_aug, g01_acc_aug, g1_acc_aug])
    plt.xlabel("Noise Variance")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Noise Variance")
    plt.legend()
    plt.savefig("noise_variance_augmented.png")
    plt.clf()
    plt.close()


def __main__(script):
    print(f"Running  script for question 1.{script}")
    if script == 1:
        B()
    elif script == 2:
        C()
    elif script == 3:
        D()
    else:
        print("Invalid script number")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NN for Digit Classification")
    parser.add_argument("script", type=int, help="which answer to run")
    args = parser.parse_args()
    __main__(args.script)
