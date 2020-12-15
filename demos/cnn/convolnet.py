import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tfs
import dataset



trainset, testset = dataset.make_training_sets()
trainloader, testloader = dataset.make_training_loaders(trainset, testset)

# Let us define the imput size as the number of channels this
in_size=3
hid1_size=16
hid2_size=32
out_size=10
k_conv_size=5 # Use a 5x5 convolutional window

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_size, hid1_size, k_conv_size),
            nn.BatchNorm2d(hid1_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        # The size of the output of the first layer, is the
        # size of the input of the second layer.
        self.layer2 = nn.Sequential(
            nn.Conv2d(hid1_size, hid2_size, k_conv_size),
            nn.BatchNorm2d(hid2_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        # Calculate how many features need to be represented:
        # (last layer size x 5 pixel k_conv x 5 pixel k_conv)
        feature_size = hid2_size * k_conv_size * k_conv_size
        self.fc = nn.Linear(feature_size, out_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # Reshape the output so each image is represented as
        # a 1D vecotr to a feed into the linear layer.
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)

        return out


if __name__ == '__main__':
    model = ConvNet()
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(trainloader)
    num_epochs = 5

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(trainloader):
            outputs = model(images)
            loss=criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 2000 == 0:
                print('Epoch [{}/{}], Step[{},{}], Loss:{:.4f}'.format(
                    epoch+1, num_epochs, i+1, total_step, loss.item()))


    # We now have a fully trained model. What we do next?
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in testloader:
            outputs=model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on 10000 test images: {}%'
        .format(100*correct/total))
