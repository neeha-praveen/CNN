import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision
import torchvision.transforms as transforms

class SampleCNN(nn.Module):
    def __init__(self):
        super(SampleCNN,self).__init__()
        # Convolutional layer
        self.conv = nn.Conv2d(  
            in_channels=1,
            out_channels=32,
            kernel_size=3, #3x3
            stride=1,
            padding=1
        )
        # Activation function
        self.relu = nn.ReLU()  
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  
        # Dense layer
        self.dl = nn.Linear(32*14*14,10)  
    
    def forward(self,x):
        x = self.pool(self.relu(self.conv(x)))
        x = x.view(x.size(0),-1) # flattening to 1D
        x = self.dl(x)
        return x
    
# Loading Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))
])
trainset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download = True
)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=32,
    shuffle=True
)

# initializing model, loss, and optimizer
model = SampleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop : 1 Epoch
for images, labels in trainloader:
    optimizer.zero_grad()
    outputs=model(images)
    loss=criterion(outputs,labels)
    loss.backward()
    optimizer.step()
    break # for 1 batch only

print("CNN training completed")