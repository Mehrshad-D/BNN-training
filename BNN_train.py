import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Binarize function with STE (Straight Through Estimator)
class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # STE: identity gradient where |input| <= 1, otherwise 0
        grad_input[input.abs() > 1] = 0
        grad_input *= 0.5  # Gradient scaling
        return grad_input

# Define a Binarized Linear Layer
class BinarizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BinarizedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        # Binarize the weights
        binarized_weights = Binarize.apply(self.linear.weight)
        # Apply weight clipping
        with torch.no_grad():
            binarized_weights.clamp_(-1, 1)
        x = nn.functional.linear(x, binarized_weights)
        return x

# Define the Binarized Neural Network with Batch Normalization
class BNN(nn.Module):
    def __init__(self):
        super(BNN, self).__init__()
        self.fc1 = BinarizedLinear(28 * 28, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = BinarizedLinear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = BinarizedLinear(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc4 = BinarizedLinear(1024, 10)
        self.bn4 = nn.BatchNorm1d(10)
        

    def forward(self, x):
        # x = x.view(-1, 28 * 28)  # Flatten the input
        # x = self.bn1(torch.relu(self.fc1(x)))
        # x = self.bn2(torch.relu(self.fc2(x)))
        # x = self.fc3(x)
        # return x
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = Binarize.apply(self.bn1(self.fc1(x)))
        x = Binarize.apply(self.bn2(self.fc2(x)))
        x = Binarize.apply(self.bn3(self.fc3(x)))
        x = self.fc4(x) # output stays real (not binary)
        return x
    
# Training Loop
def train(model, trainloader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(trainloader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0
    print("Training completed!")

# Evaluation Function
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

# Save Binarized Weights
def save_binarized_weights(model, filename='binarized_weights_1024.pth'):
    binarized_weights = {name: Binarize.apply(param).cpu().numpy() for name, param in model.state_dict().items()}
    torch.save(binarized_weights, filename)
    print(f'Binarized weights saved to {filename}')    

# Transformations for MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Datasets and Dataloaders
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

# Model, Loss, and Optimizer
model = BNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate

# Training and Evaluation
train(model, trainloader, criterion, optimizer, epochs=30)
evaluate(model, testloader)
torch.save(model.state_dict(), 'trained_bnn_1024.pth') # full model float weights
save_binarized_weights(model) # binarized weights