import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random

import random
import math

class NoisyBinarize(torch.autograd.Function):
    recorded_abs_vals = []
    @staticmethod
    def forward(ctx, input, sigma=5.0):
        noisy_output = input.clone()

        for i in range(noisy_output.shape[0]):
            for j in range(noisy_output.shape[1]):
                val = noisy_output[i, j].item()
                abs_val = abs(val)
                # print(abs_val)
                NoisyBinarize.recorded_abs_vals.append(abs_val)
                # noisy_output[i, j] = 1 if val > 0 else -1

                if abs_val > 50:
                    # Strong activation → no noise
                    noisy_output[i, j] = 1 if val > 0 else -1
                else:
                    # Gaussian-based probability of flipping
                    prob = 0.5 * math.exp(- (abs_val ** 2) / (2 * sigma ** 2))

                    if random.random() < prob:
                        # Flip the sign
                        noisy_output[i, j] = -1 if val > 0 else 1
                    else:
                        # Keep normal sign
                        noisy_output[i, j] = 1 if val > 0 else -1

        return noisy_output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input *= 0.5
        return grad_input, None  # second arg is for sigma


# class NoisyBinarize(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         noisy_output = input.clone()

#         # Loop through all elements and apply noise depending on magnitude
#         for i in range(noisy_output.shape[0]):
#             for j in range(noisy_output.shape[1]):
#                 val = noisy_output[i, j].item()
#                 abs_val = abs(val)
                
#                 if abs_val > 100:
#                     noisy_output[i, j] = 1 if val > 0 else -1
#                 elif abs_val < 5:
#                     noisy_output[i, j] = 1 if random.random() < 0.5 else -1
#                 else:
#                     # Linear interpolation of noise probability
#                     prob = 0.5 * (1 - (abs_val - 5) / 95)  # goes from 0.5 to 0 as abs_val→100
#                     if random.random() < prob:
#                         noisy_output[i, j] = -1 if val > 0 else 1  # flip
#                     else:
#                         noisy_output[i, j] = 1 if val > 0 else -1
#         return noisy_output

#     @staticmethod
#     def backward(ctx, grad_output):
#         # Use STE for backward (same as Binarize)
#         grad_input = grad_output.clone()
#         grad_input *= 0.5
#         return grad_input


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
        # x = NoisyBinarize.apply(self.bn1(self.fc1(x)))
        # x = NoisyBinarize.apply(self.bn2(self.fc2(x)))
        # x = NoisyBinarize.apply(self.bn3(self.fc3(x)))
        # x = self.fc4(x) # output stays real (not binary)
        # return x

        x = x.view(-1, 28 * 28)  # Flatten the input
        x = Binarize.apply(self.bn1(self.fc1(x)))
        x = Binarize.apply(self.bn2(NoisyBinarize.apply(self.fc2(x))))
        x = Binarize.apply(self.bn3(NoisyBinarize.apply(self.fc3(x))))
        x = self.fc4(x) # output stays real (not binary)
        return x
    
        # x = x.view(-1, 28 * 28)  # Flatten the input
        # x = Binarize.apply(self.bn1(self.fc1(x)))
        # x = Binarize.apply(self.bn2(self.fc2(x)))
        # x = Binarize.apply(self.bn3(self.fc3(x)))
        # x = self.fc4(x) # output stays real (not binary)
        # return x
    
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

# Transformations for MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

def test_weight_perturbation(model_file='trained_bnn.pth', testloader=None, flip_ratio=0.01):
    print(f"\n--- Testing weight flipping (flip ratio: {flip_ratio*100:.2f}%) ---")
    
    # Step 1: Load the trained model from file
    perturbed_model = BNN()
    perturbed_model.load_state_dict(torch.load(model_file))
    perturbed_model.eval()

    # Step 2: Perturb the weights
    with torch.no_grad():
        for name, param in perturbed_model.named_parameters():
            if "weight" in name and param.ndim == 2:  # Only perturb linear weights
                binary_weights = Binarize.apply(param)

                # Flatten and flip random indices
                flat = binary_weights.view(-1)
                num_to_flip = int(flip_ratio * flat.numel())
                flip_indices = torch.randperm(flat.numel())[:num_to_flip]

                # Toggle: +1 ↔ -1
                flat[flip_indices] *= -1

                # Reshape and assign back
                new_weights = flat.view_as(param)
                param.copy_(new_weights)

    # Step 3: Evaluate the perturbed model
    evaluate(perturbed_model, testloader)

    import matplotlib.pyplot as plt

def perturb_and_plot(model_file='trained_bnn.pth', testloader=None, flip_ratios=[0.0, 0.01, 0.05, 0.10, 0.20, 0.30]):
    accuracies = []

    for flip_ratio in flip_ratios:
        print(f"\n--- Testing flip ratio {flip_ratio*100:.1f}% ---")

        # Load the trained model
        perturbed_model = BNN()
        perturbed_model.load_state_dict(torch.load(model_file))
        perturbed_model.eval()

        # Perturb the weights
        with torch.no_grad():
            for name, param in perturbed_model.named_parameters():
                if "weight" in name and param.ndim == 2:
                    binary_weights = Binarize.apply(param)

                    flat = binary_weights.view(-1)
                    num_to_flip = int(flip_ratio * flat.numel())
                    if num_to_flip > 0:
                        flip_indices = torch.randperm(flat.numel())[:num_to_flip]
                        flat[flip_indices] *= -1

                    new_weights = flat.view_as(param)
                    param.copy_(new_weights)

        # Evaluate the perturbed model
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                outputs = perturbed_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        accuracies.append(acc)
        print(f"Accuracy: {acc:.2f}%")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot([r * 100 for r in flip_ratios], accuracies, marker='o')
    plt.title('Accuracy vs. Flip Percentage')
    plt.xlabel('Flip Percentage (%)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.show()

def plot_activation_magnitudes(model, dataloader):
    model.eval()
    abs_activations = []

    with torch.no_grad():
        for images, _ in dataloader:
            x = images.view(-1, 28 * 28)

           
            # First layer
            a1 = model.fc1(x)
            # abs_activations.append(a1.abs().flatten())
            a1 = model.bn1(a1)

            # Second layer
            a2 = model.fc2(torch.sign(a1))  # simulate binarized activations
            abs_activations.append(a2.abs().flatten())
            a2 = model.bn2(a2)
            
            
            # Third layer
            a3 = model.fc3(torch.sign(a2))
            abs_activations.append(a3.abs().flatten())
            a3 = model.bn3(a3)

            a4 = model.fc4(torch.sign(a3))
            # print(a4)
            abs_activations.append(a4.abs().flatten())
            
            # break  # Only use first batch to keep it fast

    # Concatenate all absolute activations
    # print(abs_activations)
    all_abs_vals = torch.cat(abs_activations).cpu().numpy()
    

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(all_abs_vals, bins=100, color='skyblue', edgecolor='black', density=True)
    plt.title("Distribution of Absolute Pre-Binarization Activations")
    plt.xlabel("|activation|")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()

    # Plot CDF
    sorted_vals = np.sort(all_abs_vals)
    cdf = np.arange(len(sorted_vals)) / len(sorted_vals)

    plt.figure(figsize=(8, 5))
    plt.plot(sorted_vals, cdf, color='green')
    plt.title("Cumulative Distribution of |activation|")
    plt.xlabel("|activation|")
    plt.ylabel("Cumulative Probability")
    plt.grid(True)
    plt.show()


def plot_abs():
    abs_vals = NoisyBinarize.recorded_abs_vals

    plt.figure(figsize=(8, 5))
    plt.hist(abs_vals, bins=100, color='orange', edgecolor='black', density=True)
    plt.title("Histogram of Absolute Activation Values (Before Noisy Binarization)")
    plt.xlabel("|activation|")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()


# Datasets and Dataloaders
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

# Model, Loss, and Optimizer
model = BNN()
model.load_state_dict(torch.load('trained_bnn_1024.pth'))
evaluate(model, testloader) # main model
# test_weight_perturbation('trained_bnn_1024.pth', testloader, 0.05)
# perturb_and_plot('trained_bnn_1024.pth', testloader)
# plot_activation_magnitudes(model, testloader)
plot_abs()
