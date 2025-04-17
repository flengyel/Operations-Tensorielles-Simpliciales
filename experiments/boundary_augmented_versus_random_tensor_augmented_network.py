import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tensor_ops import bdry_mod1, degen  # Ensure tensor_ops.py is correctly implemented and accessible

# pytorch implementation of max_norm
def max_norm_pytorch(t: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.abs(t))


# ----------------------------
# Network Classes
# ----------------------------

class OriginalNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, negative_slope=0.01):
        super(OriginalNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], hidden_sizes[4])
        self.fc6 = nn.Linear(hidden_sizes[4], output_size)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))
        x = self.fc6(x)
        return x

class BoundaryAugmentedNet(OriginalNet):
    def __init__(self, input_size, hidden_sizes, output_size, boundary_scale=7e-3):
        super(BoundaryAugmentedNet, self).__init__(input_size, hidden_sizes, output_size)
        self.boundary_scale = boundary_scale

    def integrate_boundary(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'fc5.weight' in name:  # Only modify the fc5 layer
                    boundary = bdry_mod1(param.detach().cpu().numpy())
                    k = min(boundary.shape) // 2
                    boundary_degen = degen(boundary, k)
                    boundary_degen_tensor = torch.from_numpy(boundary_degen).to(param.device).type(param.dtype)

                    # Normalize the boundary tensor to avoid large updates
                    norm = max_norm_pytorch(boundary_degen_tensor)
                    if norm != 0:  # Prevent division by zero
                        boundary_degen_tensor = boundary_degen_tensor / norm

                    if param.data.shape != boundary_degen_tensor.shape:
                        raise ValueError(f"Shape mismatch for {name}: {param.data.shape} vs {boundary_degen_tensor.shape}")

                    # Update weights with normalized boundary tensor
                    param.data += self.boundary_scale * boundary_degen_tensor




class RandomTensorAugmentedNet(OriginalNet):
    def __init__(self, input_size, hidden_sizes, output_size, random_tensor_scale=7e-3):
        super(RandomTensorAugmentedNet, self).__init__(input_size, hidden_sizes, output_size)
        self.random_tensor_scale = random_tensor_scale

    def integrate_random_tensor(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'fc5.weight' in name:  # Only modify the fc5 layer
                    original_shape = param.detach().cpu().numpy().shape
                    boundary_shape = tuple(dim - 1 for dim in original_shape)
                    rng = np.random.default_rng(seed=42)
                    random_tensor = rng.standard_normal(boundary_shape)
                    k = min(boundary_shape) // 2
                    random_degen_tensor = degen(random_tensor, k)
                    random_degen_tensor = torch.from_numpy(random_degen_tensor).to(param.device).type(param.dtype)

                    param.data += self.random_tensor_scale * random_degen_tensor

# ----------------------------
# Data Loading
# ----------------------------

def load_data(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

# ----------------------------
# Unified Training Function
# ----------------------------

def train_network(net, train_loader, criterion, optimizer, scheduler, num_epochs=40, augment_type=None, device=torch.device('cpu')):
    net.to(device)  # Ensure the network is on the correct device
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            if augment_type == 'boundary' and isinstance(net, BoundaryAugmentedNet):
                net.integrate_boundary()
            elif augment_type == 'random' and isinstance(net, RandomTensorAugmentedNet):
                net.integrate_random_tensor()

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.6f}")

# ----------------------------
# Main Experiment
# ----------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select device dynamically

    # Define network parameters
    input_size = 28 * 28  # MNIST images are 28x28
    hidden_sizes = [512, 256, 128, 64, 32]  # Larger fully connected network
    output_size = 10  # MNIST has 10 classes (digits 0-9)

    # Load MNIST data
    train_loader, _ = load_data(batch_size=64)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # ----------------------------
    # Training Original Network
    # ----------------------------
#    print("Training Original Network with ReLU:")
#    original_net = OriginalNet(input_size, hidden_sizes, output_size).to(device)
#    original_optimizer = optim.Adam(original_net.parameters(), lr=0.001)
#    original_scheduler = optim.lr_scheduler.StepLR(original_optimizer, step_size=20, gamma=0.1)
#    train_network(original_net, train_loader, criterion, original_optimizer, original_scheduler, num_epochs=40, device=device)

    # ----------------------------
    # Training Boundary-Augmented Network
    # ----------------------------
    print("\nTraining Boundary-Augmented Network with ReLU:")
    boundary_augmented_net = BoundaryAugmentedNet(input_size, hidden_sizes, output_size, boundary_scale=7e-3).to(device)
    boundary_augmented_optimizer = optim.Adam(boundary_augmented_net.parameters(), lr=0.001, weight_decay=1e-5)
    boundary_augmented_scheduler = optim.lr_scheduler.StepLR(boundary_augmented_optimizer, step_size=20, gamma=0.1)
    train_network(boundary_augmented_net, train_loader, criterion, boundary_augmented_optimizer, boundary_augmented_scheduler, num_epochs=40, augment_type='boundary', device=device)

    # ----------------------------
    # Training Random Tensor Augmented Network
    # ----------------------------
    print("\nTraining Random Tensor Augmented Network with ReLU:")
    random_tensor_augmented_net = RandomTensorAugmentedNet(input_size, hidden_sizes, output_size, random_tensor_scale=7e-3).to(device)
    random_tensor_augmented_optimizer = optim.Adam(random_tensor_augmented_net.parameters(), lr=0.001, weight_decay=1e-5)
    random_tensor_augmented_scheduler = optim.lr_scheduler.StepLR(random_tensor_augmented_optimizer, step_size=20, gamma=0.1)
    train_network(random_tensor_augmented_net, train_loader, criterion, random_tensor_augmented_optimizer, random_tensor_augmented_scheduler, num_epochs=40, augment_type='random', device=device)

if __name__ == "__main__":
    main()
