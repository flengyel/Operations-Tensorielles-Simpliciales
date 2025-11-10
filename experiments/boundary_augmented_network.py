import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from simplicial_tensors.tensor_ops import bdry_mod1, degen  # Ensure tensor_ops.py is correctly implemented and accessible

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
        self.fc5 = nn.Linear(hidden_sizes[3], output_size)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc5(x)
        return x

class BoundaryAugmentedNet(OriginalNet):
    def __init__(self, input_size, hidden_sizes, output_size, boundary_scale=1e-5, negative_slope=0.01):
        super(BoundaryAugmentedNet, self).__init__(input_size, hidden_sizes, output_size, negative_slope)
        self.boundary_scale = boundary_scale

    def integrate_boundary(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'weight' in name and ('fc3' in name or 'fc4' in name or 'fc5' in name):  # Apply to later layers only
                    # Compute boundary tensor
                    boundary = bdry_mod1(param.detach().cpu().numpy())

                    # Ensure boundary tensor is not scalar
                    if boundary.ndim == 0:
                        raise ValueError(f"Boundary tensor for {name} is scalar, cannot determine axis for duplication.")

                    # Compute k as half of the minimum dimension
                    k = min(boundary.shape) // 2

                    # Degenerate the boundary tensor along the calculated index
                    boundary_degen = degen(boundary, k)

                    # Convert the degenerated boundary back to a PyTorch tensor
                    boundary_degen_tensor = torch.from_numpy(boundary_degen).to(param.device).type(param.dtype)

                    # Ensure shapes are compatible
                    if param.data.shape != boundary_degen_tensor.shape:
                        raise ValueError(f"Shape mismatch for {name}: {param.data.shape} vs {boundary_degen_tensor.shape}")

                    # Update weights by adding scaled boundary_degen
                    param.data += self.boundary_scale * boundary_degen_tensor

class RandomTensorAugmentedNet(OriginalNet):
    def __init__(self, input_size, hidden_sizes, output_size, random_tensor_scale=7e-3, negative_slope=0.01):
        super(RandomTensorAugmentedNet, self).__init__(input_size, hidden_sizes, output_size, negative_slope)
        self.random_tensor_scale = random_tensor_scale

    def integrate_random_tensor(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'weight' in name and ('fc3' in name or 'fc4' in name or 'fc5' in name):  # Apply to later layers only
                    # Get the shape of the original tensor
                    original_shape = param.detach().cpu().numpy().shape
                    
                    # Compute the boundary shape by subtracting 1 from each dimension
                    boundary_shape = tuple(dim - 1 for dim in original_shape)
                    
                    # Generate a random tensor of boundary shape
                    rng = np.random.default_rng(seed=42)
                    random_tensor = rng.standard_normal(boundary_shape)
                    
                    # Degenerate the random tensor along the middle axes (like boundary degeneration)
                    k = min(boundary_shape) // 2
                    random_degen_tensor = degen(random_tensor, k)

                    # Convert the degenerated random tensor to a PyTorch tensor and apply it to the weights
                    random_degen_tensor = torch.from_numpy(random_degen_tensor).to(param.device).type(param.dtype)
                    
                    # Scale and add to the weights
                    param.data += self.random_tensor_scale * random_degen_tensor

# ----------------------------
# Unified Training Function
# ----------------------------

def train_network(net, criterion, optimizer, scheduler, num_epochs=40, augment_type=None):
    for epoch in range(num_epochs):
        # Generate random input data
        inputs = torch.randn(32, 64)  # Batch size of 32, input size of 64
        labels = torch.randn(32, 10)  # Random labels for demonstration

        # Forward pass
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # Integrate augmentation if applicable
        if augment_type == 'boundary' and isinstance(net, BoundaryAugmentedNet):
            try:
                net.integrate_boundary()
            except ValueError as e:
                print(f"Boundary integration failed: {e}")
        elif augment_type == 'random' and isinstance(net, RandomTensorAugmentedNet):
            net.integrate_random_tensor()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

        optimizer.step()

        # Step the scheduler
        scheduler.step()

        # Print loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

# ----------------------------
# Main Experiment
# ----------------------------

def main():
    # Define network parameters
    input_size = 64
    hidden_sizes = [256, 128, 64, 32]
    output_size = 10

    # Loss function
    criterion = nn.MSELoss()

    # ----------------------------
    # Training Original Network
    # ----------------------------
    print("Training Original Network with Leaky ReLU:")
    original_net = OriginalNet(input_size, hidden_sizes, output_size)
    original_optimizer = optim.Adam(original_net.parameters(), lr=0.001, weight_decay=0.01)
    original_scheduler = optim.lr_scheduler.StepLR(original_optimizer, step_size=20, gamma=0.1)
    train_network(original_net, criterion, original_optimizer, original_scheduler, num_epochs=40)

    # ----------------------------
    # Training Boundary-Augmented Network
    # ----------------------------
    print("\nTraining Boundary-Augmented Network with Leaky ReLU:")
    boundary_augmented_net = BoundaryAugmentedNet(input_size, hidden_sizes, output_size, boundary_scale=1e-5)
    boundary_augmented_optimizer = optim.Adam(boundary_augmented_net.parameters(), lr=0.001, weight_decay=0.01)
    boundary_augmented_scheduler = optim.lr_scheduler.StepLR(boundary_augmented_optimizer, step_size=20, gamma=0.1)
    train_network(boundary_augmented_net, criterion, boundary_augmented_optimizer, boundary_augmented_scheduler, num_epochs=40, augment_type='boundary')

    # ----------------------------
    # Training Random Tensor Augmented Network
    # ----------------------------
    print("\nTraining Random Tensor Augmented Network with Leaky ReLU:")
    random_tensor_augmented_net = RandomTensorAugmentedNet(input_size, hidden_sizes, output_size, random_tensor_scale=7e-3)
    random_tensor_augmented_optimizer = optim.Adam(random_tensor_augmented_net.parameters(), lr=0.001, weight_decay=0.01)
    random_tensor_augmented_scheduler = optim.lr_scheduler.StepLR(random_tensor_augmented_optimizer, step_size=20, gamma=0.1)
    train_network(random_tensor_augmented_net, criterion, random_tensor_augmented_optimizer, random_tensor_augmented_scheduler, num_epochs=40, augment_type='random')

if __name__ == "__main__":
    main()
