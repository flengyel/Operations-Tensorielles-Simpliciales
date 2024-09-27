import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensor_ops import bdry, degen  # Ensure tensor_ops.py is correctly implemented and accessible

# ----------------------------
# Helper Functions
# ----------------------------

def _dims(m):
    """
    Helper function to get the axes of the tensor.
    """
    return list(range(len(m.shape)))

def _face(m, axes, i):
    """
    Extract the first slice along the specified axis.

    Parameters:
    - m (np.ndarray): Input array.
    - axes (list): List of axes.
    - i (int): Current dimension index.

    Returns:
    - np.ndarray: Extracted face.
    """
    index = [slice(None)] * len(m.shape)
    index[axes[i]] = 0  # Modify this as per your boundary logic
    return m[tuple(index)]

# ----------------------------
# Network Classes
# ----------------------------

class OriginalNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(OriginalNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class BoundaryAugmentedNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, boundary_scale=7e-3):
        super(BoundaryAugmentedNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0], bias=True)
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1], bias=True)
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2], bias=True)
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3], bias=True)
        self.fc5 = nn.Linear(hidden_sizes[3], output_size, bias=True)
        self.boundary_scale = boundary_scale  # Scaling factor to control boundary influence

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def integrate_boundary(self):
        """
        Integrate boundary tensors into the network's weights after each optimization step.
        """
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'weight' in name:
                    # Compute boundary tensor
                    boundary = bdry(param)
                    
                    # Degenerate the boundary tensor
                    boundary_degen = degen(boundary, 0)
                    
                    # Ensure shapes are compatible
                    if param.data.shape != boundary_degen.shape:
                        raise ValueError(f"Shape mismatch for {name}: {param.data.shape} vs {boundary_degen.shape}")
                    
                    # Update weights by adding scaled boundary_degen
                    param.data += self.boundary_scale * boundary_degen
                    
                    
# ----------------------------
# Training and Verification Functions
# ----------------------------

def train_network_tensor_reg(net, criterion, optimizer, scheduler, num_epochs=40, lambda_reg=1e-4):
    for epoch in range(num_epochs):
        # Generate random input data
        inputs = torch.randn(32, 64)  # Batch size of 32, input size of 64
        labels = torch.randn(32, 10)  # Random labels for demonstration

        # Forward pass
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # Boundary regularization (tensor-wise)
        reg_loss = 0.0
        for name, param in net.named_parameters():
            if 'weight' in name:
                try:
                    # Detach the tensor to prevent gradient tracking
                    param_detached = param.detach()
                    
                    # Convert the detached tensor to NumPy
                    param_np = param_detached.cpu().numpy()
                    
                    # Pass the NumPy array to bdry()
                    boundary = bdry(param_np)
                    
                    # Perform degeneration using degen()
                    boundary_degen = degen(boundary, 0)
                    
                    # Convert the degenerate boundary back to a PyTorch tensor
                    # Ensure it's on the same device and dtype as the original parameter
                    boundary_degen_tensor = torch.from_numpy(boundary_degen).to(param.device).type(param.dtype)
                    
                    # Accumulate the Frobenius norm for regularization
                    reg_loss += torch.norm(boundary_degen_tensor, p='fro')
                except ValueError as e:
                    print(f"Boundary condition check failed for {name}: {e}")
                    reg_loss += 0.0  # Optionally handle the exception as needed

        # Total loss
        total_loss = loss + lambda_reg * reg_loss

        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

        optimizer.step()

        # Boundary integration if applicable
        if isinstance(net, BoundaryAugmentedNet):
            try:
                net.integrate_boundary()
            except ValueError as e:
                print(f"Boundary integration failed: {e}")

        # Step the scheduler
        scheduler.step()

        # Print loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.6f}")

def verify_boundary_conditions(net):
    for name, param in net.named_parameters():
        if 'weight' in name:
            # Detach and convert to NumPy
            param_np = param.detach().cpu().numpy()
            
            # Compute boundary
            boundary = bdry(param_np)
            
                        # Compute second boundary
            second_boundary = bdry(boundary)
            
            # Check if second boundary is approximately zero
            if not np.allclose(second_boundary, np.zeros_like(second_boundary), atol=1e-6):
                print(f"Boundary condition violated for {name}")
            else:
                print(f"Boundary condition holds for {name}")

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
    print("Training Original Network:")
    original_net = OriginalNet(input_size, hidden_sizes, output_size)
    original_optimizer = optim.Adam(original_net.parameters(), lr=0.001)
    original_scheduler = optim.lr_scheduler.StepLR(original_optimizer, step_size=20, gamma=0.1)
    train_network_tensor_reg(original_net, criterion, original_optimizer, original_scheduler, num_epochs=40)

    # ----------------------------
    # Training Boundary-Augmented Network
    # ----------------------------
    print("\nTraining Boundary-Augmented Network:")
    boundary_augmented_net = BoundaryAugmentedNet(input_size, hidden_sizes, output_size, boundary_scale=7e-3)
    boundary_augmented_optimizer = optim.Adam(boundary_augmented_net.parameters(), lr=0.001)
    boundary_augmented_scheduler = optim.lr_scheduler.StepLR(boundary_augmented_optimizer, step_size=20, gamma=0.1)
    train_network_tensor_reg(boundary_augmented_net, criterion, boundary_augmented_optimizer, boundary_augmented_scheduler, num_epochs=40)

    # ----------------------------
    # Verify Boundary Conditions
    # ----------------------------
    print("\nVerifying Boundary Conditions for Boundary-Augmented Network:")
    verify_boundary_conditions(boundary_augmented_net)

if __name__ == "__main__":
    main()
