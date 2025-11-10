import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from simplicial_tensors.tensor_ops import bdry, degen  # Ensure tensor_ops.py is correctly implemented and accessible

# ----------------------------
# Original Network Class
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
        # Layer 1
        x = torch.relu(self.fc1(x))
        # Layer 2
        x = torch.relu(self.fc2(x))
        # Layer 3
        x = torch.relu(self.fc3(x))
        # Layer 4
        x = torch.relu(self.fc4(x))
        # Final Layer (fc5)
        x = self.fc5(x)
        return x

# ----------------------------
# Boundary-Augmented Network Class
# ----------------------------

class BoundaryAugmentedNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(BoundaryAugmentedNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0], bias=True)
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1], bias=True)
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2], bias=True)
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3], bias=True)
        self.fc5 = nn.Linear(hidden_sizes[3], output_size, bias=True)

    def forward(self, x):
        # Layer 1
        x = torch.relu(self.fc1(x))
        # Layer 2
        x = torch.relu(self.fc2(x))
        # Layer 3
        x = torch.relu(self.fc3(x))
        # Layer 4
        x = torch.relu(self.fc4(x))
        # Final Layer
        x = self.fc5(x)
        return x

    def integrate_boundary(self):
        """
        Integrate boundary tensors into the network's weights.
        This method should be called after gradients have been computed.
        """
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'weight' in name:
                    # Compute boundary tensor
                    boundary = bdry(param.data)
                    # Degenerate the boundary tensor
                    boundary_degen = degen(boundary, 0)
                    # Ensure shapes are compatible
                    assert param.data.shape == boundary_degen.shape, f"Shape mismatch for {name}: {param.data.shape} vs {boundary_degen.shape}"
                    # Update weights by adding boundary_degen
                    param.data += boundary_degen
                    # Since bdry(bdry(T))=0, no further action is needed

# ----------------------------
# Training Function
# ----------------------------

def train_network(net, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        # Generate random input data
        inputs = torch.randn(32, 64)  # Batch size of 32, input size of 64
        labels = torch.randn(32, 10)  # Random labels for demonstration

        # Forward pass
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Integrate boundary tensors after weight update
        if isinstance(net, BoundaryAugmentedNet):
            net.integrate_boundary()

        # Print loss for each epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

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
    original_optimizer = optim.Adam(original_net.parameters(), lr=0.001, weight_decay=0.01)
    train_network(original_net, criterion, original_optimizer)

    # ----------------------------
    # Training Boundary-Augmented Network
    # ----------------------------
    print("\nTraining Boundary-Augmented Network:")
    boundary_augmented_net = BoundaryAugmentedNet(input_size, hidden_sizes, output_size)
    boundary_augmented_optimizer = optim.Adam(boundary_augmented_net.parameters(), lr=0.001, weight_decay=0.01)
    train_network(boundary_augmented_net, criterion, boundary_augmented_optimizer)

if __name__ == "__main__":
    main()
