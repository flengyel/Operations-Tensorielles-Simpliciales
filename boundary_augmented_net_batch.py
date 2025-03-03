import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tensor_ops import bdry, degen  # Ensure tensor_ops.py is correctly implemented and accessible
import matplotlib.pyplot as plt

# ----------------------------
# Network Classes
# ----------------------------

class OriginalNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, negative_slope=0.01):
        super(OriginalNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(hidden_sizes[3], hidden_sizes[4])
        self.dropout5 = nn.Dropout(p=0.5)
        self.fc6 = nn.Linear(hidden_sizes[4], output_size)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.activation(self.fc1(x))
        x = self.dropout1(x)
        x = self.activation(self.fc2(x))
        x = self.dropout2(x)
        x = self.activation(self.fc3(x))
        x = self.dropout3(x)
        x = self.activation(self.fc4(x))
        x = self.dropout4(x)
        x = self.activation(self.fc5(x))
        x = self.dropout5(x)
        x = self.fc6(x)
        return x

FC5_WEIGHT = 'fc5.weight'

class BoundaryAugmentedNet(OriginalNet):
    def __init__(self, input_size, hidden_sizes, output_size, boundary_scale=5e-4, operation='multiply', layers_to_augment=None):
        """
        Initialize the BoundaryAugmentedNet.

        Args:
            input_size (int): Size of the input layer.
            hidden_sizes (list of int): Sizes of the hidden layers.
            output_size (int): Size of the output layer.
            boundary_scale (float): Scaling factor for boundary augmentation.
            operation (str): 'add' for addition or 'multiply' for Hadamard product.
            layers_to_augment (list of str): List of layer names to augment (e.g., [FC5_WEIGHT]).
        """
        super(BoundaryAugmentedNet, self).__init__(input_size, hidden_sizes, output_size)
        self.boundary_scale = boundary_scale
        self.operation = operation  # 'add' or 'multiply'
        if layers_to_augment is None:
            self.layers_to_augment = [FC5_WEIGHT]  # Default to fc5
        else:
            self.layers_to_augment = layers_to_augment

    def integrate_boundary(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in self.layers_to_augment:
                    boundary = bdry(param.detach().cpu().numpy())
                    k = min(boundary.shape) // 2
                    boundary_degen = degen(boundary, k)
                    boundary_degen_tensor = torch.from_numpy(boundary_degen).to(param.device).type(param.dtype)

                    # Normalize the boundary tensor to avoid large updates
                    norm = torch.norm(boundary_degen_tensor, dim=0)
                    if norm != 0:  # Prevent division by zero
                        boundary_degen_tensor = boundary_degen_tensor / norm

                    if param.data.shape != boundary_degen_tensor.shape:
                        raise ValueError(f"Shape mismatch for {name}: {param.data.shape} vs {boundary_degen_tensor.shape}")

                    # Perform the chosen operation
                    if self.operation == 'add':
                        param.data += self.boundary_scale * boundary_degen_tensor
                    elif self.operation == 'multiply':
                        param.data *= (1 + self.boundary_scale * boundary_degen_tensor)
                    else:
                        raise ValueError(f"Unsupported operation '{self.operation}'. Use 'add' or 'multiply'.")

class RandomTensorAugmentedNet(OriginalNet):
    def __init__(self, input_size, hidden_sizes, output_size, random_tensor_scale=7e-3, operation='add', layers_to_augment=None):
        """
        Initialize the RandomTensorAugmentedNet.

        Args:
            input_size (int): Size of the input layer.
            hidden_sizes (list of int): Sizes of the hidden layers.
            output_size (int): Size of the output layer.
            random_tensor_scale (float): Scaling factor for random tensor augmentation.
            operation (str): 'add' for addition or 'multiply' for Hadamard product.
            layers_to_augment (list of str): List of layer names to augment (e.g., [FC5_WEIGHT]).
        """
        super(RandomTensorAugmentedNet, self).__init__(input_size, hidden_sizes, output_size)
        self.random_tensor_scale = random_tensor_scale
        self.operation = operation  # 'add' or 'multiply'
        if layers_to_augment is None:
            self.layers_to_augment = [FC5_WEIGHT]  # Default to fc5
        else:
            self.layers_to_augment = layers_to_augment

    def integrate_random_tensor(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in self.layers_to_augment:
                    original_shape = param.detach().cpu().numpy().shape
                    boundary_shape = tuple(dim - 1 for dim in original_shape)
                    rng = np.random.default_rng(seed=42)
                    random_tensor = rng.standard_normal(boundary_shape)
                    k = min(boundary_shape) // 2
                    random_degen_tensor = degen(random_tensor, k)
                    random_degen_tensor = torch.from_numpy(random_degen_tensor).to(param.device).type(param.dtype)

                    # Normalize the random tensor to avoid large updates
                    norm = torch.norm(random_degen_tensor, dim=0)
                    if norm != 0:  # Prevent division by zero
                        random_degen_tensor = random_degen_tensor / norm

                    if param.data.shape != random_degen_tensor.shape:
                        raise ValueError(f"Shape mismatch for {name}: {param.data.shape} vs {random_degen_tensor.shape}")

                    # Perform the chosen operation
                    if self.operation == 'add':
                        param.data += self.random_tensor_scale * random_degen_tensor
                    elif self.operation == 'multiply':
                        param.data *= (1 + self.random_tensor_scale * random_degen_tensor)
                    else:
                        raise ValueError(f"Unsupported operation '{self.operation}'. Use 'add' or 'multiply'.")

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

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader

# ----------------------------
# Weight Monitoring Function
# ----------------------------

def monitor_weights(net):
    for name, param in net.named_parameters():
        if param.requires_grad:
            max_val = param.data.max().item()
            min_val = param.data.min().item()
            print(f"{name}: min={min_val}, max={max_val}")

# ----------------------------
# Unified Training and Evaluation Function
# ----------------------------

def train_and_evaluate_network(net, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=40, augment_type=None, device='cpu'):
    net.to(device)  # Ensure the network is on the correct device
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        net.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Apply augmentation once per epoch
        if augment_type == 'boundary' and isinstance(net, BoundaryAugmentedNet):
            net.integrate_boundary()
            monitor_weights(net)  # Optional: monitor weights after augmentation
        elif augment_type == 'random' and isinstance(net, RandomTensorAugmentedNet):
            net.integrate_random_tensor()
            monitor_weights(net)  # Optional: monitor weights after augmentation

        scheduler.step()
        train_accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        # Validation phase
        net.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for val_inputs, val_labels in test_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = net(val_inputs)
                v_loss = criterion(val_outputs, val_labels)
                val_loss += v_loss.item()
                
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(test_loader)
        
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {avg_loss:.6f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {avg_val_loss:.6f}, Val Acc: {val_accuracy:.2f}%")
    
    return history

# ----------------------------
# Evaluation Function
# ----------------------------

def evaluate_model(net, test_loader, criterion, device='cpu'):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f"Test Loss: {avg_loss:.6f}, Test Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

# ----------------------------
# Plotting Function
# ----------------------------

def plot_learning_curves(history, num_epochs=40):
    epochs = range(1, num_epochs + 1)
    train_losses = history['train_loss']
    val_losses = history['val_loss']
    train_acc = history['train_acc']
    val_acc = history['val_acc']

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'g', label='Training loss')
    plt.plot(epochs, val_losses, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'g', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.show()

# ----------------------------
# Main Experiment with Adjustments
# ----------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select device dynamically

    # Define network parameters
    input_size = 28 * 28  # MNIST images are 28x28
    hidden_sizes = [512, 256, 128, 64, 32]  # Larger fully connected network
    output_size = 10  # MNIST has 10 classes (digits 0-9)

    # Load MNIST data
    train_loader, test_loader = load_data(batch_size=64)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # ----------------------------
    print("\nTraining Boundary-Augmented Network with ReLU (Hadamard multiplication on multiple layers):")
    print("\nTraining Boundary-Augmented Network with ReLU (Hadamard multiplication on multiple layers):")
    boundary_layers = [FC5_WEIGHT]  # Start with augmenting only fc5
    boundary_augmented_net = BoundaryAugmentedNet(
        input_size,
        hidden_sizes,
        output_size,
        boundary_scale=5e-4,  # Reduced scaling factor
        operation='multiply',
        layers_to_augment=boundary_layers
    ).to(device)
    boundary_augmented_optimizer = optim.Adam(
        boundary_augmented_net.parameters(),
        lr=0.001,
        weight_decay=1e-5  # Added weight decay
    )
    boundary_augmented_scheduler = optim.lr_scheduler.StepLR(boundary_augmented_optimizer, step_size=20, gamma=0.1)
    boundary_history = train_and_evaluate_network(
        boundary_augmented_net,
        train_loader,
        test_loader,
        criterion,
        boundary_augmented_optimizer,
        boundary_augmented_scheduler,
        num_epochs=40,
        augment_type='boundary',
        device=device
    )

    # ----------------------------
    # Evaluate Boundary-Augmented Network on Test Data
    # ----------------------------
    print("\nEvaluating Boundary-Augmented Network on Test Data:")
    evaluate_model(boundary_augmented_net, test_loader, criterion, device=device)

    # ----------------------------
    # Plot Learning Curves
    # ----------------------------
    plot_learning_curves(boundary_history, num_epochs=40)

    # ----------------------------
    # Training Random Tensor Augmented Network with Addition on Multiple Layers (if desired)
    # ----------------------------
    # Uncomment below to train the Random Tensor Augmented Network
    """
    print("\nTraining Random Tensor Augmented Network with ReLU (Addition on multiple layers):")
    random_layers = ['fc2.weight', 'fc4.weight']  # Specify layers to augment
    random_tensor_augmented_net = RandomTensorAugmentedNet(
        input_size,
        hidden_sizes,
        output_size,
        random_tensor_scale=7e-3,
        operation='add',
        layers_to_augment=random_layers
    ).to(device)
    random_tensor_augmented_optimizer = optim.Adam(
        random_tensor_augmented_net.parameters(),
        lr=0.001,
        weight_decay=1e-5  # Added weight decay
    )
    random_tensor_augmented_scheduler = optim.lr_scheduler.StepLR(random_tensor_augmented_optimizer, step_size=20, gamma=0.1)
    random_history = train_and_evaluate_network(
        random_tensor_augmented_net,
        train_loader,
        test_loader,
        criterion,
        random_tensor_augmented_optimizer,
        random_tensor_augmented_scheduler,
        num_epochs=40,
        augment_type='random',
        device=device
    )

    # ----------------------------
    # Evaluate Random Tensor Augmented Network on Test Data
    # ----------------------------
    print("\nEvaluating Random Tensor Augmented Network on Test Data:")
    evaluate_model(random_tensor_augmented_net, test_loader, criterion, device=device)

    # ----------------------------
    # Plot Learning Curves
    # ----------------------------
    plot_learning_curves(random_history, num_epochs=40)
    """

if __name__ == "__main__":
    main()
