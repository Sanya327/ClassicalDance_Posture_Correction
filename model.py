import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PoseLSTM(nn.Module):
    def __init__(self, input_size=99, hidden_size=128, num_layers=2):
        super(PoseLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output: score

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # last timestep
        return out


# Define the training function
def train(model, train_loader, criterion, optimizer, epochs=10):
    print("Training started...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)
            loss.backward()  # Backward pass
            optimizer.step()  # Optimizer step

            running_loss += loss.item()
            
            # Optional: Print loss every 100 iterations
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / (i+1):.4f}")

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

    print("Training finished!")

# Example to load data and train the model
def main():
    # Dummy data, replace with your actual training data
    # Your input data should be of shape (batch_size, sequence_length, input_size)
    # Labels should be a single score for each sequence
    inputs = np.random.randn(1000, 50, 99)  # 1000 samples, 50 timesteps, 99 features per timestep
    labels = np.random.randn(1000, 1)  # 1000 samples, 1 target label per sample

    inputs = torch.Tensor(inputs)  # Convert to torch tensors
    labels = torch.Tensor(labels)

    # Create a DataLoader (replace with your actual DataLoader)
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(inputs, labels)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = PoseLSTM(input_size=99, hidden_size=128, num_layers=2)
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train(model, train_loader, criterion, optimizer, epochs=10)

    # Save the trained model
    torch.save(model.state_dict(), 'pose_lstm_model.pth')
    print("✅ Model saved to 'pose_lstm_model.pth'")

if __name__ == "__main__":
    main()
