import torch
import numpy as np

input_size = 8
hidden_size = 128
output_size = 4
num_layers = 3
nhead = 8

model = torch.load('model.pth')

# Load the saved model weights
model.load_state_dict(torch.load('transformer_model.pth'))
model.eval()  # Set the model to evaluation mode

# Example function to make predictions
def predict(model, input_seq):
    with torch.no_grad():
        input_seq = input_seq.unsqueeze(0)  # Add batch dimension
        tgt = torch.zeros(1, input_seq.size(1), output_size)  # Create a target sequence
        output_seq = model(input_seq, tgt)
    return output_seq.squeeze(0)  # Remove batch dimension

# Example usage
# Load new data (replace with your actual data loading code)
new_data = np.array([[0, 0, 0, 0, 1, 1, 1, 0.1], [1, 1, 1, 0.1, 1, 1, 1, 0.1]])
new_data = torch.tensor(new_data, dtype=torch.float32)

# Make predictions
predictions = predict(model, new_data)
print(predictions)