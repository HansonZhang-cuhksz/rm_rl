import torch
from model import MotionPredictionModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('model.pth', map_location=device)

def predict(model, input_seq):
    model.eval()
    with torch.no_grad():
        input_seq = input_seq.to(device)
        output_seq = model(input_seq)
    return output_seq.cpu().numpy()

# Example usage
# Assuming input_seq is a tensor of shape (batch_size, sequence_length, input_size)
predicted_future_states = predict(model, input_seq)