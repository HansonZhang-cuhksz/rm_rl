import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import Float32MultiArray

# 创建训练数据和标签
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# 在线学习函数
def online_learning(model, optimizer, loss_function, new_data, time_step, scaler):
    new_data_scaled = scaler.transform(new_data)
    X_new, y_new = create_dataset(new_data_scaled, time_step)
    X_new = X_new.reshape(X_new.shape[0], X_new.shape[1], 1)
    X_new = torch.tensor(X_new, dtype=torch.float32)
    y_new = torch.tensor(y_new, dtype=torch.float32)
    dataset_new = TensorDataset(X_new, y_new)
    dataloader_new = DataLoader(dataset_new, batch_size=1, shuffle=True)

    model.train()
    for seq, labels in dataloader_new:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    return model

# class DataListener(Node):
#     def __init__(self):
#         super().__init__('data_listener')
#         self.subscription = self.create_subscription(
#             Float32MultiArray,
#             'robot_position_topic',
#             self.listener_callback,
#             10)
#         self.data = []
#         self.data_count = 0
#         self.max_data_count = 1000000  # 设置接收数据的最大数量

#     def listener_callback(self, msg):
#         self.data.append(msg.data)
#         self.data_count += 1
#         # self.get_logger().info(f'Received: {msg.data}')
#         if self.data_count >= self.max_data_count:
#             self.get_logger().info('Max data count reached, shutting down...')
#             rclpy.shutdown()

def main(args=None):
    # rclpy.init(args=args)
    # data_listener = DataListener()
    # rclpy.spin(data_listener)

    # data = np.array(data_listener.data)
    # np.save('data.npy', data)
    # print("Saved")
    data = np.load("data.npy")
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    time_step = 10
    X, y = create_dataset(data_scaled, time_step)

    X = X.reshape(X.shape[0], X.shape[1], 1)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = LSTMModel()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 100  # 设置训练轮数
    for epoch in range(epochs):
        model.train()
        for seq, labels in dataloader:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{epochs} complete')

    torch.save(model.state_dict(), 'robot_position_predictor_online.pth')

    model = LSTMModel()
    model.load_state_dict(torch.load('robot_position_predictor_online.pth'))

    test_data = data_scaled[-time_step:]
    test_data = torch.tensor(test_data, dtype=torch.float32).view(1, time_step, 1)
    model.eval()
    with torch.no_grad():
        predicted_position = model(test_data).item()
        predicted_position = scaler.inverse_transform(np.array([[predicted_position]]))

    print("Predicted Position:", predicted_position)

    # data_listener.destroy_node()
    # rclpy.shutdown()

if __name__ == '__main__':
    main()