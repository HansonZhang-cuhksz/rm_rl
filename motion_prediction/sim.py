import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import random as rd
import numpy as np

def vel_func(x):
    assert x >= 0
    x = x % 6
    if x < 1:
        out = x
    if x >= 1 and x < 2:
        out = 1
    if x >= 2 and x < 3:
        out = 3 - x
    if x >= 3 and x < 4:
        out = 4 - x
    if x >= 4 and x < 5:
        out = -1
    if x >= 5 and x < 6:
        out = x - 6

    return out

def pos_func(x):
    out = 0
    x %= 6
    for i in range(int(x*1000)):
        out += vel_func(i/1000) / 1000
    return out

class Talker(Node):
    def __init__(self):
        super().__init__('talker')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'robot_position_topic', 10)
        self.timer = self.create_timer(0, self.timer_callback)  # 100 Hz
        self.counter = 0

    def timer_callback(self):
        position = pos_func(self.counter)
        velocity = vel_func(self.counter)
        self.counter += 1
        self.counter %= 6

        # Create and populate the Float32MultiArray message
        msg = Float32MultiArray()
        msg.data = [position, 0, 0, velocity, 0, 0]
        for i in range(6):
            msg.data[i] += rd.normalvariate(0, 0.1)

        # Publish the message
        self.publisher_.publish(msg)
        # self.get_logger().info(f'Publishing: {msg.data}')

if __name__ == '__main__':
    # rclpy.init()
    # talker = Talker()
    # rclpy.spin(talker)
    # talker.destroy_node()
    # rclpy.shutdown()
    print("start")
    data = np.array([])
    for i in range(1000000):
        i_copy = 0 + i
        position = pos_func(i_copy)
        velocity = vel_func(i_copy)
        msg = [position, 0, 0, velocity, 0, 0]
        for j in range(6):
            msg[j] += rd.normalvariate(0, 0.1)
        data = np.append(data, msg)
        if i%10000 == 0:
            print(i, "done")
    np.save('data.npy', data)