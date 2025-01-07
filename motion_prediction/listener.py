import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class Listener(Node):
    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'robot_position_topic',
            self.callback,
            10)
        self.subscription  # prevent unused variable warning

    def callback(self, msg):
        self.get_logger().info(f'Received: {msg.data}')

if __name__ == '__main__':
    rclpy.init()
    listener = Listener()
    rclpy.spin(listener)
    listener.destroy_node()
    rclpy.shutdown()