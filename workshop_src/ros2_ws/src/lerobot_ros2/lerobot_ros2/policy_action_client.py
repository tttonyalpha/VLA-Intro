import sys

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from vla_interfaces.action import RunPolicy


class PolicyActionClient(Node):
    def __init__(self):
        super().__init__("policy_action_client")
        self._client = ActionClient(self, RunPolicy, "run_policy")

    def send_goal(self, task_text: str, fps: float):
        goal_msg = RunPolicy.Goal()
        goal_msg.task_text = task_text
        goal_msg.fps = fps

        self._client.wait_for_server()
        self._send_goal_future = self._client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback,
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            rclpy.shutdown()
            return

        self.get_logger().info("Goal accepted")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(f"step={fb.step}, status={fb.status}")

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"result: success={result.success}, message='{result.message}'")
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    task = "Put the red cube into the white bucket"
    fps = 30.0

    if len(sys.argv) > 1:
        task = sys.argv[1]
    if len(sys.argv) > 2:
        fps = float(sys.argv[2])

    node = PolicyActionClient()
    node.send_goal(task, fps)
    rclpy.spin(node)


if __name__ == "__main__":
    main()