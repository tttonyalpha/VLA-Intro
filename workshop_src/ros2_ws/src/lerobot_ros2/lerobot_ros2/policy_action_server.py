import time
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse

from vla_interfaces.action import RunPolicy

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.configs import Cv2Rotation
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device


HF_MODEL_ID = "MrAnton/cube_into_buck_policy"
DEFAULT_FPS = 30.0

JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


class PolicyActionServer(Node):
    def __init__(self):
        super().__init__("policy_action_server")

        self.declare_parameter("serial_number", "231122072775")
        self.declare_parameter("port", "/dev/ttyACM2")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("default_task", "Put the red cube into the white bucket")

        serial_number = self.get_parameter("serial_number").value
        port = self.get_parameter("port").value
        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        self.default_task = self.get_parameter("default_task").value

        robot_config = SO101FollowerConfig(
            id="my_awesome_follower_arm",
            cameras={
                "front": RealSenseCameraConfig(
                    serial_number_or_name=serial_number,
                    fps=int(DEFAULT_FPS),
                    width=width,
                    height=height,
                    rotation=Cv2Rotation.ROTATE_180,
                )
            },
            port=port,
        )

        self.robot = SO101Follower(robot_config)
        self.policy = ACTPolicy.from_pretrained(HF_MODEL_ID)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy,
            pretrained_path=HF_MODEL_ID,
        )
        self.device = get_safe_torch_device(self.policy.config.device)

        self.robot.connect()
        self.get_logger().info("Robot connected and policy loaded.")

        self._action_server = ActionServer(
            self,
            RunPolicy,
            "run_policy",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

    def destroy_node(self):
        try:
            self.robot.disconnect()
        except Exception:
            pass
        super().destroy_node()

    def goal_callback(self, goal_request):
        self.get_logger().info("Received goal request.")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info("Received cancel request.")
        return CancelResponse.ACCEPT

    def build_policy_observation(self, obs_raw: dict) -> dict:
        img = obs_raw["front"]
        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

        state = np.array([obs_raw[k] for k in JOINT_KEYS], dtype=np.float32)

        return {
            "observation.images.front": img,
            "observation.state": state,
        }

    def action_to_robot_dict(self, action_values) -> dict:
        if isinstance(action_values, torch.Tensor):
            action_values = action_values.detach().cpu().numpy()

        action_values = np.asarray(action_values, dtype=np.float32).squeeze()

        if action_values.shape[0] != len(JOINT_KEYS):
            raise RuntimeError(
                f"Expected {len(JOINT_KEYS)} action values, got shape {action_values.shape}"
            )

        return {k: float(v) for k, v in zip(JOINT_KEYS, action_values)}

    def execute_callback(self, goal_handle):
        task_text = goal_handle.request.task_text.strip() or self.default_task
        fps = float(goal_handle.request.fps) if goal_handle.request.fps > 0.0 else DEFAULT_FPS

        self.get_logger().info(f"Starting policy loop. task='{task_text}', fps={fps}")

        feedback_msg = RunPolicy.Feedback()
        step = 0

        try:
            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result = RunPolicy.Result()
                    result.success = False
                    result.message = "Canceled"
                    return result

                t0 = time.perf_counter()

                obs_raw = self.robot.get_observation()
                policy_obs = self.build_policy_observation(obs_raw)

                with torch.inference_mode():
                    action_values = predict_action(
                        observation=policy_obs,
                        policy=self.policy,
                        device=self.device,
                        preprocessor=self.preprocessor,
                        postprocessor=self.postprocessor,
                        use_amp=self.policy.config.use_amp,
                        task=task_text,
                        robot_type=self.robot.robot_type,
                    )

                robot_action = self.action_to_robot_dict(action_values)

                # мягкое смешивание, чтобы не было резких скачков
                current_state = np.array([obs_raw[k] for k in JOINT_KEYS], dtype=np.float32)
                target_state = np.array([robot_action[k] for k in JOINT_KEYS], dtype=np.float32)

                alpha = 0.15
                blended = (1.0 - alpha) * current_state + alpha * target_state
                safe_action = {k: float(v) for k, v in zip(JOINT_KEYS, blended)}

                self.robot.send_action(safe_action)

                step += 1
                feedback_msg.step = step
                feedback_msg.status = f"running: {task_text}"
                goal_handle.publish_feedback(feedback_msg)

                dt = time.perf_counter() - t0
                time.sleep(max(1.0 / fps - dt, 0.0))

        except Exception as e:
            self.get_logger().error(f"Policy loop failed: {e}")
            goal_handle.abort()
            result = RunPolicy.Result()
            result.success = False
            result.message = f"Exception: {e}"
            return result


def main(args=None):
    rclpy.init(args=args)
    node = PolicyActionServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()