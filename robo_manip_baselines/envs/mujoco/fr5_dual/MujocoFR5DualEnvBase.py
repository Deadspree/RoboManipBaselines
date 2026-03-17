from os import path

import mujoco
import numpy as np
from gymnasium.spaces import Box, Dict

from robo_manip_baselines.common import ArmConfig
from robo_manip_baselines.teleop import (
    GelloInputDevice,
    KeyboardInputDevice,
    SpacemouseInputDevice,
)

from ..MujocoEnvBase import MujocoEnvBase


class MujocoFR5DualEnvBase(MujocoEnvBase):
    default_camera_config = {
        "azimuth": -135.0,
        "elevation": -45.0,
        "distance": 1.8,
        "lookat": [-0.2, -0.2, 0.8],
    }
    observation_space = Dict(
        {
            "joint_pos": Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64),
            "joint_vel": Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64),
            "wrench": Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64),
        }
    )

    def setup_robot(self, init_qpos):
        self.init_qpos[: len(init_qpos)] = init_qpos
        self.init_qvel[:] = 0.0

        mujoco.mj_kinematics(self.model, self.data)

        self.body_config_list = [
            ArmConfig(
                arm_urdf_path=path.join(
                    path.dirname(__file__), "../../assets/common/robots/fairino5_v6/fairino5_v6.urdf"
                ),
                arm_root_pose=self.get_body_pose("left/fairino5_v6_root_frame"),
                ik_eef_joint_id=6,
                arm_joint_idxes=np.arange(0, 6),
                gripper_joint_idxes=np.array([], dtype=np.int_),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([], dtype=np.int_),
                eef_idx=0,
                init_arm_joint_pos=self.init_qpos[0:6],
                init_gripper_joint_pos=np.zeros(0, dtype=np.float64),
            ),
            ArmConfig(
                arm_urdf_path=path.join(
                    path.dirname(__file__), "../../assets/common/robots/fairino5_v6/fairino5_v6.urdf"
                ),
                arm_root_pose=self.get_body_pose("right/fairino5_v6_root_frame"),
                ik_eef_joint_id=6,
                arm_joint_idxes=np.arange(6, 12),
                gripper_joint_idxes=np.array([], dtype=np.int_),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([], dtype=np.int_),
                eef_idx=1,
                init_arm_joint_pos=self.init_qpos[6:12],
                init_gripper_joint_pos=np.zeros(0, dtype=np.float64),
            ),
        ]

    def setup_input_device(self, input_device_name, motion_manager, overwrite_kwargs):
        if input_device_name == "spacemouse":
            InputDeviceClass = SpacemouseInputDevice
        elif input_device_name == "gello":
            InputDeviceClass = GelloInputDevice
        elif input_device_name == "keyboard":
            InputDeviceClass = KeyboardInputDevice
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid input device key: {input_device_name}"
            )

        default_kwargs = self.get_input_device_kwargs(input_device_name)

        return [
            InputDeviceClass(
                body_manager,
                **{
                    **default_kwargs.get(device_idx, {}),
                    **overwrite_kwargs.get(device_idx, {}),
                },
            )
            for device_idx, body_manager in enumerate(motion_manager.body_manager_list)
        ]

    def get_input_device_kwargs(self, input_device_name):
        return {}

    def _get_obs(self):
        left_obs = self._get_obs_single_arm("left")
        right_obs = self._get_obs_single_arm("right")
        return {
            key: np.concatenate([left_obs[key], right_obs[key]])
            for key in left_obs.keys()
        }

    def _get_obs_single_arm(self, left_right):
        arm_joint_name_list = ["j1", "j2", "j3", "j4", "j5", "j6"]

        arm_joint_pos = np.array(
            [self.data.joint(left_right + "/" + jn).qpos[0] for jn in arm_joint_name_list]
        )
        arm_joint_vel = np.array(
            [self.data.joint(left_right + "/" + jn).qvel[0] for jn in arm_joint_name_list]
        )

        # If F/T sensor is not mounted yet, keep wrench as zeros.
        try:
            force = self.data.sensor(left_right + "/" + "force_sensor").data.flat.copy()
            torque = self.data.sensor(left_right + "/" + "torque_sensor").data.flat.copy()
            wrench = np.concatenate((force, torque), dtype=np.float64)
        except KeyError:
            wrench = np.zeros(6, dtype=np.float64)

        return {
            "joint_pos": arm_joint_pos.astype(np.float64),
            "joint_vel": arm_joint_vel.astype(np.float64),
            "wrench": wrench,
        }