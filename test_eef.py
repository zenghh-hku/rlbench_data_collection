import numpy as np
import os
import json

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import __dict__ as task_dict
from rlbench.observation_config import CameraConfig, ObservationConfig
from rlbench.tasks import CloseBox

def load_demonstration(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

data = load_demonstration('data_train_224_allinfo/CloseBox/episode_000011/data.json')

action_eef = []
action_gripperopen = []
for i in range(data['total_steps']):
    action_eef.append(data['data'][i]['gripper_pose'])
    action_gripperopen.append(data['data'][i]['gripper_open'])

action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(),
        gripper_action_mode=Discrete()
    )

cam_config = CameraConfig(image_size=(224,224))
obs_config = ObservationConfig()
obs_config.set_all(False)
obs_config.right_shoulder_camera = cam_config
obs_config.left_shoulder_camera = cam_config
obs_config.overhead_camera = cam_config
obs_config.wrist_camera = cam_config
obs_config.front_camera = cam_config
obs_config.set_all_low_dim(True)

env = Environment(
    action_mode,
    obs_config=obs_config,
    headless = False
)
env.launch()

task = env.get_task(CloseBox)
descriptions, obs = task.reset()
print(descriptions)

for i in range(data['total_steps']):
    value = 1.0 if action_gripperopen[i]/0.04>0.9 else 0.0
    act = action_eef[i] + [value]
    obs, reward, terminate = task.step(act)
    

print('Done')
env.shutdown()