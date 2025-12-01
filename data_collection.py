import numpy as np
import os
import json
from PIL import Image

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import __dict__ as task_dict
from rlbench.observation_config import CameraConfig, ObservationConfig

from openpi_client import image_tools

def read_task_list(file_path):
    with open(file_path, 'r') as f:
        tasks = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return tasks


def save_data_per_task(env, task_name, base_output_dir):
    task = env.get_task(task_dict[task_name])
    print(task_name)
    
    episode_num = 100
    
    for episode_idx in range(episode_num):
        episode_output_dir = os.path.join(base_output_dir, f"episode_{episode_idx:06d}")
        os.makedirs(episode_output_dir, exist_ok=True)
        
        descriptions, _ = task.reset()
        instruction = descriptions if descriptions else task_name
        # print(descriptions)
        
        demos = task.get_demos(amount=1, live_demos=True)
        demonstration = demos[0]
        
        data = {
            'task': instruction,
            'episode_index': episode_idx,
            'total_steps': len(demonstration),
            'data': []
        }
        
        for step_idx, observation in enumerate(demonstration):
            front_image = observation.front_rgb
            front_image_path = os.path.join(episode_output_dir, f"episode_{episode_idx:06d}_step{step_idx:06d}_main.jpg")
            Image.fromarray(front_image).save(front_image_path)
            
            # front_resize = image_tools.convert_to_uint8(
            #         image_tools.resize_with_pad(front_image, 224, 224)
            #     )
            # front_resize_path = os.path.join(episode_output_dir, f"episode_{episode_idx:06d}_step{step_idx:06d}_main_resize.jpg")
            # Image.fromarray(front_resize).save(front_resize_path)
            
            wrist_image = observation.wrist_rgb
            wrist_image_path = os.path.join(episode_output_dir, f"episode_{episode_idx:06d}_step{step_idx:06d}_wrist.jpg")
            Image.fromarray(wrist_image).save(wrist_image_path)
            
            # step_data = {
            #     'step_index': step_idx,
            #     'joint_velocities': observation.joint_velocities.tolist(),
            #     'joint_positions': observation.joint_positions.tolist(),
            #     'gripper_position': float(observation.gripper_open)
            # }
            
            step_data = {
                'step_index': step_idx,
                # 'left_shoulder_rgb': observation.left_shoulder_rgb.tolist(),
                # 'left_shoulder_depth': observation.left_shoulder_depth.tolist(),
                # 'left_shoulder_mask': observation.left_shoulder_mask.tolist(),
                # 'left_shoulder_point_cloud': observation.left_shoulder_point_cloud.tolist(),
                # 'right_shoulder_rgb': observation.right_shoulder_rgb.tolist(),
                # 'right_shoulder_depth': observation.right_shoulder_depth.tolist(),
                # 'right_shoulder_mask': observation.right_shoulder_mask.tolist(),
                # 'right_shoulder_point_cloud': observation.right_shoulder_point_cloud.tolist(),
                # 'overhead_rgb': observation.overhead_rgb.tolist(),
                # 'overhead_depth': observation.overhead_depth.tolist(),
                # 'overhead_mask': observation.overhead_mask.tolist(),
                # 'overhead_point_cloud': observation.overhead_point_cloud.tolist(),
                # 'wrist_rgb': observation.wrist_rgb.tolist(),
                # 'wrist_depth': observation.wrist_depth.tolist(),
                # 'wrist_mask': observation.wrist_mask.tolist(),
                # 'wrist_point_cloud': observation.wrist_point_cloud.tolist(),
                # 'front_rgb': observation.front_rgb.tolist(),
                # 'front_depth': observation.front_depth.tolist(),
                # 'front_mask': observation.front_mask.tolist(),
                # 'front_point_cloud': observation.front_point_cloud.tolist(),
                'joint_velocities': observation.joint_velocities.tolist(),
                'joint_positions': observation.joint_positions.tolist(),
                'joint_forces': observation.joint_forces.tolist(),
                'gripper_open': float(observation.gripper_open),
                'gripper_pose': observation.gripper_pose.tolist(),
                'gripper_matrix': observation.gripper_matrix.tolist(),
                'gripper_joint_positions': observation.gripper_joint_positions.tolist(),
                'gripper_touch_forces': observation.gripper_touch_forces.tolist(),
                'task_low_dim_state': observation.task_low_dim_state.tolist(),
                # 'misc': observation.misc,
            }
            
            data['data'].append(step_data)
        
        json_path = os.path.join(episode_output_dir, 'data.json')
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    task_list_path = 'task_list_exp1.txt' #task_list_exp1
    tasks = read_task_list(task_list_path)
    
    save_image_path = 'data_train_224_allinfo/'
    
    action_mode = MoveArmThenGripper(
        arm_action_mode=JointVelocity(),
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
        headless = True
    )
    env.launch()
    
    for task in tasks:
        if task not in task_dict:
            print(task)
            continue
        task_dir = os.path.join(save_image_path, task)
        # if os.path.exists(task_dir):
        #     continue
        save_data_per_task(env, task, task_dir)
    
    env.shutdown()

if __name__ == "__main__":
    main()
    
    