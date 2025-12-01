import collections
import dataclasses
import logging
import math
import pathlib

import imageio
import numpy as np
from pyrep import PyRep
from pyrep.objects import VisionSensor
from rlbench import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, JointPosition, EndEffectorPoseViaPlanning, EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.tasks import __dict__ as task_dict
from rlbench.backend.observation import Observation
from rlbench.observation_config import CameraConfig, ObservationConfig

import tqdm
import tyro

from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

LOCAL_LOG_TXT_PATH = "./result.txt"
# LOCAL_LOG_TXT_PATH = "./result_10.txt"

def read_task_list(file_path):
    with open(file_path, 'r') as f:
        tasks = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return tasks


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 10
    #TODO 每个任务不一样
    # replan_steps: dict = {
        
    # }
    
    #################################################################################################################
    # RLBench environment-specific parameters
    #################################################################################################################
    task_eval_path: str = "task_list_exp1.txt"  # RLBench任务类名 task_eval task_list_exp1
    num_trials_per_task: int = 20  # 每个任务的rollout次数 20
    max_steps: int = 320  # 每个episode的最大步数
    #TODO 每个任务不一样
    # max_steps: int = 320
    
    headless: bool = True  # 是否无头模式运行

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "./videos"  # 保存视频的路径
    seed: int = 777  # 随机种子
    

def eval_task(args, env, task_name):
    # 获取任务
    task = env.get_task(task_dict[task_name])
    descriptions, _ = task.reset()
    
    with open(LOCAL_LOG_TXT_PATH, "a") as file:
        file.write(f"\nTask: {task_name} {descriptions}")

    # 初始化策略客户端
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # 开始评估
    total_episodes, total_successes = 0, 0
    task_episodes, task_successes = 0, 0

    for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
        logging.info(f"\nTask: {task_name}")
        logging.info(f"Starting episode {episode_idx + 1}...")

        # 重置环境并获取初始观测
        descriptions, obs = task.reset()
        action_plan = collections.deque()
        gripper_range = obs.gripper_open
        desc_long = max(descriptions,key=len)

        # 设置变量
        t = 0
        replay_images = []
        done = False

        while t < args.max_steps:
            try:
                # 获取并预处理图像
                # 使用前视角摄像头作为主要观测
                img = obs.front_rgb
                wrist_img = obs.wrist_rgb
                
                # 转换为uint8并调整大小
                # img = image_tools.convert_to_uint8(
                #     image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                # )
                # wrist_img = image_tools.convert_to_uint8(
                #     image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                # )

                # 保存预处理图像用于回放视频
                replay_images.append(img)

                if not action_plan:
                    # 执行完之前的动作块 - 计算新的动作块
                    # 准备观测字典                   
                    request_data = {
                        "observation/exterior_image_1_left": img,
                        "observation/wrist_image_left": wrist_img,
                        "observation/joint_position": obs.joint_positions,
                        "observation/gripper_position": obs.gripper_open,
                        # "prompt": descriptions[0] if descriptions else task_name,
                        "prompt": desc_long,
                    }

                    # 查询模型获取动作
                    action_chunk = client.infer(request_data)["actions"]
                    assert (
                        len(action_chunk) >= args.replan_steps
                    ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                    action_plan.extend(action_chunk[: args.replan_steps])

                # 获取下一个动作
                action = action_plan.popleft()
                
                # RLBench动作格式: [joint_velocities, gripper_open]
                # 假设模型输出已经是正确的格式
                rlbench_action = action.tolist()
                # rlbench_action[3:7] = rlbench_action[3:7]/np.linalg.norm(rlbench_action[3:7])
                # print(rlbench_action[3:7])
                
                if rlbench_action[-1] > 0.:
                    rlbench_action[-1] = 0.0
                else:
                    rlbench_action[-1] = 1.0
                # 在环境中执行动作
                obs, reward, done = task.step(rlbench_action)
                
                # 检查任务是否成功完成
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1

            except Exception as e:
                logging.error(f"Caught exception: {e}")
                break

        task_episodes += 1
        total_episodes += 1

        # 保存回放视频
        # suffix = "success" if done else "failure"
        # imageio.mimwrite(
        #     pathlib.Path(args.video_out_path) / f"rollout_{task_name}_{episode_idx}_{suffix}.mp4",
        #     [np.asarray(x) for x in replay_images],
        #     fps=10,
        # )

        # 记录当前结果
        logging.info(f"Success: {done}")
        logging.info(f"# episodes completed so far: {total_episodes}")
        logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
        # with open(LOCAL_LOG_TXT_PATH, "a") as file:
        #     file.write(f"| Current task success rate: {float(task_successes) / float(task_episodes)}")

    # 记录最终结果
    logging.info(f"Task success rate: {float(task_successes) / float(task_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    with open(LOCAL_LOG_TXT_PATH, "a") as file:
        file.write(f"\nTotal success rate: {float(total_successes) / float(total_episodes)}")


def _get_rlbench_state(obs: Observation) -> np.ndarray:
    """从RLBench观测中提取状态向量"""
    state_parts = []
    
    # 关节位置
    if obs.joint_positions is not None:
        state_parts.append(obs.joint_positions)
    
    # 夹爪状态
    if obs.gripper_open is not None:
        state_parts.append([obs.gripper_open])
      
    return np.concatenate(state_parts)


def eval_rlbench(args: Args) -> None:
    # 设置随机种子
    np.random.seed(args.seed)

    # 创建输出目录
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    action_mode = MoveArmThenGripper(
        arm_action_mode=JointVelocity(),
        # arm_action_mode=JointPosition(),
        # arm_action_mode=EndEffectorPoseViaIK(),
        # arm_action_mode=EndEffectorPoseViaPlanning(),
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
    
    # 初始化RLBench环境
    env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=args.headless,
    )
    env.launch()
    
    task_eval_list = read_task_list(args.task_eval_path)
    if task_eval_list is None:
        raise ValueError(f"Empty eval task list: {args.task_eval_path}")
    
    for task_name in task_eval_list:
        eval_task(args, env, task_name)
        
    env.shutdown()
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_rlbench)