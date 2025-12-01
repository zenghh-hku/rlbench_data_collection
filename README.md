# rlbench_data_collection
Collect demonstrations from tasks. It's a part of my capstone project([Github](http://github.com/HMTCuro/ARIN7600_2025)).

There are some dataset collected ([Huggingface](https://huggingface.co/datasets/Zenghh2000/rlbench_dataset))

data_collection.py can run directly with RLBench library.

eval.py is written in imitation of the main.py in the example of $\pi_0$([Github](https://github.com/Physical-Intelligence/openpi)) project and requires a corresponding environment.

test_eef.py is used to verify the availability of the saved gripper pose data.

# Dataset Structure

The dataset is organized in a hierarchical structure to facilitate efficient model training and data analysis. Each task maintains a dedicated root directory containing individual episode subdirectories.

## Directory Organization
```
task_root_directory/
├── episode_0001/
│ ├── step0000001_main_images
│ ├── step0000002_main_images
│ ├── ...
│ ├── step0000001_wrist_images
│ ├── step0000002_wrist_images
│ ├── ...
│ └── data.json
├── episode_0002/
│ ├── step0000001_main_images
│ ├── step0000002_main_images
│ ├── ...
│ ├── step0000001_wrist_images
│ ├── step0000002_wrist_images
│ ├── ...
│ └── data.json
└── ...
```

## Data Contents

Within each episode directory, the dataset contains:

- **Visual Streams**: Sequential JPEG images from both frontal and wrist-mounted cameras at each time step
- **State Metadata**: A comprehensive JSON file containing the complete episode record

## JSON Data Structure

The `data.json` file follows this hierarchical structure:
```
data.json (Root Dict)
├── task: "[instruction]"
├── episode_index: [episode_idx]
├── total_steps: [len(demonstration)]
└── data: (List)
├── [0] (Dict)
│ ├── step_index: [step_idx]
│ ├── joint_velocities: [List of Floats]
│ ├── joint_positions: [List of Floats]
│ ├── joint_forces: [List of Floats]
│ ├── gripper_open: [Float]
│ ├── gripper_pose: [List of Floats]
│ ├── gripper_matrix: [List of Floats]
│ ├── gripper_joint_positions: [List of Floats]
│ ├── gripper_touch_forces: [List of Floats]
│ └── task_low_dim_state: [List of Floats]
├── [1] (Dict)
│ └── ... (Same structure as index 0)
└── [n] (Dict)
└── ... (Same structure as index 0)
```

## Key Fields Description

| Field | Type | Description |
|-------|------|-------------|
| **task** | String | Natural language instruction describing the task objective |
| **episode_index** | Integer | Unique identifier for the episode |
| **total_steps** | Integer | Total number of steps in the demonstration |
| **data** | List | Step-wise records containing robot state and proprioceptive data |

### Step Data Fields:
- **joint_velocities**: Robot joint velocities
- **joint_positions**: Robot joint positions
- **joint_forces**: Robot joint forces/torques
- **gripper_open**: Gripper open amount (0=closed, 1=open)
- **gripper_pose**: Gripper pose in world coordinates [x, y, z, qx, qy, qz, qw]
- **gripper_matrix**: 4×4 transformation matrix flattened to list
- **gripper_joint_positions**: Individual gripper joint positions
- **gripper_touch_forces**: Tactile sensor readings from gripper
- **task_low_dim_state**: Task-specific low-dimensional state representation
