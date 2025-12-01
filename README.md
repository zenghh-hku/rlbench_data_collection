# rlbench_data_collection
collect demonstrations from tasks

There are some dataset collected ([Huggingface](https://huggingface.co/datasets/Zenghh2000/rlbench_dataset))

data_collection.py can run directly with RLBench library.

eval.py is written in imitation of the main.py in the example of $\pi_0$([Github](https://github.com/Physical-Intelligence/openpi)) project and requires a corresponding environment.

test_eef.py is used to verify the availability of the saved gripper pose data.

# Dataset Structure

The dataset is organized in a hierarchical structure to facilitate efficient model training and data analysis. Each task maintains a dedicated root directory containing individual episode subdirectories.

## Directory Organization
