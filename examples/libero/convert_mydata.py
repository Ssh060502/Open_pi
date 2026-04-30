"""
Convert custom H5 dataset to LeRobot format.

Expected H5 structure:
- episode_0/
    - cam_front: (T, 480, 640, 3) uint8
    - qpos:      (T, 7) float32
    - action:    (T, 7) float32
- episode_1/
    ...
"""

import shutil
from pathlib import Path
import h5py
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

REPO_NAME = "shenghe/libero" 

def main(data_dir: str, *, push_to_hub: bool = False):
    data_path = Path(data_dir)

    if data_path.is_file() and data_path.suffix in [".h5", ".hdf5"]:
        h5_files = [data_path]
    else:
        h5_files = sorted(list(data_path.glob("*.h5")) + list(data_path.glob("*.hdf5")))

    if not h5_files:
        raise FileNotFoundError(f"No .h5/.hdf5 files found in: {data_dir}")

    # 清理旧输出
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # 目标输出
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="my_robot",  # 先定义成这个
        fps=30,  
        # 根据.h5文件
        features={
            "image": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # 遍历每个 h5 文件
    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as f:
            # 找所有 episode_* group(根据.h5文件结构)
            episode_keys = sorted([k for k in f.keys() if k.startswith("episode_")])
            if not episode_keys:
                continue

            for ep in episode_keys:
                cam = f[f"{ep}/cam_front"]      # (T,H,W,3) uint8
                qpos = f[f"{ep}/qpos"]          # (T,7) float32
                act = f[f"{ep}/action"]         # (T,7) float32

                # 取最短长度的帧数，多余的丢掉，确保对齐
                T = min(len(cam), len(qpos), len(act))

                # 没有task字段，先定义一个
                task_text = "perform the task"

                # 一帧一帧开始处理
                for t in range(T):
                    img = np.asarray(cam[t], dtype=np.uint8)
                    dataset.add_frame(
                        {
                            "image": img,
                            "wrist_image": img,
                            "state": np.asarray(qpos[t], dtype=np.float32),
                            "actions": np.asarray(act[t], dtype=np.float32),
                            "task": task_text,
                        }
                    )

                dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["custom", "h5"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

if __name__ == "__main__":
    tyro.cli(main)