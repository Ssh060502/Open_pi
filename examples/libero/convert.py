"""
Convert custom H5 dataset to LeRobot format.

Expected H5 structure (best-effort):
- episode_0/
    - cam_front: (T, 480, 640, 3) uint8
    - qpos:      (T, 7) float32
    - action:    (T, 7) float32
- episode_1/
    ...

If some episodes are corrupted / unreadable, they will be skipped with a warning.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import h5py
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

REPO_NAME = "shenghe/libero"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def _safe_open_dataset(group: h5py.Group, name: str) -> h5py.Dataset | None:
    """Return dataset if present and readable; otherwise None."""
    if name not in group:
        return None
    obj = group[name]
    if not isinstance(obj, h5py.Dataset):
        return None
    try:
        _ = obj.shape
        _ = obj.dtype
    except Exception:
        return None
    return obj


def main(data_dir: str, *, push_to_hub: bool = False):
    data_path = Path(data_dir)

    if data_path.is_file() and data_path.suffix.lower() in [".h5", ".hdf5"]:
        h5_files = [data_path]
    else:
        h5_files = sorted(list(data_path.glob("*.h5")) + list(data_path.glob("*.hdf5")))

    if not h5_files:
        raise FileNotFoundError(f"No .h5/.hdf5 files found in: {data_dir}")

    # Clean previous output for this repo_id
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="my_robot",
        fps=30,  # TODO: set to your real control frequency if known
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
        # If you see flaky reads on NFS/WSL mounts, try (1, 1).
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Global counters
    ok_eps = 0
    skipped_eps = 0
    ok_frames = 0
    skipped_files = 0

    for h5_path in h5_files:
        logging.info("Processing file: %s", h5_path)

        try:
            f = h5py.File(h5_path, "r")
        except Exception as e:
            skipped_files += 1
            logging.warning("skip whole file %s: cannot open: %s: %s", h5_path.name, type(e).__name__, e)
            continue

        with f:
            episode_keys = sorted(
                [k for k in f.keys() if isinstance(f[k], h5py.Group) and k.startswith("episode_")]
            )
            if not episode_keys:
                logging.warning("skip file %s: no episode_* groups", h5_path.name)
                continue

            file_ok_eps = 0
            file_skip_eps = 0
            file_ok_frames = 0

            for ep in episode_keys:
                try:
                    g = f[ep]
                    if not isinstance(g, h5py.Group):
                        skipped_eps += 1
                        file_skip_eps += 1
                        logging.warning("skip %s/%s: not a group", h5_path.name, ep)
                        continue

                    cam = _safe_open_dataset(g, "cam_front")
                    qpos = _safe_open_dataset(g, "qpos")
                    act = _safe_open_dataset(g, "action")
                    if cam is None or qpos is None or act is None:
                        skipped_eps += 1
                        file_skip_eps += 1
                        logging.warning(
                            "skip %s/%s: missing/unreadable cam_front/qpos/action", h5_path.name, ep
                        )
                        continue

                    T = int(min(len(cam), len(qpos), len(act)))
                    if T <= 0:
                        skipped_eps += 1
                        file_skip_eps += 1
                        logging.warning("skip %s/%s: empty episode (T=%s)", h5_path.name, ep, T)
                        continue

                    task_text = "perform the task"

                    for t in range(T):
                        img = np.asarray(cam[t], dtype=np.uint8)
                        dataset.add_frame(
                            {
                                "image": img,
                                # Temporary: duplicate main camera as wrist camera for compatibility
                                "wrist_image": img,
                                "state": np.asarray(qpos[t], dtype=np.float32),
                                "actions": np.asarray(act[t], dtype=np.float32),
                                "task": task_text,
                            }
                        )

                    dataset.save_episode()

                    ok_eps += 1
                    file_ok_eps += 1
                    ok_frames += T
                    file_ok_frames += T

                except Exception as e:
                    skipped_eps += 1
                    file_skip_eps += 1
                    logging.warning("skip %s/%s: %s: %s", h5_path.name, ep, type(e).__name__, e)
                    continue

            logging.info(
                "file summary: %s ok_episodes=%d skipped_episodes=%d ok_frames=%d",
                h5_path.name,
                file_ok_eps,
                file_skip_eps,
                file_ok_frames,
            )

    logging.info(
        "TOTAL: ok_episodes=%d skipped_episodes=%d ok_frames=%d skipped_files=%d",
        ok_eps,
        skipped_eps,
        ok_frames,
        skipped_files,
    )

    if ok_eps == 0:
        raise RuntimeError("No episodes were successfully converted. Check warnings above.")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["custom", "h5"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

if __name__ == "__main__":
    tyro.cli(main)