from collections import namedtuple
from typing import List, Tuple

from utils.utils import main_process_print

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

Batch = namedtuple("Batch", "hand_joint_pose desired_hand_joint_pose image depth contact terminals")

def segment(
    observations: np.ndarray, 
    terminals: np.ndarray, 
    max_path_length: int
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Segment observations into sequences based on terminal flags.

    Args:
        observations (np.ndarray): The array of observations.
        terminals (np.ndarray): Boolean flags indicating trajectory ends.
        max_path_length (int): Maximum length for padded trajectories.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[int]]:
            - Padded trajectories.
            - Terminal flags for the padded trajectories.
            - Original path lengths.
    """
    assert len(observations) == len(terminals), "Observations and terminals must have the same length"

    observation_dim = observations.shape[1:]
    trajectories, path_lengths = [], []

    for obs, terminal in zip(observations, terminals):
        if not trajectories or terminal:
            trajectories.append([])
        trajectories[-1].append(obs)

    # Convert to arrays and pad
    padded_trajectories = np.zeros((len(trajectories), max_path_length) + observation_dim, dtype=observations.dtype)
    terminal_flags = np.zeros((len(trajectories), max_path_length), dtype=bool)

    for i, traj in enumerate(trajectories):
        traj = np.stack(traj)
        path_lengths.append(len(traj))
        padded_trajectories[i, :len(traj)] = traj
        terminal_flags[i, len(traj):] = True

    return padded_trajectories, terminal_flags, path_lengths

class ContactDataset(Dataset):
    def __init__(
            self,
            data_dir: str = "assets/data",
            horizon: int = 32,
    ):
        """
        Dataset for processing multimodal sequence data.

        Args:
            data_dir (str): Base directory containing the data subdirectories.
            horizon (int): Length of the input sequence window.
        """
        
        #### Shapes for each modalities
        # hand_joint_pose_history: [timesteps , joint_values]               -> [path_length*num_files, 16]
        # desired_hand_joint_pose_history: [timesteps , joint_values]       -> [path_length*num_files, 16]
        # image_history: [timesteps , height , width , channels]            -> [path_length*num_files, 240, 320, 3]
        # depth_history: [timesteps , height , width , 1]                   -> [path_length*num_files, 240, 320, 1]
        # raw_contact_history: [timesteps , sensors , force_vector]         -> [path_length*num_files, 4, 3]
        # booled_contact_history: [timesteps , sensors]                     -> [path_length*num_files, 4]
        # terminal_flags: [timesteps]                                       -> [path_length*num_files]
        ####

        main_process_print(
            f"[ transformer/sequence ] Sequence Length: {horizon}"
        )

        self.data_dir = Path(data_dir)
        self.horizon = horizon

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Directory {data_dir} does not exist")

        dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        if not dirs:
            raise ValueError(f"No subdirectories found in {self.data_dir}")

        # Reading all data from files
        main_process_print(f"[ transformer/sequence ] Reading data lazily", end="...")
        # Each modality: list aligned with dirs; each entry is a memmapped np.ndarray
        self.hand_joint_pose_files, self.terminals_files = self._read_file_list(
            dirs, "hand_joint_pose_history.npy", return_terminals=True
        )
        self.desired_hand_joint_pose_files = self._read_file_list(dirs, "desired_hand_joint_pose_history.npy")
        self.image_files = self._read_file_list(dirs, "image_history.npy")
        self.depth_files = self._read_file_list(dirs, "depth_history.npy")
        contact_files = self._read_file_list(dirs, "contact_history.npy")
        # Convert contact vectors to booleans lazily by wrapping access (we'll compute per-slice)
        self.contact_files = contact_files

        # Sanity check same lengths across modalities per file
        self.lengths = []
        for i in range(len(self.hand_joint_pose_files)):
            Ls = {
                "pose": self.hand_joint_pose_files[i].shape[0],
                "desired_pose": self.desired_hand_joint_pose_files[i].shape[0],
                "img": self.image_files[i].shape[0],
                "depth": self.depth_files[i].shape[0],
                "contact": self.contact_files[i].shape[0],
                "term": self.terminals_files[i].shape[0],
            }
            if len(set(Ls.values())) != 1:
                raise ValueError(f"Length mismatch in file {i}: {Ls}")
            self.lengths.append(Ls["pose"])

        main_process_print('✓')

        # Build valid window indices per file (no padding needed)
        main_process_print(f"[ transformer/sequence ] Building indices", end="...")
        self._build_indices()
        main_process_print('✓')

    # ---------- lazy readers ----------

    def _read_file_list(
        self,
        dirs: List[Path],
        file_name: str,
        return_terminals: bool = False,
    ):
        arrays = []
        terminals = []

        for dir in dirs:
            file_path = dir / file_name
            if not file_path.exists():
                main_process_print(f"Warning: File not found {file_path}")
                continue

            # memory-map: no full load into RAM
            arr = np.load(file_path, mmap_mode="r")
            arrays.append(arr)

            if return_terminals:
                # One True at the end of the file, like before
                term = np.zeros(arr.shape[0], dtype=bool)
                if term.shape[0] > 0:
                    term[-1] = True
                terminals.append(term)

        if not arrays:
            raise ValueError(f"No data found for {file_name}")

        if return_terminals:
            return arrays, terminals
        return arrays
    
    # ---------- indices ----------

    def _build_indices(self):
        """Precompute (file_idx, start, end) for all valid horizon windows."""
        self.indices = []
        for f_idx, L in enumerate(self.lengths):
            if L >= self.horizon:
                for start in range(0, L - self.horizon + 1):
                    self.indices.append((f_idx, start, start + self.horizon))
        self.indices = np.array(self.indices, dtype=np.int64)
        main_process_print(f"[ transformer/sequence ] #windows: {len(self.indices)}")

    # ---------- Dataset protocol ----------

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Batch:
        f_idx, start, end = self.indices[idx]

        # Slice lazily from each memmapped file
        hand_joint_pose = torch.from_numpy(np.array(self.hand_joint_pose_files[f_idx][start:end])).float()
        desired_hand_joint_pose = torch.from_numpy(np.array(self.desired_hand_joint_pose_files[f_idx][start:end])).float()
        image = torch.from_numpy(np.array(self.image_files[f_idx][start:end])).float()
        depth = torch.from_numpy(np.array(self.depth_files[f_idx][start:end])).float()

        # contact_history was [.., sensors, 3]; convert to boolean contact on-the-fly
        # any non-zero in last axis -> True (exactly as before)
        contact_raw = self.contact_files[f_idx][start:end]            # memmap slice
        contact_bool = np.any(np.array(contact_raw) != 0, axis=-1)    # (H, sensors) bool
        contact = torch.from_numpy(contact_bool).bool()

        # terminals: same semantics as before (True only at last frame of a file)
        terminals_np = self.terminals_files[f_idx][start:end]
        terminals = torch.from_numpy(np.array(terminals_np)).bool()

        return Batch(
            hand_joint_pose=hand_joint_pose,
            desired_hand_joint_pose=desired_hand_joint_pose,
            image=image,
            depth=depth,
            contact=contact,
            terminals=terminals,
        )