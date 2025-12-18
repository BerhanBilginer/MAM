#!/usr/bin/env python3
"""
Train ConvLSTM Autoencoder for panic detection using normal videos only.

Usage:
    python scripts/train_convlstm_panic.py \
        --videos "data/normal_videos/*.mp4" \
        --output models/convlstm_panic.pt \
        --epochs 50 \
        --batch-size 4
"""

import argparse
import sys
from pathlib import Path
import time
from typing import Iterator
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.panic.convlstm_model import PanicConvLSTMAutoencoder
from src.detection.utils.detection import load_model, detect_people
from src.detection.panic.features import build_convlstm_features
from src.detection.utils.pose import detect_people as detect_people_pose


class NormalMotionDataset(Dataset):
    """Dataset of normal motion sequences."""
    
    def __init__(self, sequences: list):
        """
        Args:
            sequences: List of [T, C, H, W] numpy arrays
        """
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return torch.from_numpy(sequence).float()


class DiskSequenceDataset(Dataset):
    def __init__(self, sequence_files: list[str], mmap: bool = True):
        self.sequence_files = sequence_files
        self.mmap = bool(mmap)

    def __len__(self):
        return len(self.sequence_files)

    def __getitem__(self, idx):
        path = self.sequence_files[idx]
        arr = np.load(path, mmap_mode="r" if self.mmap else None)
        return torch.from_numpy(arr).float()


class H5SequenceDataset(Dataset):
    def __init__(self, h5_path: str, dataset_name: str = "sequences"):
        self.h5_path = str(h5_path)
        self.dataset_name = dataset_name
        self._h5 = None
        self._pid = None

    def _ensure_open(self):
        import os
        import h5py

        pid = os.getpid()
        if self._h5 is None or self._pid != pid:
            if self._h5 is not None:
                try:
                    self._h5.close()
                except Exception:
                    pass
            self._h5 = h5py.File(self.h5_path, "r")
            self._pid = pid
        return self._h5[self.dataset_name]

    def __len__(self):
        dset = self._ensure_open()
        return int(dset.shape[0])

    def __getitem__(self, idx):
        dset = self._ensure_open()
        arr = dset[idx]
        return torch.from_numpy(arr).float()




def extract_sequences_from_video(
    video_path: str,
    model,
    sequence_length: int = 16,
    image_size: int = 96,
    stride: int = 8,
    vmax: float = 10.0,
) -> list:
    """Extract sequences from a video using unified feature builder."""
    print(f"\nProcessing: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open video")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Total frames: {total_frames}")
    
    sequences = []
    frame_buffer = []
    frame_idx = 0
    prev_gray_small = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            detections = detect_people_pose(model, frame, conf=0.25, keypoint_conf_threshold=0.3)
            
            features, prev_gray_small = build_convlstm_features(
                frame,
                detections,
                prev_gray_small,
                image_size=image_size,
                vmax=vmax,
            )
            
            if features is not None:
                frame_buffer.append(features)
                
                if len(frame_buffer) == sequence_length:
                    sequence = np.stack(frame_buffer, axis=0)
                    sequences.append(sequence)
                    
                    for _ in range(stride):
                        if frame_buffer:
                            frame_buffer.pop(0)
            
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames, {len(sequences)} sequences")
    
    finally:
        cap.release()
    
    print(f"  Extracted {len(sequences)} sequences")
    return sequences


def iter_sequences_from_video(
    video_path: str,
    model,
    sequence_length: int = 16,
    image_size: int = 96,
    stride: int = 8,
    vmax: float = 10.0,
) -> Iterator[np.ndarray]:
    """Iterate sequences from a video using unified feature builder."""
    print(f"\nProcessing: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open video")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Total frames: {total_frames}")

    frame_buffer: list[np.ndarray] = []
    frame_idx = 0
    prev_gray_small = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            detections = detect_people_pose(model, frame, conf=0.25, keypoint_conf_threshold=0.3)
            
            features, prev_gray_small = build_convlstm_features(
                frame,
                detections,
                prev_gray_small,
                image_size=image_size,
                vmax=vmax,
            )

            if features is not None:
                frame_buffer.append(features)

                if len(frame_buffer) == sequence_length:
                    yield np.stack(frame_buffer, axis=0)

                    for _ in range(stride):
                        if frame_buffer:
                            frame_buffer.pop(0)

            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames")

    finally:
        cap.release()


def train_model(
    model: PanicConvLSTMAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
    save_path: str,
    feature_vmax: float,
):
    """Train the ConvLSTM autoencoder."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        val_errors = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                val_loss += loss.item()
                
                errors = model.compute_reconstruction_error(batch, reconstructed)
                val_errors.extend(errors.cpu().numpy())
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        val_errors = np.array(val_errors)
        threshold = np.percentile(val_errors, 95)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Threshold (95th percentile): {threshold:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'threshold': float(threshold),
                'vmax': float(feature_vmax),
                'sequence_length': getattr(model, '_sequence_length', 16),
                'image_size': getattr(model, '_image_size', 96),
            }, save_path)
            print(f"  ✓ Saved best model to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ConvLSTM autoencoder for panic detection")
    parser.add_argument("--videos", nargs="+", required=True, help="Normal video paths")
    parser.add_argument(
        "--val-videos",
        nargs="+",
        default=None,
        help="Optional validation video paths. If provided, overrides --val-split.",
    )
    parser.add_argument("--output", required=True, help="Output model path")
    parser.add_argument("--model", default="weights/yolo11x-pose.pt", help="YOLO pose model")
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda:0)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--sequence-length", type=int, default=16, help="Sequence length")
    parser.add_argument("--image-size", type=int, default=96, help="Image size (square)")
    parser.add_argument("--vmax", type=float, default=10.0, help="Flow vmax for normalization (px/frame)")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader num_workers (use >0 for faster training on Colab)",
    )
    parser.add_argument(
        "--hdf5-cache",
        default=None,
        help="Optional single-file HDF5 cache path (e.g. data/sequences.h5) to reduce RAM usage",
    )
    parser.add_argument(
        "--reuse-hdf5-cache",
        action="store_true",
        help="Reuse HDF5 cache if it already exists",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional directory to cache extracted sequences as .npy to reduce RAM usage",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse cache if an index file exists in cache-dir",
    )
    parser.add_argument(
        "--cache-float16",
        action="store_true",
        help="Store cached sequences as float16 to reduce disk usage",
    )
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    print("\nLoading YOLO model...")
    yolo_model = load_model(args.model, device=args.device)
    
    print("\nExtracting sequences from videos...")
    from glob import glob

    if args.hdf5_cache:
        import h5py

        h5_path = Path(args.hdf5_cache)
        h5_path.parent.mkdir(parents=True, exist_ok=True)

        train_dset_name = "sequences_train" if args.val_videos else "sequences"
        val_dset_name = "sequences_val" if args.val_videos else "sequences"

        if args.reuse_hdf5_cache and h5_path.exists():
            print(f"Reusing HDF5 cache: {h5_path}")
        else:
            seq_shape = (args.sequence_length, 5, args.image_size, args.image_size)
            dtype = np.float16 if args.cache_float16 else np.float32
            print(f"Writing HDF5 cache to: {h5_path}")

            with h5py.File(h5_path, "w") as f:
                dset_train = f.create_dataset(
                    train_dset_name,
                    shape=(0,) + seq_shape,
                    maxshape=(None,) + seq_shape,
                    dtype=dtype,
                    chunks=(1,) + seq_shape,
                    compression="lzf",
                )
                dset_val = None
                if args.val_videos:
                    dset_val = f.create_dataset(
                        val_dset_name,
                        shape=(0,) + seq_shape,
                        maxshape=(None,) + seq_shape,
                        dtype=dtype,
                        chunks=(1,) + seq_shape,
                        compression="lzf",
                    )
                f.attrs["sequence_length"] = int(args.sequence_length)
                f.attrs["image_size"] = int(args.image_size)
                f.attrs["stride"] = int(args.sequence_length // 2)
                f.attrs["dtype"] = str(np.dtype(dtype))
                f.attrs["vmax"] = float(args.vmax)

                n_train = 0
                for video_pattern in args.videos:
                    video_paths = glob(video_pattern)
                    for video_path in video_paths:
                        for seq in iter_sequences_from_video(
                            video_path,
                            yolo_model,
                            sequence_length=args.sequence_length,
                            image_size=args.image_size,
                            stride=args.sequence_length // 2,
                            vmax=float(args.vmax),
                        ):
                            if dtype == np.float16:
                                seq = seq.astype(np.float16)
                            dset_train.resize((n_train + 1,) + seq_shape)
                            dset_train[n_train] = seq
                            n_train += 1
                            if n_train % 50 == 0:
                                print(f"  Cached train {n_train} sequences")

                n_val = 0
                if args.val_videos and dset_val is not None:
                    for video_pattern in args.val_videos:
                        video_paths = glob(video_pattern)
                        for video_path in video_paths:
                            for seq in iter_sequences_from_video(
                                video_path,
                                yolo_model,
                                sequence_length=args.sequence_length,
                                image_size=args.image_size,
                                stride=args.sequence_length // 2,
                                vmax=float(args.vmax),
                            ):
                                if dtype == np.float16:
                                    seq = seq.astype(np.float16)
                                dset_val.resize((n_val + 1,) + seq_shape)
                                dset_val[n_val] = seq
                                n_val += 1
                                if n_val % 50 == 0:
                                    print(f"  Cached val {n_val} sequences")

            print("HDF5 cache write complete")

        if args.val_videos:
            train_dataset = H5SequenceDataset(str(h5_path), dataset_name=train_dset_name)
            val_dataset = H5SequenceDataset(str(h5_path), dataset_name=val_dset_name)
            print(f"\nTrain sequences extracted: {len(train_dataset)}")
            print(f"Val sequences extracted: {len(val_dataset)}")
            if len(train_dataset) == 0:
                print("ERROR: No train sequences extracted!")
                return 1
            if len(val_dataset) == 0:
                print("ERROR: No val sequences extracted!")
                return 1
        else:
            base_dataset = H5SequenceDataset(str(h5_path), dataset_name=train_dset_name)
            total = len(base_dataset)
            print(f"\nTotal sequences extracted: {total}")
            if total == 0:
                print("ERROR: No sequences extracted!")
                return 1

            indices = np.random.permutation(total)
            split_idx = int(total * (1 - args.val_split))
            train_idx = indices[:split_idx].tolist()
            val_idx = indices[split_idx:].tolist()

            print(f"Train sequences: {len(train_idx)}")
            print(f"Val sequences: {len(val_idx)}")

            train_dataset = Subset(base_dataset, train_idx)
            val_dataset = Subset(base_dataset, val_idx)
    elif args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        train_index_path = cache_dir / ("train_index.txt" if args.val_videos else "index.txt")
        val_index_path = cache_dir / "val_index.txt"

        if args.val_videos:
            if args.reuse_cache and train_index_path.exists() and val_index_path.exists():
                train_files = [
                    line.strip() for line in train_index_path.read_text().splitlines() if line.strip()
                ]
                val_files = [
                    line.strip() for line in val_index_path.read_text().splitlines() if line.strip()
                ]
                print(f"Loaded {len(train_files)} cached train sequences from: {train_index_path}")
                print(f"Loaded {len(val_files)} cached val sequences from: {val_index_path}")
            else:
                run_id = int(time.time())
                train_files = []
                val_files = []
                train_idx = 0
                val_idx = 0

                for video_pattern in args.videos:
                    video_paths = glob(video_pattern)
                    for video_path in video_paths:
                        for seq in iter_sequences_from_video(
                            video_path,
                            yolo_model,
                            sequence_length=args.sequence_length,
                            image_size=args.image_size,
                            stride=args.sequence_length // 2,
                            vmax=float(args.vmax),
                        ):
                            out = cache_dir / f"train_seq_{run_id}_{train_idx:08d}.npy"
                            np.save(out, seq.astype(np.float16) if args.cache_float16 else seq)
                            train_files.append(str(out))
                            train_idx += 1
                            if train_idx % 50 == 0:
                                print(f"  Cached train {train_idx} sequences")

                for video_pattern in args.val_videos:
                    video_paths = glob(video_pattern)
                    for video_path in video_paths:
                        for seq in iter_sequences_from_video(
                            video_path,
                            yolo_model,
                            sequence_length=args.sequence_length,
                            image_size=args.image_size,
                            stride=args.sequence_length // 2,
                            vmax=float(args.vmax),
                        ):
                            out = cache_dir / f"val_seq_{run_id}_{val_idx:08d}.npy"
                            np.save(out, seq.astype(np.float16) if args.cache_float16 else seq)
                            val_files.append(str(out))
                            val_idx += 1
                            if val_idx % 50 == 0:
                                print(f"  Cached val {val_idx} sequences")

                train_index_path.write_text("\n".join(train_files) + ("\n" if train_files else ""))
                val_index_path.write_text("\n".join(val_files) + ("\n" if val_files else ""))
                print(f"Saved train cache index: {train_index_path}")
                print(f"Saved val cache index: {val_index_path}")

            print(f"\nTrain sequences extracted: {len(train_files)}")
            print(f"Val sequences extracted: {len(val_files)}")
            if len(train_files) == 0:
                print("ERROR: No train sequences extracted!")
                return 1
            if len(val_files) == 0:
                print("ERROR: No val sequences extracted!")
                return 1

            train_dataset = DiskSequenceDataset(train_files, mmap=True)
            val_dataset = DiskSequenceDataset(val_files, mmap=True)
        else:
            index_path = train_index_path
            if args.reuse_cache and index_path.exists():
                sequence_files = [
                    line.strip() for line in index_path.read_text().splitlines() if line.strip()
                ]
                print(f"Loaded {len(sequence_files)} cached sequences from: {index_path}")
            else:
                run_id = int(time.time())
                sequence_files = []
                seq_idx = 0

                for video_pattern in args.videos:
                    video_paths = glob(video_pattern)
                    for video_path in video_paths:
                        for seq in iter_sequences_from_video(
                            video_path,
                            yolo_model,
                            sequence_length=args.sequence_length,
                            image_size=args.image_size,
                            stride=args.sequence_length // 2,
                            vmax=float(args.vmax),
                        ):
                            out = cache_dir / f"seq_{run_id}_{seq_idx:08d}.npy"
                            np.save(out, seq.astype(np.float16) if args.cache_float16 else seq)
                            sequence_files.append(str(out))
                            seq_idx += 1

                            if seq_idx % 50 == 0:
                                print(f"  Cached {seq_idx} sequences")

                index_path.write_text("\n".join(sequence_files) + ("\n" if sequence_files else ""))
                print(f"Saved cache index: {index_path}")

            print(f"\nTotal sequences extracted: {len(sequence_files)}")
            if len(sequence_files) == 0:
                print("ERROR: No sequences extracted!")
                return 1

            np.random.shuffle(sequence_files)
            split_idx = int(len(sequence_files) * (1 - args.val_split))
            train_files = sequence_files[:split_idx]
            val_files = sequence_files[split_idx:]

            print(f"Train sequences: {len(train_files)}")
            print(f"Val sequences: {len(val_files)}")

            train_dataset = DiskSequenceDataset(train_files, mmap=True)
            val_dataset = DiskSequenceDataset(val_files, mmap=True)
    else:
        if args.val_videos:
            train_sequences: list[np.ndarray] = []
            val_sequences: list[np.ndarray] = []

            for video_pattern in args.videos:
                video_paths = glob(video_pattern)
                for video_path in video_paths:
                    sequences = extract_sequences_from_video(
                        video_path,
                        yolo_model,
                        sequence_length=args.sequence_length,
                        image_size=args.image_size,
                        stride=args.sequence_length // 2,
                        vmax=float(args.vmax),
                    )
                    train_sequences.extend(sequences)

            for video_pattern in args.val_videos:
                video_paths = glob(video_pattern)
                for video_path in video_paths:
                    sequences = extract_sequences_from_video(
                        video_path,
                        yolo_model,
                        sequence_length=args.sequence_length,
                        image_size=args.image_size,
                        stride=args.sequence_length // 2,
                        vmax=float(args.vmax),
                    )
                    val_sequences.extend(sequences)

            print(f"\nTrain sequences extracted: {len(train_sequences)}")
            print(f"Val sequences extracted: {len(val_sequences)}")
            if len(train_sequences) == 0:
                print("ERROR: No train sequences extracted!")
                return 1
            if len(val_sequences) == 0:
                print("ERROR: No val sequences extracted!")
                return 1

            train_dataset = NormalMotionDataset(train_sequences)
            val_dataset = NormalMotionDataset(val_sequences)
        else:
            all_sequences: list[np.ndarray] = []

            for video_pattern in args.videos:
                video_paths = glob(video_pattern)
                for video_path in video_paths:
                    sequences = extract_sequences_from_video(
                        video_path,
                        yolo_model,
                        sequence_length=args.sequence_length,
                        image_size=args.image_size,
                        stride=args.sequence_length // 2,
                        vmax=float(args.vmax),
                    )
                    all_sequences.extend(sequences)

            print(f"\nTotal sequences extracted: {len(all_sequences)}")
            if len(all_sequences) == 0:
                print("ERROR: No sequences extracted!")
                return 1

            np.random.shuffle(all_sequences)
            split_idx = int(len(all_sequences) * (1 - args.val_split))
            train_sequences = all_sequences[:split_idx]
            val_sequences = all_sequences[split_idx:]

            print(f"Train sequences: {len(train_sequences)}")
            print(f"Val sequences: {len(val_sequences)}")

            train_dataset = NormalMotionDataset(train_sequences)
            val_dataset = NormalMotionDataset(val_sequences)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )
    
    print("\nCreating model...")
    print(f"Feature channels: 5 (P3: flow_x, flow_y, flow_mag, bbox_heatmap, divergence)")
    
    model = PanicConvLSTMAutoencoder(
        input_channels=5,
        hidden_dims=[32, 64, 32],
        kernel_size=(3, 3),
        num_layers=3,
    )
    model._sequence_length = args.sequence_length
    model._image_size = args.image_size
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    print("\nStarting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        device=device,
        save_path=args.output,
        feature_vmax=float(args.vmax),
    )
    
    print(f"\n✓ Training complete!")
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    sys.exit(main())
