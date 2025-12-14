from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from numpy import typing as npt

from src.detection.utils.detection import Detection


@dataclass
class PanicEmbedding:
    """Container for normal motion features extracted from a video segment.
    
    Used for anomaly detection: deviations from normal patterns indicate panic.
    """
    
    # Flow-based features
    flow_mag_mean: float
    flow_mag_std: float
    flow_mag_p50: float
    flow_mag_p95: float
    flow_mag_p99: float
    
    # Direction entropy features
    direction_entropy_mean: float
    direction_entropy_std: float
    direction_entropy_max: float
    
    # People count features
    people_count_mean: float
    people_count_std: float
    people_count_max: float
    
    # Motion density features (flow in person ROIs)
    roi_flow_mean: float
    roi_flow_std: float
    roi_flow_p95: float
    
    # Temporal features
    flow_acceleration: float  # change in flow magnitude over time
    entropy_acceleration: float  # change in entropy over time
    
    # Label (for training, always "normal" for anomaly detection)
    label: str = "normal"
    
    def to_vector(self) -> npt.NDArray[np.float32]:
        """Convert embedding to feature vector."""
        return np.array([
            self.flow_mag_mean,
            self.flow_mag_std,
            self.flow_mag_p50,
            self.flow_mag_p95,
            self.flow_mag_p99,
            self.direction_entropy_mean,
            self.direction_entropy_std,
            self.direction_entropy_max,
            self.people_count_mean,
            self.people_count_std,
            self.people_count_max,
            self.roi_flow_mean,
            self.roi_flow_std,
            self.roi_flow_p95,
            self.flow_acceleration,
            self.entropy_acceleration,
        ], dtype=np.float32)
    
    @staticmethod
    def feature_names() -> list[str]:
        """Get feature names for the vector."""
        return [
            "flow_mag_mean", "flow_mag_std", "flow_mag_p50", "flow_mag_p95", "flow_mag_p99",
            "direction_entropy_mean", "direction_entropy_std", "direction_entropy_max",
            "people_count_mean", "people_count_std", "people_count_max",
            "roi_flow_mean", "roi_flow_std", "roi_flow_p95",
            "flow_acceleration", "entropy_acceleration",
        ]


class EmbeddingExtractor:
    """Extract panic detection embeddings from video segments."""
    
    def __init__(self, window_size: int = 30) -> None:
        """
        Args:
            window_size: Number of frames to aggregate for one embedding
        """
        self.window_size = window_size
        self._reset_buffer()
    
    def _reset_buffer(self) -> None:
        """Reset internal buffers."""
        self.flow_mags: list[float] = []
        self.direction_entropies: list[float] = []
        self.people_counts: list[int] = []
        self.roi_flows: list[float] = []
    
    def add_frame_features(
        self,
        flow_mag: float,
        direction_entropy: float,
        people_count: int,
        roi_flow: float,
    ) -> Optional[PanicEmbedding]:
        """
        Add features from one frame. Returns embedding when window is full.
        
        Args:
            flow_mag: Flow magnitude (e.g., p95 or mean)
            direction_entropy: Direction entropy
            people_count: Number of people detected
            roi_flow: Flow magnitude in person ROIs
            
        Returns:
            PanicEmbedding if window is complete, None otherwise
        """
        self.flow_mags.append(flow_mag)
        self.direction_entropies.append(direction_entropy)
        self.people_counts.append(people_count)
        self.roi_flows.append(roi_flow)
        
        if len(self.flow_mags) >= self.window_size:
            embedding = self._compute_embedding()
            self._reset_buffer()
            return embedding
        
        return None
    
    def _compute_embedding(self, label: str = "unknown") -> PanicEmbedding:
        """Compute embedding from buffered features."""
        flow_arr = np.array(self.flow_mags, dtype=np.float32)
        entropy_arr = np.array(self.direction_entropies, dtype=np.float32)
        people_arr = np.array(self.people_counts, dtype=np.float32)
        roi_arr = np.array(self.roi_flows, dtype=np.float32)
        
        # Temporal acceleration (simple diff)
        flow_accel = float(np.mean(np.diff(flow_arr))) if len(flow_arr) > 1 else 0.0
        entropy_accel = float(np.mean(np.diff(entropy_arr))) if len(entropy_arr) > 1 else 0.0
        
        return PanicEmbedding(
            flow_mag_mean=float(np.mean(flow_arr)),
            flow_mag_std=float(np.std(flow_arr)),
            flow_mag_p50=float(np.percentile(flow_arr, 50)),
            flow_mag_p95=float(np.percentile(flow_arr, 95)),
            flow_mag_p99=float(np.percentile(flow_arr, 99)),
            direction_entropy_mean=float(np.mean(entropy_arr)),
            direction_entropy_std=float(np.std(entropy_arr)),
            direction_entropy_max=float(np.max(entropy_arr)),
            people_count_mean=float(np.mean(people_arr)),
            people_count_std=float(np.std(people_arr)),
            people_count_max=float(np.max(people_arr)),
            roi_flow_mean=float(np.mean(roi_arr)),
            roi_flow_std=float(np.std(roi_arr)),
            roi_flow_p95=float(np.percentile(roi_arr, 95)),
            flow_acceleration=flow_accel,
            entropy_acceleration=entropy_accel,
            label=label,
        )
    
    def finalize(self, label: str = "unknown") -> Optional[PanicEmbedding]:
        """Finalize and return embedding from remaining frames."""
        if len(self.flow_mags) > 0:
            embedding = self._compute_embedding(label)
            self._reset_buffer()
            return embedding
        return None


class EmbeddingDatabase:
    """Store and load normal motion embeddings for anomaly detection.
    
    This database stores only normal motion patterns. Panic is detected
    as deviation from these normal patterns (anomaly detection).
    """
    
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.embeddings: list[PanicEmbedding] = []
        self.normal_vectors: Optional[npt.NDArray[np.float32]] = None
        self.mean_vector: Optional[npt.NDArray[np.float32]] = None
        self.std_vector: Optional[npt.NDArray[np.float32]] = None
    
    def add_embedding(self, embedding: PanicEmbedding) -> None:
        """Add an embedding to the database."""
        self.embeddings.append(embedding)
    
    def save(self) -> None:
        """Save embeddings to disk."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "embeddings": [asdict(emb) for emb in self.embeddings],
            "feature_names": PanicEmbedding.feature_names(),
        }
        
        with open(self.db_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(self.embeddings)} embeddings to {self.db_path}")
    
    def load(self) -> None:
        """Load embeddings from disk and compute statistics."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Embedding database not found: {self.db_path}")
        
        with open(self.db_path, "r") as f:
            data = json.load(f)
        
        self.embeddings = [PanicEmbedding(**emb) for emb in data["embeddings"]]
        
        # Only use normal embeddings for anomaly detection
        normal_embs = [emb for emb in self.embeddings if emb.label == "normal"]
        
        if not normal_embs:
            raise ValueError("No normal embeddings found in database. Need normal motion patterns for anomaly detection.")
        
        self.normal_vectors = np.stack([emb.to_vector() for emb in normal_embs])
        
        # Compute statistics for anomaly detection
        self.mean_vector = np.mean(self.normal_vectors, axis=0)
        self.std_vector = np.std(self.normal_vectors, axis=0) + 1e-6  # avoid division by zero
        
        print(f"Loaded {len(normal_embs)} normal motion embeddings for anomaly detection")
    
    def compute_anomaly_score(self, query: PanicEmbedding) -> dict[str, float]:
        """
        Compute anomaly score for query embedding.
        
        Anomaly detection approach:
        1. Compute reconstruction error (distance from normal patterns)
        2. Compute z-score (how many std deviations from mean)
        3. Compute nearest neighbor distance
        
        Higher scores = more anomalous = more likely panic
        
        Returns:
            Dictionary with anomaly metrics
        """
        query_vec = query.to_vector()
        
        results = {}
        
        if self.normal_vectors is None or self.mean_vector is None:
            return results
        
        # 1. Mahalanobis-like distance (normalized by std)
        normalized_diff = (query_vec - self.mean_vector) / self.std_vector
        mahalanobis_dist = float(np.sqrt(np.sum(normalized_diff ** 2)))
        results["mahalanobis_distance"] = mahalanobis_dist
        
        # 2. Reconstruction error (L2 distance to mean)
        reconstruction_error = float(np.linalg.norm(query_vec - self.mean_vector))
        results["reconstruction_error"] = reconstruction_error
        
        # 3. Nearest neighbor distance (minimum distance to any normal pattern)
        distances = np.linalg.norm(self.normal_vectors - query_vec, axis=1)
        nn_distance = float(np.min(distances))
        results["nearest_neighbor_distance"] = nn_distance
        
        # 4. Average distance to k nearest neighbors
        k = min(5, len(self.normal_vectors))
        k_nearest_distances = np.partition(distances, k-1)[:k]
        results["knn_distance"] = float(np.mean(k_nearest_distances))
        
        # 5. Z-scores for each feature (max absolute z-score)
        z_scores = np.abs(normalized_diff)
        results["max_z_score"] = float(np.max(z_scores))
        results["mean_z_score"] = float(np.mean(z_scores))
        
        # 6. Composite anomaly score (weighted combination)
        # Normalize each metric to similar scale
        norm_mahal = mahalanobis_dist / 10.0  # typical range 0-10
        norm_recon = reconstruction_error / 5.0  # typical range 0-5
        norm_nn = nn_distance / 5.0
        
        anomaly_score = (
            0.4 * norm_mahal +
            0.3 * norm_recon +
            0.3 * norm_nn
        )
        results["anomaly_score"] = float(anomaly_score)
        
        return results
    
    @staticmethod
    def _cosine_similarity_batch(query: npt.NDArray[np.float32], vectors: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Compute cosine similarity between query and batch of vectors."""
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        return np.dot(vectors_norm, query_norm)
