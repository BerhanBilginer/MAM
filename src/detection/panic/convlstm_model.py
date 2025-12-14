"""
ConvLSTM-based Autoencoder for Panic Detection (Anomaly Detection)

This module implements a ConvLSTM autoencoder that learns normal motion patterns.
Panic is detected as high reconstruction error (anomaly).
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell implementation."""
    
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: tuple, bias: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.bias = bias
        
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
    
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        )


class ConvLSTM(nn.Module):
    """Multi-layer ConvLSTM."""
    
    def __init__(self, input_dim: int, hidden_dims: list, kernel_size: tuple, num_layers: int, 
                 batch_first: bool = True, bias: bool = True, return_all_layers: bool = False):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        
        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dims[i],
                    kernel_size=self.kernel_size,
                    bias=self.bias
                )
            )
        
        self.cell_list = nn.ModuleList(cell_list)
    
    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        
        b, seq_len, _, h, w = input_tensor.size()
        
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
        
        layer_output_list = []
        last_state_list = []
        
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c]
                )
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        
        return layer_output_list, last_state_list
    
    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states


class PanicConvLSTMAutoencoder(nn.Module):
    """
    ConvLSTM Autoencoder for normal motion pattern learning.
    
    Architecture:
    - Input: Sequence of optical flow + pose features [B, T, C, H, W]
    - Encoder: ConvLSTM layers
    - Decoder: ConvLSTM layers
    - Output: Reconstructed sequence
    
    Training: Only on normal videos
    Inference: High reconstruction error = Panic (anomaly)
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        hidden_dims: list = [32, 64, 32],
        kernel_size: tuple = (3, 3),
        num_layers: int = 3,
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        
        self.encoder = ConvLSTM(
            input_dim=input_channels,
            hidden_dims=hidden_dims,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True,
            return_all_layers=False
        )
        
        decoder_hidden_dims = hidden_dims[::-1]
        self.decoder = ConvLSTM(
            input_dim=hidden_dims[-1],
            hidden_dims=decoder_hidden_dims,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True,
            return_all_layers=False
        )
        
        self.output_layer = nn.Conv2d(
            in_channels=decoder_hidden_dims[-1],
            out_channels=input_channels,
            kernel_size=1,
            padding=0
        )
    
    def forward(self, x):
        encoder_output, encoder_state = self.encoder(x)
        encoded = encoder_output[0]
        
        decoder_output, _ = self.decoder(encoded)
        decoded = decoder_output[0]
        
        b, t, c, h, w = decoded.shape
        decoded_reshaped = decoded.view(b * t, c, h, w)
        reconstructed = self.output_layer(decoded_reshaped)
        reconstructed = reconstructed.view(b, t, self.input_channels, h, w)
        
        return reconstructed
    
    def compute_reconstruction_error(self, x, reconstructed):
        return torch.mean((x - reconstructed) ** 2, dim=[1, 2, 3, 4])
    
    def predict_anomaly(self, x, threshold: float = 0.1):
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = self.compute_reconstruction_error(x, reconstructed)
            is_anomaly = error > threshold
        return is_anomaly, error


class PanicConvLSTMDetector:
    """
    Wrapper for ConvLSTM-based panic detection.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        threshold: float = 0.1,
        sequence_length: int = 16,
        image_size: tuple = (64, 64),
    ):
        self.device = torch.device(device)
        self.threshold = threshold
        self.sequence_length = sequence_length
        self.image_size = image_size
        
        self.model = PanicConvLSTMAutoencoder(
            input_channels=3,
            hidden_dims=[32, 64, 32],
            kernel_size=(3, 3),
            num_layers=3,
        )
        
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.threshold = checkpoint.get('threshold', self.threshold)
            print(f"Loaded ConvLSTM model from {model_path}")
            print(f"Threshold: {self.threshold:.4f}")
        else:
            print(f"WARNING: Model not found at {model_path}")
            print("Using untrained model - train first!")
        
        self.model.to(self.device)
        self.model.eval()
        
        self.frame_buffer = []
    
    def prepare_input(self, flow_mag: np.ndarray, flow_angle: np.ndarray, 
                     pose_heatmap: Optional[np.ndarray] = None) -> np.ndarray:
        import cv2
        
        flow_mag = cv2.resize(flow_mag, self.image_size)
        flow_angle = cv2.resize(flow_angle, self.image_size)
        
        flow_x = flow_mag * np.cos(flow_angle)
        flow_y = flow_mag * np.sin(flow_angle)
        
        if pose_heatmap is None:
            pose_heatmap = np.zeros_like(flow_mag)
        else:
            pose_heatmap = cv2.resize(pose_heatmap, self.image_size)
        
        features = np.stack([flow_x, flow_y, pose_heatmap], axis=0)
        features = features / (np.max(np.abs(features)) + 1e-6)
        
        return features.astype(np.float32)
    
    def add_frame(self, flow_mag: np.ndarray, flow_angle: np.ndarray,
                  pose_heatmap: Optional[np.ndarray] = None) -> Optional[tuple]:
        features = self.prepare_input(flow_mag, flow_angle, pose_heatmap)
        self.frame_buffer.append(features)
        
        if len(self.frame_buffer) < self.sequence_length:
            return None
        
        if len(self.frame_buffer) > self.sequence_length:
            self.frame_buffer.pop(0)
        
        sequence = np.stack(self.frame_buffer, axis=0)
        sequence = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
        
        is_panic, error = self.model.predict_anomaly(sequence, self.threshold)
        
        return bool(is_panic[0].item()), float(error[0].item())
    
    def reset(self):
        self.frame_buffer = []


def create_model(input_channels: int = 3, hidden_dims: list = [32, 64, 32]) -> PanicConvLSTMAutoencoder:
    return PanicConvLSTMAutoencoder(
        input_channels=input_channels,
        hidden_dims=hidden_dims,
        kernel_size=(3, 3),
        num_layers=len(hidden_dims),
    )
