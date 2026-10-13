"""Shared vector/image encoder and PPO/REINFORCE agents for zeroRL.

Install as zerorl/helpers/vision_agents.py and re-export both agents
from zerorl/helpers/factory.py for backward compatibility.
"""

import math
import torch
from typing import Literal
from torch import nn


class ObservationEncoder(nn.Module):
    """Encode vectors or CHW/HWC images to a fixed-size feature vector.

    Images are assumed to contain pixels in [0, 255] when image_scale=255.
    Supply image_scale=None for already-normalized image inputs.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...] | int,
        hidden_dim: int = 64,
        *,
        image_layout: Literal["auto", "CHW", "HWC"] = "auto",
        image_scale: float | None = 255.0):
        super().__init__()
        self.obs_shape = (obs_shape,) if isinstance(obs_shape, int) else tuple(obs_shape)
        if not self.obs_shape or any(dim <= 0 for dim in self.obs_shape):
            raise ValueError(f"Invalid observation shape: {self.obs_shape}")
        if image_scale is not None and image_scale <= 0:
            raise ValueError("image_scale must be positive or None")

        self.is_image = len(self.obs_shape) == 3
        self.image_scale = image_scale
        self.image_layout: str | None = None
        self.output_dim = hidden_dim

        if self.is_image:
            if image_layout == "auto":
                first_is_channel = self.obs_shape[0] in (1, 3, 4)
                last_is_channel = self.obs_shape[2] in (1, 3, 4)
                if first_is_channel == last_is_channel:
                    raise ValueError(
                        f"Cannot infer image layout from {self.obs_shape}; "
                        "pass image_layout='CHW' or 'HWC'."
                    )
                image_layout = "CHW" if first_is_channel else "HWC"
            elif image_layout not in ("CHW", "HWC"):
                raise ValueError("image_layout must be 'auto', 'CHW' or 'HWC'")

            self.image_layout = image_layout
            c, h, w = (self.obs_shape if image_layout == "CHW"
                       else (self.obs_shape[2], self.obs_shape[0], self.obs_shape[1]))

            self.features = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            try:
                with torch.no_grad():
                    self.n_features = self.features(torch.zeros(1, c, h, w)).shape[-1]
            except RuntimeError as exc:
                raise ValueError(
                    f"Image resolution {h}x{w} is too small for NatureCNN; "
                    "use larger images or inject a custom encoder."
                ) from exc
        else:
            self.features = nn.Flatten(start_dim=1) #type: ignore
            self.n_features = math.prod(self.obs_shape)

    def get_features(self): return (self.features, self.n_features)



def make_encoder(
    obs_shape: tuple[int, ...] | int,
    hidden_dim: int,
    encoder: nn.Module | None,
    encoder_output_dim: int | None,
    image_layout: Literal["auto", "CHW", "HWC"],
    image_scale: float | None,
) -> tuple[nn.Module, int]:
    if encoder is None:
        obs_encoder = ObservationEncoder(
            obs_shape,
            hidden_dim,
            image_layout=image_layout,
            image_scale=image_scale,
        )
        return obs_encoder.get_features()
    dim = encoder_output_dim
    if dim is None:
        dim = getattr(encoder, "output_dim", None)
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError("Custom encoder requires encoder_output_dim > 0")
    return encoder, dim
