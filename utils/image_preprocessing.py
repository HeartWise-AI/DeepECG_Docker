import numpy as np
import torch
from PIL import Image


def load_ecg_image(path: str, preprocessing: dict) -> torch.Tensor:
    """
    Load an ECG page image and preprocess it exactly as DeepECG-PNG was trained
    (DeepECG-PNG utils.image.process_image, no augmentation):

    1. convert to grayscale ("L"; color_mode "rgb" keeps RGB),
    2. bilinear resize to (height, width), stretching the page (torchvision's v2.Resize
       on a PIL image is Pillow's BILINEAR resize),
    3. three identical channels (v2.Grayscale(num_output_channels=3)),
    4. uint8 -> float32 in [0, 1] (v2.ToDtype(scale=True) multiplies by 1/255),
    5. per-channel (x - mean) / std (v2.Normalize).

    Args:
        path: Path to the PNG (any format Pillow reads).
        preprocessing: The weights' config.json "preprocessing" block
            (resize [h, w], color_mode, mean [3], std [3]).

    Returns:
        Float32 tensor of shape (3, height, width).
    """
    height, width = (int(v) for v in preprocessing["resize"])
    gray = str(preprocessing.get("color_mode", "grayscale")).lower().startswith("gray")
    with Image.open(path) as image:
        image = image.convert("L" if gray else "RGB")
        image = image.resize((width, height), Image.BILINEAR)
        array = np.asarray(image, dtype=np.uint8)
    array = np.repeat(array[None], 3, axis=0) if gray else array.transpose(2, 0, 1)
    tensor = torch.from_numpy(np.ascontiguousarray(array)).to(torch.float32).mul_(1.0 / 255)
    mean = torch.tensor(preprocessing["mean"], dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor(preprocessing["std"], dtype=torch.float32).view(-1, 1, 1)
    return tensor.sub_(mean).div_(std)
