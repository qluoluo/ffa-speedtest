from pathlib import Path
from typing import Optional

import torch


def load_qkvh(load_dir: str, device="cpu", start_layer: int = 0, max_length: Optional[int] = None):
    """
    Load q/k/v/h data from layer_* folders.
    """
    load_root = Path(load_dir)
    dirname_list = sorted(
        [x for x in load_root.iterdir() if x.is_dir() and x.name.startswith("layer")],
        key=lambda x: int(x.name.split("_")[1]),
    )
    layer_num = len(dirname_list)
    assert [p.name for p in dirname_list] == [
        f"layer_{i}" for i in range(layer_num)
    ], "Layer directories must be named layer_0, layer_1, ..."
    if not (0 <= start_layer < layer_num):
        raise ValueError(f"start_layer must be in [0, {layer_num - 1}], got {start_layer}")

    def _truncate_tensor(t: torch.Tensor):
        if max_length is None or max_length <= 0:
            return t
        if t.dim() >= 3:
            return t[..., :max_length, :]
        if t.dim() == 2:
            return t[:, :max_length]
        return t

    for i in range(start_layer, layer_num):
        layer_dir = load_root / f"layer_{i}"
        load_data_list = ["q_rope", "k_rope", "q_unrope", "k_unrope", "v", "h"]
        data = {}
        for data_name in load_data_list:
            data_path = layer_dir / f"{data_name}.pt"
            tensor = torch.load(data_path, weights_only=True, map_location=device)
            data[data_name] = _truncate_tensor(tensor)
        yield data
