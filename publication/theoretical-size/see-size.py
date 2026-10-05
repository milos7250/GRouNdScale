import json

import torch

FILE = 'paper-v3/results/checkpoints/step_50000.pth'
# FILE = '../../upstream/GRouNdGAN/paper-v2/results/GRouNdGAN/checkpoints/step_500000.pth'

with open(FILE, 'rb') as f:
    data = torch.load(f, map_location=torch.device('cpu'))


def get_key_name(key):
    return key
    if isinstance(key, str):
        if "weight" in key:
            return "weight"
        elif "indices" in key:
            return "indices"
        elif "bias" in key:
            return "bias"
        elif "mean" in key:
            return "mean"
        elif "var" in key:
            return "var"
        elif "num_batches_tracked" in key:
            return "num_batches_tracked"
    if isinstance(key, int):
        return "int"
    return key

def deep_merge_dicts(dict1, dict2):
    merged_dict = dict1.copy()
    for key, value in dict2.items():
        if key in merged_dict:
            if isinstance(merged_dict[key], dict) and isinstance(value, dict):
                merged_dict[key] = deep_merge_dicts(merged_dict[key], value)
            else:
                merged_dict[key] += value
        else:
            merged_dict[key] = value
    return merged_dict

def total_dict_size(d):
    total_size = 0
    for value in d.values():
        if isinstance(value, dict):
            total_size += total_dict_size(value)
        else:
            total_size += value
    return total_size

def get_size(obj, factor=1024 * 1024):
    if isinstance(obj, torch.Tensor):
        return obj.element_size() * obj.nelement() / factor
    elif isinstance(obj, dict):
        size = {}
        for key, value in obj.items():
            key = get_key_name(key)
            if isinstance(value, dict):
                size[key] = deep_merge_dicts(size.get(key, {}), get_size(value, factor))
            elif isinstance(value, torch.Tensor):
                size[key] = size.get(key, 0) + value.element_size() * value.nelement() / factor
        return size
    return 0

def get_shape(obj):
    if isinstance(obj, torch.Tensor):
        return obj.shape
    elif isinstance(obj, dict):
        shape = {}
        for key, value in obj.items():
            key = get_key_name(key)
            if isinstance(value, dict):
                shape[key] = get_shape(value)
            elif isinstance(value, torch.Tensor):
                shape[key] = shape.get(key, {})
                shape[key][str(value.shape)] = shape[key].get(str(value.shape), 0) + 1
        return shape
    return None

def total_params(d):
    total_params = 0
    for value in d.values():
        if isinstance(value, dict):
            total_params += total_params(value)
        elif isinstance(value, torch.Tensor):
            total_params += value.numel()
    return total_params

if "generator_state_dict" in data:
    data["gen_state_dict"] = data.pop("generator_state_dict")
    data["crit_state_dict"] = data.pop("critic_state_dict")

print(f"Keys in the checkpoint: {list(data.keys())}")
data = {k: v for k, v in data.items() if k in ["gen_state_dict", "crit_state_dict", "labeler_state_dict", "antilabeler_state_dict"]}

# Extract causal controller from generator state dict
data["causal_controller"] = {k: v for k, v in data["gen_state_dict"].items() if k.startswith("_causal_controller")}
data["gen_state_dict"] = {k: v for k, v in data["gen_state_dict"].items() if not k.startswith("_causal_controller")}

# Remove spare labeller
data['gen_state_dict'] = {k: v for k, v in data['gen_state_dict'].items() if not k.startswith('_labeler')}

for key, value in data.items():
    memory = get_size(value)
    size = get_shape(value)
    # print(f"{key}: {json.dumps(memory, indent=2)} MiB")
    # print(f"{key} shape: {json.dumps(size, indent=2)}")
    print(f"total: {total_dict_size(memory)} MiB")
    print(f"total params: {total_params(value)}")