"""Export a frozen learned actor and one actual closed-loop trace for MATLAB."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--environment", type=int, default=0)
    args = parser.parse_args()
    checkpoint = torch.load(args.folder/"checkpoint.pt", weights_only=True, map_location="cpu")
    state = checkpoint["actor"]
    trace = np.load(args.folder/"trace.npz")
    result = {}
    for number, index in enumerate([0, 2, 4], start=1):
        result[f"W{number}"] = state[f"net.{index}.weight"].double().numpy()
        result[f"b{number}"] = state[f"net.{index}.bias"].double().numpy()[:, None]
    env = args.environment
    for name in ["command", "disturbance", "state", "action"]:
        value = trace[name][:, env].astype(np.float64)
        if value.ndim == 1:
            value = value[:, None]
        result[{"state": "state_expected", "action": "action_expected"}.get(name, name)] = value
    result["initial_state"] = trace["initial_state"][env].astype(np.float64)[:, None]
    result["dt"] = 0.02
    result["response_raw"] = checkpoint["mean"].double().numpy()
    destination = args.folder/"matlab_export.mat"
    savemat(destination, result)
    print(destination.resolve())


if __name__ == "__main__":
    main()
