"""CTM + GBDT model serialization utilities.

Includes save/load for individual models and full ensemble trainer state.
"""

from __future__ import annotations
import dataclasses
import json
import os
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import yaml


def save_ctm_model(
    model: nn.Module,
    save_dir: str,
    model_name: str = "ctm_model",
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[str]]:
    """Save CTM model state dict + optional config.

    Returns
    -------
    (state_dict_path, config_path)
    """
    os.makedirs(save_dir, exist_ok=True)
    state_path = os.path.join(save_dir, f"{model_name}.pt")
    torch.save(model.state_dict(), state_path)
    cfg_path = None
    if config is not None:
        cfg_path = os.path.join(save_dir, f"{model_name}_config.yaml")
        with open(cfg_path, "w") as f:
            yaml.dump(config, f)
    return state_path, cfg_path


def load_ctm_model(
    model_class: type,
    model_params: Dict[str, Any],
    state_path: str,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Load CTM model state dict into a fresh model instance.

    Returns model with loaded weights, on specified device.
    """
    model = model_class(**model_params).to(device)
    state = torch.load(state_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model


def save_gbdt_model(
    gbdt_model: Any,
    save_dir: str,
    model_name: str = "gbdt_model",
) -> str:
    """Save GBDT model as JSON via Hoffnung's to_json().

    Supports both raw C++ GBDT and GBDTTrainer wrapper objects.

    Returns path to saved JSON file.
    """
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, f"{model_name}.json")

    # Try raw GBDT (C++ pybind) first, then GBDTTrainer wrapper
    if hasattr(gbdt_model, "to_json"):
        json_str = gbdt_model.to_json()
    elif hasattr(gbdt_model, "_model") and hasattr(gbdt_model._model, "to_json"):
        json_str = gbdt_model._model.to_json()
    else:
        raise TypeError(
            f"gbdt_model has no to_json() method: {type(gbdt_model)}"
        )

    with open(json_path, "w") as f:
        f.write(json_str)
    return json_path


def load_gbdt_model(
    gbdt_model_class: type,
    gbdt_config_class: Optional[type] = None,
    json_path: str = "",
    config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Load GBDT model from JSON file.

    Returns GBDT model instance with loaded trees.
    """
    if gbdt_config_class is not None:
        model = gbdt_model_class(gbdt_config_class())
    elif config is not None:
        model = gbdt_model_class()
        for k, v in config.items():
            if hasattr(model, k):
                setattr(model, k, v)
    else:
        raise ValueError("load_gbdt_model requires gbdt_config_class or config dict")
    with open(json_path) as f:
        json_str = f.read()
    model.from_json(json_str)
    return model


def save_ensemble(
    ctm_model: nn.Module,
    gbdt_model: Any,
    save_dir: str,
    model_name: str = "ensemble",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Save both CTM and GBDT models + optional config.

    Returns dict of saved paths.
    """
    saved = {}
    ckpt_path, cfg_path = save_ctm_model(ctm_model, save_dir, f"{model_name}_ctm", config)
    saved["ctm_state_dict"] = ckpt_path
    if cfg_path:
        saved["config"] = cfg_path
    saved["gbdt_json"] = save_gbdt_model(gbdt_model, save_dir, f"{model_name}_gbdt")
    return saved


def save_ensemble_trainer_state(
    ctm_model: nn.Module,
    gbdt_model: Any,
    save_dir: str,
    model_params: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    loss_config: Optional[Dict[str, Any]] = None,
    gbdt_config: Optional[Dict[str, Any]] = None,
    gbdt_loss: str = "mse",
    feature_importance: Optional[Dict[str, Any]] = None,
    model_name: str = "ensemble",
) -> Dict[str, str]:
    """Save full ensemble trainer state for resume or inference.

    Saves CTM state_dict, GBDT JSON, and all configuration needed to
    reconstruct the ensemble trainer.  The saved ``.pt`` file can be
    loaded by ``load_ensemble_trainer_state()`` for inference-only
    fusion or warm-start training.

    Parameters
    ----------
    ctm_model : CTM model instance.
    gbdt_model : GBDT model instance with ``to_json()``.
    save_dir : Output directory (created if missing).
    model_params : CTM model constructor kwargs (required for reload).
    config : Full training config dict.
    loss_config : LossConfig dataclass or dict.
    gbdt_config : GBDT hyperparameter dict.
    gbdt_loss : GBDT loss type string.
    feature_importance : Optional importance dict from ensemble trainer.
    model_name : Prefix for saved files.

    Returns
    -------
    dict with keys: ``"ctm_state_dict"``, ``"gbdt_json"``, ``"config"``.
    """
    os.makedirs(save_dir, exist_ok=True)
    saved: Dict[str, str] = {}

    # Save CTM state_dict
    ckpt_path, cfg_path = save_ctm_model(ctm_model, save_dir, f"{model_name}_ctm", config)
    saved["ctm_state_dict"] = ckpt_path
    if cfg_path:
        saved["config"] = cfg_path

    # Save GBDT JSON
    saved["gbdt_json"] = save_gbdt_model(gbdt_model, save_dir, f"{model_name}_gbdt")

    # Save full trainer state bundle
    state: Dict[str, Any] = {
        "ctm_state_dict": ctm_model.state_dict(),
    }
    if hasattr(gbdt_model, "to_json"):
        state["gbdt_json_str"] = gbdt_model.to_json()
    elif hasattr(gbdt_model, "_model") and hasattr(gbdt_model._model, "to_json"):
        state["gbdt_json_str"] = gbdt_model._model.to_json()
    if model_params is not None:
        state["model_params"] = model_params
    if config is not None:
        state["config"] = config
    if loss_config is not None:
        if dataclasses.is_dataclass(loss_config):
            state["loss_config"] = {
                f.name: getattr(loss_config, f.name)
                for f in dataclasses.fields(loss_config)
            }
        else:
            state["loss_config"] = dict(loss_config)
    if gbdt_config is not None:
        state["gbdt_config"] = gbdt_config
    state["gbdt_loss"] = gbdt_loss
    if feature_importance is not None:
        state["feature_importance"] = feature_importance

    state_path = os.path.join(save_dir, f"{model_name}_trainer_state.pt")
    torch.save(state, state_path)
    saved["trainer_state"] = state_path
    return saved


def load_ensemble_trainer_state(
    state_path: str,
    map_location: torch.device | str = "cpu",
) -> Dict[str, Any]:
    """Load full ensemble trainer state from a saved ``.pt`` bundle.

    Parameters
    ----------
    state_path : Path to ``*_trainer_state.pt`` created by
        ``save_ensemble_trainer_state()``.
    map_location : Device for CTM state_dict loading.

    Returns
    -------
    dict with keys: ``"ctm_state_dict"`` (OrderedDict),
    ``"gbdt_json_str"`` (str), ``"model_params"`` (dict | None),
    ``"config"`` (dict | None), ``"loss_config"`` (dict | None),
    ``"gbdt_config"`` (dict | None), ``"gbdt_loss"`` (str).
    """
    state = torch.load(state_path, map_location=map_location, weights_only=True)
    result: Dict[str, Any] = {
        "ctm_state_dict": state.get("ctm_state_dict"),
        "gbdt_json_str": state.get("gbdt_json_str", ""),
        "model_params": state.get("model_params"),
        "config": state.get("config"),
        "loss_config": state.get("loss_config"),
        "gbdt_config": state.get("gbdt_config"),
        "gbdt_loss": state.get("gbdt_loss", "mse"),
        "feature_importance": state.get("feature_importance"),
    }
    return result
