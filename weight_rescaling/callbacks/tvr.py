from collections import defaultdict

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

from weight_rescaling.utils.utils import (
    compute_fsdp_global_mean_std,
    get_layers,
    is_fsdp_model,
)


class TVRCallback(Callback):
    def __init__(
        self,
        valid_2d_module_names: list,
        algo: str = "zscore",
        target_std: float = 0.02,
        scale_with_depth: bool = False,
        layer_path: str = "model.layers",
        ratio_threshold: float = 1.2,
        step_interval: int = 50,
        log_granularity: str = "layer",
        compute_global_mean_std: bool = True,
        module_blacklist: list | None = None,
        additional_module_names_to_log: list | None = None,
        verbose: bool = False,
    ):
        """
        Target Variance Rescaling (TVR) callback.
        Args:
            valid_2d_module_names (list): List of valid 2D module names.
            algo (str, optional): Algorithm to use for weight rescaling. Default is "zscore".
            target_std (float, optional): Target standard deviation for rescaling. Default is 0.02.
            scale_with_depth (bool, optional): Whether to scale with depth. Default is False.
            layer_path (str, optional): Path to the model layers. Default is "model.layers".
            ratio_threshold (float, optional): Threshold for the ratio. Default is 1.2.
            step_interval (int, optional): Interval of global steps (not tokens) for rescaling. Default is 50. 
            log_granularity (str, optional): Granularity of logging. Options are "agg", "layer", or None. Default is "layer".
            compute_global_mean_std (bool, optional): Whether to compute global mean and standard deviation. Default is True. If False, it will use the local module's mean and standard deviation.
            module_blacklist (list, optional): List of modules to blacklist. Default is None. For example, ["self_attn.q_proj","self_attn.k_proj"]
            additional_module_names_to_log (list, optional): Additional module names to log. Default is None.
            verbose (bool, optional): Whether to enable verbose logging. Default is False.
        """

        assert algo in ["zscore"], f"Invalid algorithm: {algo}. Only 'zscore' are supported."
        assert log_granularity in [
            "agg",
            "layer",
            None,
        ], f"Invalid log granularity: {log_granularity}. Only 'agg', 'layer', and None are supported."

        self.valid_2d_module_names = valid_2d_module_names
        self.algo = algo
        self.target_std = target_std
        self.ratio_threshold = ratio_threshold
        self.step_interval = step_interval
        self.scale_with_depth = scale_with_depth
        self.layer_path = layer_path
        self.log_granularity = log_granularity
        self.compute_global_mean_std = compute_global_mean_std
        self.verbose = verbose

        if additional_module_names_to_log is None:
            self.additional_module_names_to_log = []
        else:
            self.additional_module_names_to_log = additional_module_names_to_log

        if not module_blacklist:
            module_blacklist = []
        self.module_blacklist = module_blacklist

        self.last_check_global_step = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (
            self.last_check_global_step == 0 and trainer.global_step > 0
        ):  # handle resume from checkpoint
            self.last_check_global_step = trainer.global_step

        if (
            not trainer.fit_loop._should_accumulate()
            and trainer.global_step - self.last_check_global_step >= self.step_interval
        ):
            self.rescale_weight(pl_module.model, trainer)
            self.last_check_global_step = trainer.global_step

    def rescale_weight(self, model, trainer):
        if self.algo == "zscore":
            layer_container = get_layers(model, self.layer_path)
            for layer_index, layer in enumerate(layer_container, start=1):
                if is_fsdp_model(model):
                    layer = layer._fsdp_wrapped_module

                if self.scale_with_depth:
                    depth_scaler = 1 / (layer_index**0.5)
                else:
                    depth_scaler = 1.0

                target_std_at_layer = self.target_std * depth_scaler

                valid_2d_modules = [
                    (name, module)
                    for name, module in layer.named_modules()
                    if name in self.valid_2d_module_names + self.additional_module_names_to_log
                ]

                if self.log_granularity is not None:
                    metrics_dict = defaultdict(list)

                for name, module in valid_2d_modules:
                    if is_fsdp_model(model) and self.compute_global_mean_std:
                        ori_mean, ori_std = compute_fsdp_global_mean_std(module)
                    else:
                        ori_std = module.weight.std()
                        ori_mean = module.weight.mean()

                    # Only rescale if the ratio is above the threshold and not blacklisted
                    is_rescaled = False
                    if name not in self.module_blacklist and name in self.valid_2d_module_names:
                        if ori_std / target_std_at_layer > self.ratio_threshold:
                            rescaled_module = ((module.weight - ori_mean) / ori_std) * (
                                target_std_at_layer
                            ) + ori_mean
                            module.weight.data.copy_(rescaled_module)

                            # If has bias, set it to zero
                            if module.bias is not None:
                                module.bias.data.zero_()

                            is_rescaled = True

                    if self.log_granularity == "layer":
                        trainer.callback_metrics[
                            f"weight/weight_mean_{name}/layer_{layer_index}"
                        ] = torch.tensor([ori_mean])
                        trainer.callback_metrics[
                            f"weight/weight_std_{name}/layer_{layer_index}"
                        ] = torch.tensor([ori_std])
                        trainer.callback_metrics[
                            f"weight/weight_std_ratio_{name}/layer_{layer_index}"
                        ] = torch.tensor([ori_std / target_std_at_layer])
                        trainer.callback_metrics[
                            f"weight/is_rescaled_{name}/layer_{layer_index}"
                        ] = torch.tensor([int(is_rescaled)])
                    elif self.log_granularity == "agg":
                        metrics_dict[f"weight_mean_{name}"].append(ori_mean.item())
                        metrics_dict[f"weight_std_{name}"].append(ori_std.item())
                        metrics_dict[f"weight_std_ratio_{name}"].append(
                            ori_std.item() / target_std_at_layer
                        )
                        metrics_dict[f"is_rescaled_{name}"].append(int(is_rescaled))

                if self.log_granularity == "agg":
                    for name, values in metrics_dict.items():
                        trainer.callback_metrics[f"weight/{name}"] = np.array(values)

            layer_related_module_names = [name for name, module in layer.named_modules() if name != ""]
            for name, module in model.named_modules():
                if all(x not in name for x in layer_related_module_names):
                    if name in self.additional_module_names_to_log:
                        if is_fsdp_model(model):
                            ori_mean, ori_std = compute_fsdp_global_mean_std(module)
                        else:
                            ori_std = module.weight.std()
                            ori_mean = module.weight.mean()

                        trainer.callback_metrics[f"weight/weight_mean_{name}"] = torch.tensor(
                            [ori_mean]
                        )
                        trainer.callback_metrics[f"weight/weight_std_{name}"] = torch.tensor(
                            [ori_std]
                        )
