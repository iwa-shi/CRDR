from typing import Optional, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class RateLoss(nn.Module):
    def __init__(self, loss_weight: float):
        super().__init__()
        self.lamb_rate = loss_weight

    def forward(self, bpp, **kwargs):
        bpp = torch.mean(bpp)
        return self.lamb_rate * bpp


@LOSS_REGISTRY.register()
class HificRateLoss(nn.Module):
    def __init__(
        self,
        lambda_A: float,
        lambda_B: float,
        target_rate: float,
        lambda_schedule: Optional[Dict] = None,
        target_rate_schedule: Optional[Dict[str, List]] = None,
    ) -> None:
        """Dynamic rate loss used in HiFiC

        Args:
            lambda_A (float): applied when rate is higher than target
            lambda_B (float): applied when rate is lower than target
            target_rate (float):
            lambda_schedule (Optional[Dict], optional): 'steps': [List], 'vals': [List]
            target_rate_schedule (Optional[Dict], optional): 'steps': [List], 'vals': [List]
        """
        super().__init__()
        assert (
            lambda_A > lambda_B
        ), f"Expected lambda_A > lambda_B, got (A) {lambda_A} <= (B) {lambda_B}"
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.target_rate = target_rate

        self._check_schedule(lambda_schedule)
        self._check_schedule(target_rate_schedule)
        self.lambda_schedule = lambda_schedule
        self.target_rate_schedule = target_rate_schedule

    @staticmethod
    def _check_schedule(schedule: Optional[Dict[str, List]] = None) -> None:
        if schedule is None:
            return
        assert isinstance(
            schedule, dict
        ), f"schedule must be dict, but {type(schedule)}"
        assert "vals" in schedule, 'schedule must have "vals" as a key'
        assert "steps" in schedule, 'schedule must have "steps" as a key'
        vals = schedule["vals"]
        steps = schedule["steps"]
        assert isinstance(
            vals, list
        ), f'schedule["vals"] must be list, but {type(vals)}'
        assert isinstance(
            steps, list
        ), f'schedule["steps"] must be list, but {type(steps)}'
        assert (
            len(vals) == len(steps) + 1
        ), f"Requirement: len(vals) = len(steps)+1, but {len(vals)} vs {len(steps)}"

    @staticmethod
    def get_scheduled_params(
        param: float, param_schedule: Dict, step_counter: int
    ) -> float:
        vals, steps = param_schedule["vals"], param_schedule["steps"]
        idx = np.where(step_counter < np.array(steps + [step_counter + 1]))[0][0]
        param *= vals[idx]
        return param

    def forward(
        self, bpp: torch.Tensor, qbpp: torch.Tensor, current_iter: int, **kwargs
    ) -> torch.Tensor:
        lambda_A, lambda_B = self.lambda_A, self.lambda_B
        if self.lambda_schedule:
            lambda_A = self.get_scheduled_params(
                lambda_A, self.lambda_schedule, current_iter
            )
            lambda_B = self.get_scheduled_params(
                lambda_B, self.lambda_schedule, current_iter
            )

        target_bpp = self.target_rate
        if self.target_rate_schedule:
            target_bpp = self.get_scheduled_params(
                target_bpp, self.target_rate_schedule, current_iter
            )

        qbpp = torch.mean(qbpp.detach()).item()
        weight = (
            lambda_A if qbpp > target_bpp else lambda_B
        )
        return weight * torch.mean(bpp)


@LOSS_REGISTRY.register()
class HificVariableRateLoss(HificRateLoss):
    def __init__(
        self,
        lambda_A: List[float],
        lambda_B: Union[List[float], float],
        target_rate: List[float],
        lambda_schedule: Optional[Dict] = None,
        target_rate_schedule: Optional[Dict[str, List]] = None,
    ) -> None:
        super(HificRateLoss, self).__init__()
        if isinstance(lambda_B, float):
            lambda_B = [lambda_B] * len(lambda_A)
        self.check_lambda_target(lambda_A, lambda_B, target_rate)
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.target_rate = target_rate

        self._check_schedule(lambda_schedule)
        self._check_schedule(target_rate_schedule)
        self.lambda_schedule = lambda_schedule
        self.target_rate_schedule = target_rate_schedule

    @staticmethod
    def check_lambda_target(lambda_A, lambda_B, target_rate):
        assert len(lambda_A) == len(lambda_B)
        assert len(lambda_A) == len(target_rate)

        target_rate_ = sorted(target_rate)
        assert target_rate == target_rate_
        lambda_A_ = sorted(lambda_A, reverse=True)
        assert lambda_A_ == lambda_A

        for i, (a, b) in enumerate(zip(lambda_A, lambda_B)):
            assert (
                a > b
            ), f"Expected lambda_A > lambda_B, got (A[{i}]) {a} <= (B[{i}]) {b}"

    def forward(
        self,
        bpp: torch.Tensor,
        qbpp: torch.Tensor,
        current_iter: int,
        rate_ind: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        assert rate_ind.numel() == 1
        _rate_ind = rate_ind.long().item()
        lambda_A, lambda_B = self.lambda_A[_rate_ind], self.lambda_B[_rate_ind]
        if self.lambda_schedule:
            lambda_A = self.get_scheduled_params(
                lambda_A, self.lambda_schedule, current_iter
            )
            lambda_B = self.get_scheduled_params(
                lambda_B, self.lambda_schedule, current_iter
            )

        target_bpp = self.target_rate[rate_ind]
        if self.target_rate_schedule:
            target_bpp = self.get_scheduled_params(
                target_bpp, self.target_rate_schedule, current_iter
            )

        qbpp = torch.mean(qbpp.detach()).item()
        weight = (
            lambda_A if qbpp > target_bpp else lambda_B
        )
        return weight * torch.mean(bpp)



