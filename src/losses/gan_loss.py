from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class VanillaGANLoss(nn.Module):
    def __init__(self, loss_weight: float, real_label: float=1.0, fake_label: float=0.0, loss_reduction: str='mean'):
        super().__init__()
        self.lamb_gan = loss_weight
        self.loss = nn.BCEWithLogitsLoss(reduction=loss_reduction)
        self.real_label_val = real_label
        self.fake_label_val = fake_label

    def get_label_like_x(self, x: torch.Tensor, is_real: bool):
        if is_real:
            if not(hasattr(self, 'real_label')):
                self.real_label = torch.tensor([self.real_label_val], device=x.device, dtype=x.dtype)
            return self.real_label.expand_as(x)
        
        if not(hasattr(self, 'fake_label')):
            self.fake_label = torch.tensor([self.fake_label_val], device=x.device, dtype=x.dtype)
        return self.fake_label.expand_as(x)

    def forward(self, x, is_real: bool, is_disc: bool=False, **kwargs):
        label = self.get_label_like_x(x, is_real)
        loss = self.loss(x, label)
        return loss if is_disc else self.lamb_gan * loss


@LOSS_REGISTRY.register()
class MaskedVanillaGANLoss(VanillaGANLoss):
    def __init__(self, loss_weight: float, real_label: float=1.0, fake_label: float=0.0):
        """
        For RectifiedGAN
        """
        super().__init__(loss_weight, real_label, fake_label, loss_reduction='none')

    def forward(self,
                x: torch.Tensor,
                is_real: bool,
                mask: Optional[torch.Tensor]=None,
                is_disc: bool=False,
                **kwargs):
        label = self.get_label_like_x(x, is_real)
        loss = self.loss(x, label)
        if isinstance(mask, torch.Tensor):
            loss = loss * mask
        loss = torch.mean(loss)
        return loss if is_disc else self.lamb_gan * loss
    


@LOSS_REGISTRY.register()
class MultiscaleVanillaGANLoss(nn.Module):
    def __init__(self, loss_weight: float):
        super().__init__()
        self.lamb_gan = loss_weight
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.real_label_list = []
        self.fake_label_list = []

    def forward(self, x, is_real: bool, is_disc: bool=False, **kwargs):
        assert isinstance(x, list)
        num_scale = len(x)
        loss = 0.

        for i, feat in enumerate(x):
            if is_real:
                if len(self.real_label_list) < num_scale:
                    self.real_label_list.append(torch.ones_like(feat))
                loss += self.loss(feat, self.real_label_list[i])
            else:
                if len(self.fake_label_list) < num_scale:
                    self.fake_label_list.append(torch.zeros_like(feat))
                loss += self.loss(feat, self.fake_label_list[i])
        loss = loss / num_scale
        return loss if is_disc else self.lamb_gan * loss


@LOSS_REGISTRY.register()
class HingeGANLoss(nn.Module):
    def __init__(self, loss_weight: float) -> None:
        super().__init__()
        self.lamb_gan = loss_weight
        self.relu = nn.ReLU()

    def forward(self, x, is_real: bool, is_disc: bool=False, **kwargs):
        ## For D
        if is_disc:
            if is_real:
                loss = self.relu(1 - x)
            else: ## fake
                loss = self.relu(1 + x)
            return torch.mean(loss)

        ## For G
        assert is_real, 'For G loss `is_real` should be True'
        loss = -torch.mean(x)
        return self.lamb_gan * loss


@LOSS_REGISTRY.register()
class MultiscaleHingeGANLoss(HingeGANLoss):
    def forward(self, x, is_real: bool, is_disc: bool=False, **kwargs):
        loss_list = []
        for feat in x:
            loss_list.append(super().forward(feat, is_real=is_real, is_disc=is_disc))
        return sum(loss_list)
