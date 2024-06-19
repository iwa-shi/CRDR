from typing import Dict, Tuple

import torch
import torch.nn as nn
import numpy as np

from src.utils.registry import TRAINER_REGISTRY
from .gan_rate_distortion_trainer import GANRateDistortionTrainer

@TRAINER_REGISTRY.register()
class RaGANRateDistortionTrainer(GANRateDistortionTrainer):
    def optimize_parameters(self, current_iter: int, data_dict: Dict):
        ###################################################################
        #                             Train G                             
        ###################################################################
        self.discriminator.requires_grad_(False)
        self.g_optimizer.zero_grad()
        if self.aux_optimizer:
            self.aux_optimizer.zero_grad()

        # run model
        real_images, fake_images, bpp, other_outputs = self.run_comp_model(data_dict)

        # calculate losses
        g_loss_dict = {}
        g_loss_dict['distortion'] = self.distortion_loss(real_images, fake_images, **other_outputs)
        g_loss_dict['rate'] = self.rate_loss(bpp, **other_outputs, current_iter=current_iter)
        if self.perceptual_loss:
            g_loss_dict['perceptual'] = self.perceptual_loss(real_images, fake_images)

        # RaGAN adv loss
        real_d_pred = self.discriminator(real_images, **other_outputs).detach()
        fake_g_pred = self.discriminator(fake_images, **other_outputs)

        l_g_real = self.gan_loss(real_d_pred - torch.mean(fake_g_pred), is_real=False, is_disc=False)
        l_g_fake = self.gan_loss(fake_g_pred - torch.mean(real_d_pred), is_real=True, is_disc=False)
        g_loss_dict['adv'] = (l_g_real + l_g_fake) / 2

        l_total = sum(_v for _v in g_loss_dict.values())

        # For stability
        if (loss_anomaly := self.check_loss_nan_inf(l_total)):
            self.logger.warning(f'iter{current_iter}: skipped because loss is {loss_anomaly}')
            return # skip back-propagation part
        
        # back prop & update parameters
        l_total.backward()
        if self.opt.optim.get('clip_max_norm'):
            nn.utils.clip_grad_norm_(self.comp_model.parameters(), self.opt.optim.clip_max_norm)
        self.g_optimizer.step()

        log_dict = dict(**g_loss_dict, qbpp=other_outputs.get('qbpp', -1))

        if self.aux_optimizer:
            aux_loss = self.comp_model.aux_loss()
            aux_loss.backward()
            log_dict['aux'] = aux_loss
            self.aux_optimizer.step()

        if self.g_scheduler:
            self.g_scheduler.step()

        ###################################################################
        #                             Train D                             
        ###################################################################
        self.discriminator.requires_grad_(True)
        self.d_optimizer.zero_grad()

        # real
        fake_d_pred = self.discriminator(fake_images, **other_outputs).detach()
        real_d_pred = self.discriminator(real_images, **other_outputs)
        l_d_real = self.gan_loss(real_d_pred - torch.mean(fake_d_pred), is_real=True, is_disc=True) * 0.5
        l_d_real.backward()

        # fake
        fake_d_pred = self.discriminator(fake_images.detach(), **other_outputs)
        l_d_fake = self.gan_loss(fake_d_pred - torch.mean(real_d_pred.detach()), is_real=False, is_disc=True) * 0.5
        l_d_fake.backward()

        log_dict.update({
            'd_real': l_d_real,
            'd_fake': l_d_fake,
            'd_total': l_d_real + l_d_fake,
            'out_d_real': torch.mean(real_d_pred.detach()),
            'out_d_fake': torch.mean(fake_d_pred.detach()),
        })

        self.d_optimizer.step()
        if self.d_scheduler:
            self.d_scheduler.step()

        return log_dict