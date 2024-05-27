from typing import Dict, Tuple, Optional

import os
from copy import deepcopy

import torch
import torch.nn as nn

from src.losses import build_loss
from src.utils.logger import IndentedLog, log_dict_items
from src.utils.registry import TRAINER_REGISTRY
from src.utils.path import PathHandler
from .base_trainer import BaseTrainer
from .optimizer import build_optimizer, build_scheduler

@TRAINER_REGISTRY.register()
class RateDistortionTrainer(BaseTrainer):
    def _set_losses(self):
        loss_opt = deepcopy(self.opt.loss)
        self.distortion_loss = build_loss(loss_opt.distortion_loss, loss_name='distortion_loss')
        self.rate_loss = build_loss(loss_opt.rate_loss, loss_name='rate_loss')

        if loss_opt.get('perceptual_loss'):
            self.perceptual_loss = build_loss(loss_opt.perceptual_loss, loss_name='perceptual_loss').to(self.device)
        else:
            self.logger.warn('perceptual_loss is NOT build')
            self.perceptual_loss = None

    def _set_optimizer_scheduler(self):
        parameters_dict, aux_parameters_dict = self.comp_model.separete_aux_parameters()

        # set g_optimizer
        optim_opt = deepcopy(self.opt.optim)
        with IndentedLog(level="INFO", msg='building g_optimizer'):
            self.g_optimizer = build_optimizer(parameters_dict, optim_opt.g_optimizer)
            if optim_opt.get('g_scheduler'):
                self.g_scheduler = build_scheduler(self.g_optimizer, optim_opt.g_scheduler)
            else:
                self.logger.warn('g_scheduler is NOT build')
                self.g_scheduler = None

        # set aux_optimizer
        if len(aux_parameters_dict) > 0:
            with IndentedLog(level="INFO", msg='building aux_optimizer'):
                self.aux_optimizer = build_optimizer(aux_parameters_dict, optim_opt.aux_optimizer)
        else:
            self.logger.warn('aux_optimizer is NOT build.')
            self.aux_optimizer = None

    def run_comp_model(self, data_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        out_dict = self.comp_model.run_model(**data_dict)
        real_images = out_dict.pop('real_images')
        fake_images = out_dict.pop('fake_images')
        bpp = out_dict.pop('bpp')
        return real_images, fake_images, bpp, out_dict

    def optimize_parameters(self, current_iter: int, data_dict: Dict):
        log_dict = {}

        self.g_optimizer.zero_grad()
        if self.aux_optimizer:
            self.aux_optimizer.zero_grad()

        real_images, fake_images, bpp, other_outputs = self.run_comp_model(data_dict)
        log_dict['qbpp'] = other_outputs.get('qbpp', -1)
        
        # calculate losses
        g_loss_dict = {}
        g_loss_dict['distortion'] = self.distortion_loss(real_images, fake_images, **other_outputs)
        g_loss_dict['rate'] = self.rate_loss(bpp, **other_outputs, current_iter=current_iter)
        if self.perceptual_loss:
            g_loss_dict['perceptual'] = self.perceptual_loss(real_images, fake_images)

        l_total = sum(_v for _v in g_loss_dict.values())

        # For stability
        if (loss_anomaly := self.check_loss_nan_inf(l_total)):
            self.logger.warning(f'iter{current_iter}: skipped because loss is {loss_anomaly}')
            return # skip back-propagation part
        
        # back prop & update parameters
        l_total.backward()
        if self.opt.optim.get('clip_max_norm', None):
            nn.utils.clip_grad_norm_(self.comp_model.parameters(), self.opt.optim.clip_max_norm)
        self.g_optimizer.step()

        log_dict.update(g_loss_dict)

        if self.g_scheduler:
            self.g_scheduler.step()

        if self.aux_optimizer:
            log_dict['aux'] = self.optimize_aux_parameters()

        return log_dict
    
    def optimize_aux_parameters(self):
        aux_loss = self.comp_model.aux_loss()
        aux_loss.backward()
        self.aux_optimizer.step()
        return aux_loss

    def save(self, current_iter: int):
        # save model
        self.model_saver.save({'comp_model': self.comp_model}, 'comp_model', current_iter, keep=True)
        
        # save training_state
        optimizer_scheduler_dict = {'g_optimizer': self.g_optimizer}
        if self.aux_optimizer:
            optimizer_scheduler_dict['aux_optimizer'] = self.aux_optimizer
        if self.g_scheduler:
            optimizer_scheduler_dict['g_scheduler'] = self.g_scheduler
        self.model_saver.save(optimizer_scheduler_dict, 'training_state', current_iter, keep=self.opt.get('keep_training_state', False))

    def _load_checkpoint(self,
                         exp: str,
                         itr: int,
                         load_optimizer: bool=True,
                         load_scheduler: bool=True,
                         new_g_lr: Optional[float]=None,
                         strict: bool=True,
                         **kwargs) -> None:
        ## get checkpoint path
        path_handler = PathHandler(self.opt.path.ckpt_dir, exp)
        model_ckpt_path = path_handler.get_ckpt_path('comp_model', itr)
        assert os.path.exists(model_ckpt_path)

        if load_optimizer:
            optim_ckpt_path = path_handler.get_ckpt_path('training_state', itr)
            assert os.path.exists(optim_ckpt_path)
        else:
            optim_ckpt_path = None
            self.logger.warn('optimizer is not loaded')

        log_dict_items(
            {'model_ckpt_path': model_ckpt_path, 'optim_ckpt_path': optim_ckpt_path},
            level='INFO', indent=False,
        )

        model_ckpt = torch.load(model_ckpt_path, map_location=self.device)
        out = self.comp_model.load_state_dict(model_ckpt['comp_model'], strict=strict)
        self.logger.debug(f'comp_model.load_state_dict: "{out}"')

        if not load_optimizer:
            return

        training_state_ckpt = torch.load(optim_ckpt_path, map_location=self.device)
        self.g_optimizer.load_state_dict(training_state_ckpt['g_optimizer'])
        self.logger.debug(f'load checkpoint: g_optimizer')

        if new_g_lr is not None:
            self.update_learning_rate(self.g_optimizer, new_g_lr)
            self.logger.info(f'g_optimizer: lr is changed to {new_g_lr}')

        if self.g_scheduler and load_scheduler:
            self.g_scheduler.load_state_dict(training_state_ckpt['g_scheduler'])
            self.logger.debug(f'load checkpoint: g_scheduler')
        else:
            self.logger.warn('g_scheduler is not loaded')

        if self.aux_optimizer:
            self.aux_optimizer.load_state_dict(training_state_ckpt['aux_optimizer'])
            self.logger.debug(f'load checkpoint: aux_optimizer')