from typing import Dict, Tuple, Optional, Union, List

from copy import deepcopy

import torch
import torch.nn as nn
from torch import Tensor

from src.losses import build_loss
from src.models.discriminator import build_discriminator
from src.utils.logger import IndentedLog, log_dict_items, bolded_log
from src.utils.registry import TRAINER_REGISTRY
from src.utils.path import PathHandler
from .optimizer import build_optimizer, build_scheduler
from .rate_distortion_trainer import RateDistortionTrainer

@TRAINER_REGISTRY.register()
class GANRateDistortionTrainer(RateDistortionTrainer):
    def _print_models(self) -> None:
        super()._print_models()
        bolded_log('discriminator', level="DEBUG", new_line=True, prefix='', suffix='')
        self.logger.debug(str(self.discriminator))

    def _set_models(self) -> None:
        super()._set_models()
        self.discriminator = build_discriminator(self.opt.discriminator).to(self.device)
        self.discriminator.train()

    def _set_losses(self) -> None:
        super()._set_losses() # set rate-distortion loss funcs
        loss_opt = deepcopy(self.opt.loss)
        self.gan_loss = build_loss(loss_opt.gan_loss, loss_name='gan_loss')

    def _set_optimizer_scheduler(self) -> None:
        super()._set_optimizer_scheduler()
        optim_opt = deepcopy(self.opt.optim)
        # set d_optimizer
        with IndentedLog(level="INFO", msg='building d_optimizer'):
            self.d_optimizer = build_optimizer({k: v for k, v in self.discriminator.named_parameters()}, optim_opt.d_optimizer)
            if optim_opt.get('d_scheduler'):
                self.d_scheduler = build_scheduler(self.d_optimizer, optim_opt.d_scheduler)
            else:
                self.logger.warn('d_cheduler is NOT build')
                self.d_scheduler = None

    def optimize_parameters(self, current_iter: int, data_dict: Dict):
        log_dict = {}

        ###################################################################
        #                             Train G                             
        ###################################################################
        self.discriminator.requires_grad_(False)
        self.g_optimizer.zero_grad()
        if self.aux_optimizer:
            self.aux_optimizer.zero_grad()

        # run model
        real_images, fake_images, bpp, other_outputs = self.run_comp_model(data_dict)
        g_fake = self.discriminator(fake_images, **other_outputs)
        log_dict['qbpp'] = other_outputs.get('qbpp', -1)
        
        # calculate losses
        g_loss_dict: Dict[str, Tensor] = {}
        g_loss_dict['distortion'] = self.distortion_loss(real_images, fake_images, **other_outputs)
        g_loss_dict['adv'] = self.gan_loss(g_fake, is_real=True, is_disc=False)

        if self.rate_loss:
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
        if self.opt.optim.get('clip_max_norm'):
            nn.utils.clip_grad_norm_(self.comp_model.parameters(), self.opt.optim.clip_max_norm)
        self.g_optimizer.step()

        log_dict.update(g_loss_dict)

        if self.g_scheduler:
            self.g_scheduler.step()

        if self.aux_optimizer:
            log_dict['aux'] = self.optimize_aux_parameters()

        ###################################################################
        #                             Train D                             
        ###################################################################
        self.discriminator.requires_grad_(True)
        self.d_optimizer.zero_grad()
        d_real = self.discriminator(real_images, **other_outputs)
        d_fake = self.discriminator(fake_images.detach(), **other_outputs)

        l_d_real = self.gan_loss(d_real, is_real=True, is_disc=True) * 0.5
        l_d_fake = self.gan_loss(d_fake, is_real=False, is_disc=True) * 0.5
        d_loss = l_d_real + l_d_fake
        d_loss.backward()

        out_d_real, out_d_fake = self.calc_avg_d_score_for_log(d_real, d_fake)
        log_dict.update({
            'd_real': l_d_real,
            'd_fake': l_d_fake,
            'd_total': d_loss,
            'out_d_real': out_d_real,
            'out_d_fake': out_d_fake,
        })
        
        self.d_optimizer.step()
        if self.d_scheduler:
            self.d_scheduler.step()

        return log_dict
    
    @staticmethod
    def calc_avg_d_score_for_log(d_real: Union[Tensor, List], d_fake: Union[Tensor, List]) -> Tuple[Tensor, Tensor]:
        if isinstance(d_real, list):
            out_d_real = torch.mean(torch.Tensor([torch.mean(rp.detach()) for rp in d_real]))
            out_d_fake = torch.mean(torch.Tensor([torch.mean(fp.detach()) for fp in d_fake]))
        elif isinstance(d_real, torch.Tensor):
            out_d_real = torch.mean(d_real.detach())
            out_d_fake = torch.mean(d_fake.detach())
        else:
            raise TypeError(f'd_real should be list or torch.Tensor, but {type(d_real)}')
        return out_d_real, out_d_fake

    def save(self, current_iter: int) -> None:
        # save model
        self.model_saver.save({'comp_model': self.comp_model}, 'comp_model', current_iter, keep=True)
        self.model_saver.save({'discriminator': self.discriminator}, 'discriminator', current_iter, keep=self.opt.get('keep_discriminator', False))
        
        # save training_state
        optimizer_scheduler_dict = {
                'g_optimizer': self.g_optimizer,
                'd_optimizer': self.d_optimizer,
            }
        if self.aux_optimizer:
            optimizer_scheduler_dict['aux_optimizer'] = self.aux_optimizer
        if self.g_scheduler:
            optimizer_scheduler_dict['g_scheduler'] = self.g_scheduler
        if self.d_scheduler:
            optimizer_scheduler_dict['d_scheduler'] = self.d_scheduler
        self.model_saver.save(optimizer_scheduler_dict, 'training_state', current_iter, keep=self.opt.get('keep_training_state', False))

    def _load_checkpoint(self,
                         exp: str,
                         itr: int,
                         load_optimizer: bool=True,
                         load_discriminator: bool=True,
                         load_scheduler: bool=True,
                         new_g_lr: Optional[float]=None,
                         new_d_lr: Optional[float]=None,
                         strict: bool=True,
                         **kwargs) -> None:
        ## get checkpoint path
        path_handler = PathHandler(self.opt.path.ckpt_root, exp)
        model_ckpt_path = path_handler.get_ckpt_path('comp_model', itr)
        optim_ckpt_path = path_handler.get_ckpt_path('training_state', itr)
        discriminator_ckpt_path = path_handler.get_ckpt_path('discriminator', itr)
        log_dict_items({
            'model_ckpt_path': model_ckpt_path,
            'optim_ckpt_path': optim_ckpt_path,
            'discriminator_ckpt_path': discriminator_ckpt_path,
            }, level='INFO', indent=False)

        # Load G & g_optimizer #######################################################################
        comp_model_ckpt = torch.load(model_ckpt_path, map_location=self.device)
        out = self.comp_model.load_state_dict(comp_model_ckpt['comp_model'], strict=strict)
        self.logger.debug(f'comp_model.load_state_dict: "{out}"')

        if load_optimizer:
            optim_ckpt = torch.load(optim_ckpt_path, map_location=self.device)

            self.g_optimizer.load_state_dict(optim_ckpt['g_optimizer'])
            self.logger.debug(f'g_optimizer is loaded')

            if new_g_lr is not None: # update only lr
                self.update_learning_rate(self.g_optimizer, new_g_lr)
                self.logger.info(f'g_optimizer lr is changed to {new_g_lr}')

            if self.g_scheduler and load_scheduler:
                self.g_scheduler.load_state_dict(optim_ckpt['g_scheduler'])
                self.logger.debug(f'g_scheduler is loaded')
            else:
                self.logger.warn('g_scheduler is NOT loaded')

            if self.aux_optimizer:
                self.aux_optimizer.load_state_dict(optim_ckpt['aux_optimizer'])
                self.logger.debug(f'aux_optimizer is loaded')
        else:
            self.logger.warn('g_optimizer, g_scheduler, and aux_optimizer are NOT loaded')

        # Load D & d_optimizer #######################################################################
        if not load_discriminator:
            self.logger.warn('discriminator, d_optimizer, d_scheduler are NOT loaded')
            return
        
        discriminator_ckpt = torch.load(discriminator_ckpt_path, map_location=self.device)
        out = self.discriminator.load_state_dict(discriminator_ckpt['discriminator'], strict=strict)
        self.logger.debug(f'discriminator.load_state_dict: "{out}"')

        if load_optimizer:
            self.d_optimizer.load_state_dict(optim_ckpt['d_optimizer'])
            self.logger.debug(f'd_optimizer is loaded')

            if new_d_lr is not None: #update lr
                self.update_learning_rate(self.d_optimizer, new_d_lr)
                self.logger.info(f'd_optimizer lr is changed to {new_d_lr}')

            if self.d_scheduler and load_scheduler:
                self.d_scheduler.load_state_dict(optim_ckpt['d_scheduler'])
                self.logger.debug(f'd_scheduler is loaded')
            else:
                self.logger.warn('d_scheduler is NOT loaded')
        else:
            self.logger.warn('d_optimizer and d_scheduler are NOT loaded')

        self.logger.debug('g_optimizer:')
        self.logger.debug(self.g_optimizer.__str__())
        self.logger.debug('d_optimizer:')
        self.logger.debug(self.d_optimizer.__str__())