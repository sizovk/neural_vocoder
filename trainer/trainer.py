import librosa
import numpy as np
import torch
import torch.nn.functional as F
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from model import feature_loss, discriminator_loss, generator_loss


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, generator, mp_discriminator, ms_discriminator, generator_optimizer, generator_lr_scheduler, discriminator_optimizer, discriminator_lr_scheduler,
                 config, device, data_loader, valid_data_loader, mel_spec):
        super().__init__(generator, mp_discriminator, ms_discriminator, generator_optimizer, discriminator_optimizer, config)

        self.config = config
        self.device = device
        self.data_loader = data_loader
        if self.len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.generator_lr_scheduler = generator_lr_scheduler
        self.discriminator_lr_scheduler = discriminator_lr_scheduler
        self.mel_spec = mel_spec

        self.train_metrics = MetricTracker('discriminator_loss', 'generator_loss', 'discriminator_grad_norm', 'generator_grad_norm')
        self.valid_metrics = MetricTracker('discriminator_loss', 'generator_loss')

    def _loss_discriminator(self, true_sound, gen_sound):
        true_output, gen_output, _, _ = self.mp_discriminator(true_sound.unsqueeze(1), gen_sound.detach())
        mp_loss = discriminator_loss(true_output, gen_output)

        true_output, gen_output, _, _ = self.ms_discriminator(true_sound.unsqueeze(1), gen_sound.detach())
        ms_loss = discriminator_loss(true_output, gen_output)

        return mp_loss + ms_loss

    def _loss_generator(self, true_sound, gen_sound, true_melspec):
        gen_melspec = self.mel_spec(gen_sound.squeeze())

        mel_loss = F.l1_loss(true_melspec, gen_melspec) * 45

        true_output, gen_output, true_feature, gen_feature = self.mp_discriminator(true_sound.unsqueeze(1), gen_sound.detach())
        mp_loss = discriminator_loss(true_output, gen_output)
        mp_feature_loss = feature_loss(true_feature, gen_feature)

        true_output, gen_output, true_feature, gen_feature = self.ms_discriminator(true_sound.unsqueeze(1), gen_sound.detach())
        ms_loss = discriminator_loss(true_output, gen_output)
        ms_feature_loss = feature_loss(true_feature, gen_feature)
        
        return mel_loss + mp_loss + mp_feature_loss + ms_loss + ms_feature_loss

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.generator.train()
        self.mp_discriminator.train()
        self.ms_discriminator.train()
        self.train_metrics.reset()
        for batch_idx, true_sound in enumerate(self.data_loader):
            true_sound = true_sound.to(self.device)
            true_melspec = self.mel_spec(true_sound)
            gen_sound = self.generator(true_melspec)
            if true_sound.shape[-1] < gen_sound.shape[-1]:
                diff = gen_sound.shape[-1] - true_sound.shape[-1]
                true_sound = F.pad(true_sound, (0, diff))
                true_melspec = self.mel_spec(true_sound)
        
            self.set_requires_grad(self.mp_discriminator, True)
            self.set_requires_grad(self.ms_discriminator, True)
            self.discriminator_optimizer.zero_grad()
            discriminator_loss = self._loss_discriminator(true_sound, gen_sound)
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            self.set_requires_grad(self.mp_discriminator, False)
            self.set_requires_grad(self.ms_discriminator, False)
            self.generator_optimizer.zero_grad()
            generator_loss = self._loss_generator(true_sound, gen_sound, true_melspec)
            generator_loss.backward()
            self.generator_optimizer.step()

            self.train_metrics.update('discriminator_loss', discriminator_loss.item())
            self.train_metrics.update('generator_loss', generator_loss.item())
            self.train_metrics.update("discriminator_grad_norm", self.get_grad_norm(self.discriminator))
            self.train_metrics.update("generator_grad_norm", self.get_grad_norm(self.generator))

            if batch_idx % self.log_step == 0:
                if self.writer is not None:
                    self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, mode="train")
                    if self.generator_lr_scheduler is not None:
                        self.writer.add_scalar(
                            "generator_learning_rate", self.generator_lr_scheduler.get_last_lr()[0]
                        )
                    if self.discriminator_lr_scheduler is not None:
                        self.writer.add_scalar(
                            "discriminator_learning_rate", self.discriminator_lr_scheduler.get_last_lr()[0]
                        )
                    for metric_name in self.train_metrics.keys():
                        self.writer.add_scalar(f"{metric_name}", self.train_metrics.avg(metric_name))
                logger_message = "Train Epoch: {} {}".format(epoch, self._progress(batch_idx))
                for metric_name in self.train_metrics.keys():
                    metric_res = self.train_metrics.avg(metric_name)
                    if self.writer is not None:
                        self.writer.add_scalar(f"{metric_name}", metric_res)
                    logger_message += f" {metric_name}: {metric_res:.2f}"
                self.logger.debug(logger_message)

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.discriminator_lr_scheduler is not None:
            self.discriminator_lr_scheduler.step()
        if self.generator_lr_scheduler is not None:
            self.generator_lr_scheduler.step()
        return log

    @torch.no_grad()
    def _log_audio_examples(self):
        for i in [1, 2, 3]:
            wav, sr = librosa.load(f"val_samples/audio_{i}.wav")
            wav = torch.from_numpy(wav).to(self.device)
            gen_audio = self.generator(self.mel_spec(wav)).squeeze()
            self.writer.add_audio(f"audio_{i}", gen_audio, sr)

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.generator.eval()
        self.discriminator.eval()
        self.valid_metrics.reset()
        self._log_audio_examples()
        with torch.no_grad():
            for true_sound in self.valid_data_loader:
                true_sound = true_sound.to(self.device)
                true_melspec = self.mel_spec(true_sound)
                gen_sound = self.generator(true_melspec)
                if true_sound.shape[-1] < gen_sound.shape[-1]:
                    diff = gen_sound.shape[-1] - true_sound.shape[-1]
                    true_sound = F.pad(true_sound, (0, diff))
                    true_melspec = self.mel_spec(true_sound)
                
                discriminator_loss = self._loss_discriminator(true_sound, gen_sound)
                generator_loss = self._loss_generator(true_sound, gen_sound, true_melspec)

                self.valid_metrics.update('discriminator_loss', discriminator_loss.item())
                self.valid_metrics.update('generator_loss', generator_loss.item())
    
        if self.writer is not None:
            self.writer.set_step(epoch * self.len_epoch, mode="val")
            for metric_name in self.valid_metrics.keys():
                self.writer.add_scalar(f"{metric_name}", self.valid_metrics.avg(metric_name))

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} steps ({:.0f}%)]'
        current = batch_idx
        total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
    
    @torch.no_grad()
    def get_grad_norm(self, model, norm_type=2):
        parameters = model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()
