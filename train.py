import argparse
import collections
import itertools
import torch
import numpy as np
import dataloader as module_data
import model as module_model
from trainer import Trainer
from utils import prepare_device
from utils.parse_config import ConfigParser


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    dataset = config.init_obj('dataset', module_data)
    data_loader = config.init_obj('dataloader', module_data, dataset=dataset)
    valid_data_loader = data_loader.split_validation()

    # create models
    generator = config.init_obj('generator', module_model)
    mp_discriminator = config.init_obj('mp_discriminator', module_model)
    ms_discriminator = config.init_obj('ms_discriminator', module_model)
    mel_spec = config.init_obj('mel_spec', module_model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    generator = generator.to(device)
    mp_discriminator = mp_discriminator.to(device)
    ms_discriminator = ms_discriminator.to(device)
    mel_spec = mel_spec.to(device)
    if len(device_ids) > 1:
        generator = torch.nn.DataParallel(generator, device_ids=device_ids)
        mp_discriminator = torch.nn.DataParallel(mp_discriminator, device_ids=device_ids)
        ms_discriminator = torch.nn.DataParallel(ms_discriminator, device_ids=device_ids)
        mel_spec = torch.nn.DataParallel(mel_spec, device_ids=device_ids)


    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    generator_optimizer = config.init_obj('generator_optimizer', torch.optim, generator.parameters())
    generator_lr_scheduler = config.init_obj('generator_lr_scheduler', torch.optim.lr_scheduler, generator_optimizer) if config["generator_lr_scheduler"] else None
    discriminator_optimizer = config.init_obj('discriminator_optimizer', torch.optim, itertools.chain(mp_discriminator.parameters(), ms_discriminator.parameters()))
    discriminator_lr_scheduler = config.init_obj('discriminator_lr_scheduler', torch.optim.lr_scheduler, discriminator_optimizer) if config["discriminator_lr_scheduler"] else None

    logger.info("Start training")

    trainer = Trainer(generator, mp_discriminator, ms_discriminator, generator_optimizer, generator_lr_scheduler, discriminator_optimizer, discriminator_lr_scheduler,
                    config=config,
                    device=device,
                    data_loader=data_loader,
                    valid_data_loader=valid_data_loader,
                    mel_spec=mel_spec
                    )

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
