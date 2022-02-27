import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


from trainer import Trainer, IteratorParams
from data import Loaders
from models import AllCNN


def main():
    DATASET_NAME = 'cifar10'
    params_trainer = {
        'model': AllCNN,
        'loader_train': Loaders(DATASET_NAME).get_blurred_loader,
        'loader_test': Loaders(DATASET_NAME).get_proper_loader,
        'criterion': torch.nn.CrossEntropyLoss,
        'optim': torch.optim.SGD,
        'scheduler': torch.optim.lr_scheduler.ExponentialLR
    }

    trainer = Trainer(**params_trainer)
    model_ls = [{}]
    loaders_ls = [{'batch_size': 128}]
    criterion_ls = [{}]
    optim_ls = [{'lr': 0.05, 'weight_decay': 0.001}]
    scheduler_ls = [{'gamma': 0.97}]
    step_ls = [{'step': step} for step in range(0, 141, 35)]

    iter_params = IteratorParams(model_ls, loaders_ls, criterion_ls, optim_ls, scheduler_ls, step_ls)

    params_runs = {
        'iter_params': iter_params,
        'epochs': 140,
        'exp_name': 'cifar10_noised_deficit',
        'log_dir': 'runs/tensorboard/',
        'val_step': 35,
        'dataset_name': DATASET_NAME,
        'device': device
    }
    trainer.run_trainer(**params_runs)


main()

