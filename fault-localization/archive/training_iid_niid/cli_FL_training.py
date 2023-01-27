import sys
sys.path.append("../")
import os
import gc
import copy
import argparse
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from diskcache import Index
from utils.dl_models import ImageClassifer, initialize_model
from utils.fl_datasets import *
from pathlib import Path
import logging

logging.basicConfig(filename='example.log', level=logging.ERROR)
logger = logging.getLogger("pytorch_lightning")

from pytorch_lightning import  seed_everything
seed_everything(786)



def _trainModel(pl_model, train_dataset, val_dataset, data_config, epochs, checkpoint_path):
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=epochs, precision=16,
                         check_val_every_n_epoch=3, limit_val_batches=0.25, enable_model_summary=None, enable_checkpointing=False, logger=False)
    dm = FedDataModule(train_dataset, val_dataset,
                       data_config["batch_size"])

    trainer.fit(pl_model, dm)
    pl_model.cpu()

    # this will create the directory if doesnot exist
    trainer.save_checkpoint(checkpoint_path)
    del pl_model
    del trainer
    del dm
    del train_dataset

    pl_model = None
    dm = None
    trainer = None
    train_dataset = None
    torch.cuda.empty_cache()
    gc.collect()


def simulateFL(model_config, data_config, clients2traindatasets, val_dataset, epochs, base_model_ws):
    for p_key, train_data in clients2traindatasets.items():
        print(f"Training : {p_key}")
        pl_m = ImageClassifer(model_config)
        pl_m.model = initialize_model(model_config)
        pl_m.model.load_state_dict(copy.deepcopy(base_model_ws))

        _trainModel(pl_m, train_data, val_dataset, data_config=data_config,
                    epochs=epochs, checkpoint_path=p_key)


def prepareIIDDataset(dname, dataset_dir, num_clients):
    train, valid, num_classes = initializeTrainAndValidationDataset(
        dname, data_dir=dataset_dir)
    clients_datasets = splitDataSetIntoNClientsIID(train, clients=num_clients)
    return clients_datasets, valid, num_classes


def prepareNIIDDataset(dname, dataset_dir, num_clients):
    train, valid, num_classes = initializeTrainAndValidationDataset(
        dname, data_dir=dataset_dir)
    clients_datasets = splitDataSetIntoNClientsNonIID(
        train, clients=num_clients)
    return clients_datasets, valid, num_classes


def main(args):
    def getFLClientsDatasets():
        if args.sampling == "iid":
            return prepareIIDDataset(args.dataset, dataset_dir, args.clients)
        elif args.sampling == "niid":
            return prepareNIIDDataset(args.dataset, dataset_dir, args.clients)
        else:
            raise NotImplementedError(
                f"Sampling {args.sampling} is not implemented")

    checkpoint_dir = args.storage + args.checkpoints_dir_name
    cache_dir = args.storage + args.cache_name

    dataset_dir = args.storage + "datasets/"

    gray_datasets = ["mnist", "fashionmnist", "femnist"]

    channels = 3

    if args.dataset in gray_datasets:
        channels = 1
    else:
        channels = 3

    model_config = {"model_name": args.model,
                    "use_pretrained": args.pretrained, "lr": args.lr, "weight_decay": args.weight_decay, "channels": channels}

    data_config = {'name': args.dataset,
                   "batch_size": args.batch_size}

    cache = Index(cache_dir)

    faulty_clients_ids = [int(x) for x in args.faulty_clients_ids.split(",")]

    # key = f"{args.sampling}_{model_config['model_name']}_{args.dataset}_clients_{args.clients}_faulty_{faulty_clients_ids}_bsize_{data_config['batch_size']}_epochs_{args.epochs}_lr_{args.lr}_pre_trained_{args.pretrained}"
    key2 = f"{args.sampling}_{model_config['model_name']}_{args.dataset}_clients_{args.clients}_faulty_{faulty_clients_ids}_bsize_{data_config['batch_size']}_epochs_{args.epochs}_lr_{args.lr}"
    key = key2
    if key in cache.keys():  # or (key2 in cache.keys() and args.pretrained):
        print(f"Cache Hit {key} (Training is already done)")
        return

    # print(f"Exisitng keys {[k for k in list(cache.keys()) if 'vgg' in k and 'clients_50' in k]}")
    # print(key)
    # exit()

    print(f"\n\n  ***Simulating FL setup {key} ***")
    model_config["checkpoint_path"] = checkpoint_dir + f"{key}/"
    clientsdatasets, valid, num_classes = getFLClientsDatasets()

    faultyclients2datasets = {}
    for faulty_id in faulty_clients_ids:
        for f in args.noise_rate:
            k = checkpoint_dir + \
                f"{key}/faulty_client_{faulty_id}_noise_rate_{f}_classes.ckpt"
            faultyclients2datasets[k] = NoisyDataset(copy.deepcopy(
                clientsdatasets[faulty_id]), num_classes=num_classes, class_ids=list(range(f)))

    normalclients2datasets = {checkpoint_dir + f"{key}/client_{normal_id}.ckpt": clientsdatasets[normal_id] for normal_id in range(args.clients)}

    data_config["single_input_shape"] = valid[0][0].unsqueeze(0).shape

    model_config["classes"] = num_classes

    base_model = initialize_model(model_config)

    simulateFL(model_config, data_config,
               faultyclients2datasets, valid, epochs=args.epochs, base_model_ws=copy.deepcopy(base_model.state_dict()))

    simulateFL(model_config, data_config,
               normalclients2datasets, valid, epochs=args.epochs, base_model_ws=copy.deepcopy(base_model.state_dict()))

    store = {"all_clients_datasets": clientsdatasets, "num_clients": args.clients,
             "faulty_clients_ids": faulty_clients_ids, "epochs": args.epochs, 'checkpoint_path': model_config['checkpoint_path'], "model_config": model_config,
             "data_config": data_config, 'data_distribution_among_clients': args.sampling, "args": args, "base_model_ws": copy.deepcopy(base_model.state_dict())}

    cache[key] = store
    print(f"++Training is done: {key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling", type=str, default="iid", required=True)
    parser.add_argument("--cache_name", type=str, default="fl/", required=True)
    parser.add_argument("--checkpoints_dir_name", type=str,
                        default="checkpoints/", required=True)

    parser.add_argument("--model", type=str, default="resnet18", required=True)
    parser.add_argument("--pretrained", type=int, default=1,
                        help="use pretrained model 0 is False and 1 is True")
    # parser.add_argument("--feature_extract", type=int,
    #                     default=0, help="to freeze layers")
    parser.add_argument("--epochs", type=int, default=1, required=True)
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate", required=True)
    parser.add_argument("--weight_decay", type=float,
                        default=3e-4, help="weight decay")

    parser.add_argument("--dataset", type=str,
                        default="cifar10", required=True)
    parser.add_argument("--batch_size", type=int,
                        default=512, help="batch size")
    
    parser.add_argument("--clients", type=int, default=5,
                        help="number of clients participating in FL", required=True)
    parser.add_argument("--faulty_clients_ids", type=str, default="0",
                        help="comma separated list of clients to faulty", required=True)

    parser.add_argument("--storage", type=str, default="../storage/",
                        help="to store models,caches, datasets", required=True)

    parser.add_argument("--noise_rate", type=str, default="",
                        help="comma seperated noise_rateing levels", required=True)

    args = parser.parse_args()

    print(
        f"Cache name {args.cache_name}, Checkpoints dir {args.checkpoints_dir_name} ")

    assert args.cache_name.endswith("/")
    assert args.checkpoints_dir_name.endswith("/")

    args.noise_rate = [int(l) for l in args.noise_rate.split(",")]
    print(f"Noise_rateing Levels {args.noise_rate}")

    if args.pretrained == 1:
        args.pretrained = True
    else:
        args.pretrained = False

    # print(f"Arguments: {args}")

    main(args)
