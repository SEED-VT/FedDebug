import copy
import gc
import logging

import pytorch_lightning as pl
import torch
from diskcache import Index
from pytorch_lightning import seed_everything

from .dl_models import ImageClassifer, initialize_model
from .fl_datasets import *

logging.basicConfig(filename='example.log', level=logging.ERROR)
logger = logging.getLogger("pytorch_lightning")
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
    del trainer
    del dm
    del train_dataset

    dm = None
    trainer = None
    train_dataset = None
    torch.cuda.empty_cache()
    gc.collect()
    return pl_model


def simulateFL(model_config, data_config, clients2traindatasets, val_dataset, epochs, base_model_ws):
    pl_trained_models = {}
    for p_key, train_data in clients2traindatasets.items():
        print(f"Training : {p_key}")
        pl_m = ImageClassifer(model_config)
        pl_m.model = initialize_model(model_config)
        pl_m.model.load_state_dict(copy.deepcopy(base_model_ws))

        pl_m = _trainModel(pl_m, train_data, val_dataset, data_config=data_config,
                           epochs=epochs, checkpoint_path=p_key)
        pl_trained_models[p_key] = pl_m
    return pl_trained_models


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


def trainFLMain(args):
    def getFLClientsDatasets():
        if args.sampling == "iid":
            return prepareIIDDataset(args.dataset, dataset_dir, args.clients)
        elif args.sampling == "niid":
            return prepareNIIDDataset(args.dataset, dataset_dir, args.clients)
        else:
            raise NotImplementedError(
                f"Sampling {args.sampling} is not implemented")

    args.pretrained = 1
    args.storage = ".storage/"
    args.cache_name = "cache/fl/"
    args.checkpoints_dir_name = "checkpoints/"
    
    checkpoint_dir = args.storage + args.checkpoints_dir_name
    cache_dir = args.storage + args.cache_name

    dataset_dir = args.storage + "datasets/"

    gray_datasets = ["mnist", "fashionmnist", "femnist"]
    channels = 3
    if args.dataset in gray_datasets:
        channels = 1

    model_config = {"model_name": args.model,
                    "use_pretrained": args.pretrained, "lr": args.lr, "weight_decay": args.weight_decay, "channels": channels}

    data_config = {'name': args.dataset,
                   "batch_size": args.batch_size}

    cache = Index(cache_dir)

    faulty_clients_ids = [int(x) for x in args.faulty_clients_ids.split(",")]

    key2 = f"{args.sampling}_{model_config['model_name']}_{args.dataset}_clients_{args.clients}_faulty_{faulty_clients_ids}_bsize_{data_config['batch_size']}_epochs_{args.epochs}_lr_{args.lr}"
    key = key2

    print(f"\n\n  ***Simulating FL setup {key} ***")
    model_config["checkpoint_path"] = checkpoint_dir + f"{key}/"
    clientsdatasets, valid, num_classes = getFLClientsDatasets()

    faultyclients2datasets = {}
    stringID2intID = {}
    for faulty_id in faulty_clients_ids:
        k = checkpoint_dir + \
            f"{key}/faulty_client_{faulty_id}_noise_rate_{args.noise_rate}_classes.ckpt"
        faultyclients2datasets[k] = NoisyDataset(copy.deepcopy(
            clientsdatasets[faulty_id]), num_classes=num_classes, noise_rate=args.noise_rate)
        stringID2intID[k] = faulty_id

    normalclients2datasets = {}
    for normal_id in range(args.clients):
        if normal_id not in faulty_clients_ids:
            k = checkpoint_dir + f"{key}/client_{normal_id}.ckpt"
            normalclients2datasets[k] = clientsdatasets[normal_id]
            stringID2intID[k] = normal_id

    data_config["single_input_shape"] = valid[0][0].unsqueeze(0).shape
    print(f'input shape, {data_config["single_input_shape"]}')
    # return

    model_config["classes"] = num_classes

    base_model = initialize_model(model_config)

    temp_d1 = simulateFL(model_config, data_config,
                         faultyclients2datasets, valid, epochs=args.epochs, base_model_ws=copy.deepcopy(base_model.state_dict()))

    temp_d2 = simulateFL(model_config, data_config,
                         normalclients2datasets, valid, epochs=args.epochs, base_model_ws=copy.deepcopy(base_model.state_dict()))

    client2models = {**temp_d1,  **temp_d2}

    print(f"Total clients: {len(client2models)}")

    store = {"all_clients_datasets": clientsdatasets, "num_clients": args.clients,
             "faulty_clients_ids": faulty_clients_ids, "epochs": args.epochs, 'checkpoint_path': model_config['checkpoint_path'], "model_config": model_config,
             "data_config": data_config, 'data_distribution_among_clients': args.sampling, "args": args, "base_model_ws": copy.deepcopy(base_model.state_dict())}

    cache[key] = store

    # changing keys to int
    client2models = {stringID2intID[k]: v for k, v in client2models.items()}
    print(f"++Training is done: {key}")
    return client2models, store
