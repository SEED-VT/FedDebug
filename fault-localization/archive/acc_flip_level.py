import sys
# sys.path.append("../")
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
from datetime import datetime
from joblib import Parallel, delayed
from typing import Dict, List, Tuple
from tqdm import tqdm

logging.basicConfig(filename='example.log', level=logging.ERROR)
logger = logging.getLogger("pytorch_lightning")
from pytorch_lightning import  seed_everything
seed_everything(786)



def loadM(arg):
    p, model_config = arg
    device = torch.device('cpu')
    temp_m = initialize_model(model_config)
    p_m = ImageClassifer.load_from_checkpoint(
        p, config=model_config, model=temp_m).eval().to(device)
    # p_m.model.eval()
    # print(f"load {p}")
    return p, p_m

def loadModels(ckpt_paths, model_config):
    # print("load .......")
    models = None
    desc = ckpt_paths[0].split("/")[-2]
    args = [(p, model_config) for p in ckpt_paths]
    models_tuples = Parallel(n_jobs=-2)(delayed(loadM)(arg)
                                 for arg in tqdm(args, desc=desc))
    path2model = {}
    for md in models_tuples:
        p , m = md
        path2model[p] = m
    return path2model


def loadModelsFromCheckpoints(checkpoint_path, model_config, total_clients, faulty_clients, faultyion_levels: List[int]) -> Dict[str, torch.nn.Module]:
    normal_clients_paths = [ckpt for ckpt in [
        checkpoint_path + f"client_{client}.ckpt" for client in range(total_clients) if client not in faulty_clients]]
    faulty_clients_paths = [ckpt for ckpt in [
        checkpoint_path + f"faulty_client_{faulty_id}_noise_rate_{noise_rate}_classes.ckpt" for faulty_id in faulty_clients for noise_rate in faultyion_levels]]
    all_paths = faulty_clients_paths + normal_clients_paths
    clients_models = loadModels(all_paths, model_config)
    print("Loading is done")
    return clients_models


# def server_Aggregation(pl_global_model, pl_client_models):
#     global_dict = pl_global_model.model.state_dict() # due to pytorch lighing 
#     for k in global_dict.keys():
#         global_dict[k] = torch.stack([pl_client_models[i].model.state_dict()[k] for i in range(len(pl_client_models))], 0).mean(0)
    
#     pl_global_model.model.load_state_dict(global_dict) # updating the global model with the mean of the client models wieights


def aggToUpdateGlobalModel2(pl_global_model, pl_clients_models):
    global_dict = copy.deepcopy(
        pl_global_model.model.state_dict())  # due to pytorch lighing
    for k in global_dict.keys():
        l = [pl_clients_models[i].model.state_dict()[k]
             for i in range(len(pl_clients_models))]
        s = 0        # print(l[0])
        for t in l:
            s += t
        global_dict[k] = s / len(l)
    # updating the global model with the mean of the client models wieights
    pl_global_model.model.load_state_dict(global_dict)
    pl_global_model.eval()
    pl_global_model.model.eval()

def test(pl_classifier, pl_dm):
    trainer = pl.Trainer(accelerator='gpu', devices=1, enable_model_summary=None, enable_checkpointing = False, logger=False, enable_progress_bar=False)
    d = trainer.test(pl_classifier, pl_dm, verbose=False)[0] # verbose=False to avoid printing
    # test_acc
    test_loss = d["test_loss"]
    test_acc =  d["test_acc"]
    return test_loss, test_acc



def main(args, dname2valid, experiment_key):
    train_cache = Index(args.train_cache_dir)
    print(f"Train Cache dir {args.train_cache_dir} ")
    faultyclientlocalization_cache = Index(args.faultyclientlocalization_cache_dir)
    print(f"FaultyClientLocalization Cache dir {args.faultyclientlocalization_cache_dir}")
    # gpu = True
    # print("using gpu:", gpu)
    # print(f"na_thresholds: {args.na_thresholds}")
    print(f"Faultyion levels: {args.noise_rate}")
    exp2info = train_cache[experiment_key]
    dname = exp2info['data_config']['name']
    ckpt_dir = f"{args.checkpoints_dir}{experiment_key}/"
    num_clients = exp2info["num_clients"]
    faulty_clients_ids = exp2info["faulty_clients_ids"]

    set_faulty_clients = set(faulty_clients_ids)
    set_normal_clients = set([cid for cid in range(
        num_clients) if cid not in faulty_clients_ids])

    print(f"Normal clients set {set_normal_clients}")
    print(f"Faulty clients set {set_faulty_clients}")


    valid = dname2valid[dname]

    dm = FedDataModule(valid, valid, 1024, 4)

    # all_combinations = makeAllSubsetsofSizeN(
    #     set(list(range(num_clients))), num_clients-len(faulty_clients_ids))

    clients_models_all_faulty_levels = loadModelsFromCheckpoints(
        checkpoint_path=ckpt_dir, model_config=exp2info["model_config"], total_clients=num_clients, faulty_clients=faulty_clients_ids, faultyion_levels=args.noise_rate)

    normal_client2model = {int(ckpt.split("client_")[-1].split(".")[0]): client_model for ckpt, client_model in clients_models_all_faulty_levels.items(
    ) if f"faulty_client_" not in ckpt}

    assert set(normal_client2model.keys()) == set_normal_clients

    
    for noise_rate in args.noise_rate:
        courrpt_client2model = {int(ckpt.split("client_")[-1].split("_")[0]): client_model for ckpt, client_model in clients_models_all_faulty_levels.items(
        ) if f"noise_rate_{noise_rate}" in ckpt}

        assert set(courrpt_client2model.keys()) == set_faulty_clients

        client2models = normal_client2model | courrpt_client2model
        temp_m = initialize_model(exp2info["model_config"])
        # print(temp_m)
        global_model = ImageClassifer(config=exp2info["model_config"], model=temp_m)
        aggToUpdateGlobalModel2(global_model, list(normal_client2model.values()))
        

        new_key = f"Acc_noise_rate_{noise_rate}_{experiment_key}"  

        # if new_key in faultyclientlocalization_cache.keys():
        #     print(f"Cache Hit Acc: {new_key}")

        # print("\n \n here")
        
        test_loss_normal, test_acc_nromal =  test(global_model, dm)
        
        aggToUpdateGlobalModel2(global_model, list(client2models.values()))

        test_loss_faulty, test_acc_faulty =  test(global_model, dm)

        # print(f"Normal (loss, acc) {(test_loss_normal, test_acc_nromal)}, Faulty (loss, acc) {(test_loss_faulty, test_acc_faulty)} ")
        d =  {"Noise_rate": noise_rate, "Normal Test Loss": test_loss_normal, "Normal Test Accuracy": test_acc_nromal, "Faulty Loss": test_loss_faulty, "Faulty Accuracy":test_acc_faulty}

        d["Dataset"] = exp2info['data_config']['name']
        d['Architecture'] = exp2info['model_config']['model_name']
        d["Number of Clients"] = exp2info['num_clients']
        d["Number of Faulty Clients"] = exp2info["faulty_clients_ids"]
        d['Epochs'] = exp2info['epochs']
        
        print(d)
        faultyclientlocalization_cache[new_key] = d
        print("\n           -------------------------------------- \n")





if __name__ == "__main__":
    print("Determing the accuracy of clients")

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_cache_dir", type=str, default="../storage/cache/fl/",
                        help="cache directory that contains training models and info", required=True)
    
    parser.add_argument("--checkpoints_dir", type=str,
                        default="../storage/checkpoints/", help="checkpoint directory", required=True)
    
    parser.add_argument("--keys_file", type=str,
                        default="keys.txt", help="text files which contains config keys", required=True)
    
    parser.add_argument("--dataset_dir", type=str,
                        default="../storage/datasets/", help="dataset dir", required=True)
    
    parser.add_argument("--faultyclientlocalization_cache_dir", type=str,
                        default="../storage/cache/random_results_initial_last_layers/", help="cache_contains_faultyclientlocalization_results", required=True)

    parser.add_argument("--noise_rate", type=str, default="5,10",
                        help="Number of classes noise_rate in the attack", required=True)
    

    args = parser.parse_args()


    assert os.path.exists(
        args.checkpoints_dir), "checkpoint_dir does not exist"
    assert os.path.exists(
        args.train_cache_dir), "train_cache_dir does not exist"
    assert os.path.exists(
        args.dataset_dir), "dataset_dir does not exist"

    assert args.train_cache_dir.endswith("/"), "cache_dir should end with '/'"
    assert args.checkpoints_dir.endswith(
        "/"), "checkpoint_dir should end with '/'"
    
    args.noise_rate = [int(t) for t in args.noise_rate.split(",")]

    dname2valid = {name: initializeTrainAndValidationDataset(
        name, data_dir=args.dataset_dir)[1] for name in ["cifar10", "femnist"]}

    

    print(f"Timestampe {datetime.now()}")

    with open(args.keys_file, "r") as f:
        config_keys = [k.strip() for k in f.readlines()]


    for experiment_key in config_keys:
        print(f"\n\n\n ** Experiment_key: {experiment_key} **")
        main(args, dname2valid, experiment_key)
