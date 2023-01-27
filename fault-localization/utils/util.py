
import torch
import torch.nn.functional as F
import itertools
from tqdm import tqdm
from .dl_models import initialize_model


def makeAllSubsetsofSizeN(s, n):
    assert n < len(s)
    l_of_subsets = list(itertools.combinations(s, n))
    l_of_lists = [set(sub) for sub in l_of_subsets]
    return l_of_lists


# def loadModelsFromCheckpoints(checkpoint_path, model_config, total_clients, faulty_clients, faultyion_level):
#     def loadM(p):
#         model = initialize_model(model_config)
#         model.load_state_dict(torch.load(p))
#         model.eval()
#         return model

#     normal_clients_ms = [ckpt for ckpt in [
#         checkpoint_path + f"client_{client}.pt" for client in range(total_clients) if client not in faulty_clients]]
#     faulty_clients_ms = [ckpt for ckpt in [
#         checkpoint_path + f"faulty_client_{faulty_id}_noise_rate_{faultyion_level}_classes.pt" for faulty_id in faulty_clients]]

#     all_ckpts = faulty_clients_ms + normal_clients_ms

#     clients_models = [loadM(ckpt) for ckpt in tqdm(
#         all_ckpts, desc=f"Loading {model_config['model_name']} models")]
#     return clients_models


def getAndprintClientsValAcc(models, valid, use_gpu):

    to_evaluete = 5
    print(f"> Evaluating {len(models)} Client Accuracies but only for {to_evaluete}")
    models = {k:models[k] for k in list(models.keys())[:to_evaluete]}

    testloader = torch.utils.data.DataLoader(
        valid, batch_size=2048, shuffle=False, num_workers=4)
    client2Acc = {}

    device = torch.device("cpu")
    if use_gpu:
        device = torch.device("cuda")
    for cid, client_model in models.items():
        correct = 0
        total = 0
        client_model = client_model.to(device)
        client_model.eval()
        with torch.no_grad():
            count = 0
            for data in testloader:
                 
                images, labels = data[0].to(device), data[1].to(device)

                    

                batch_outputs = client_model(images)

                batch_probs = F.softmax(batch_outputs, dim=1)
                confs_batch, preds_batch = torch.max(batch_probs, dim=1)

                total += labels.size(0)
                correct += (preds_batch == labels).sum().item()
                count += 1
                if count > 3:
                    break


        # print("--------> accuracy: ", correct/total)
        client2Acc[cid] = round((correct/total)*100,2)

        client_model = client_model.to(torch.device("cpu"))

    # print(f"Clients Validation Accuracy {client2Acc}")

    return client2Acc


def evalauteFaultyClientLocalization2(args, noise_rate, client2acc, exp2info, t, valid_set_of_clients,  faultyclientlocalization_clients_combs,  input_gen_time, faultyclientlocalization_time):
    # detection_accuracy = (seq2count.get(
    #     known_val_seq, 0)/args.total_random_inputs) * 100
    
    true_seq = 0

    for comb in faultyclientlocalization_clients_combs:
        # print(comb, valid_set_of_clients)
        if comb == valid_set_of_clients:
            true_seq += 1
    
    detection_accuracy = (true_seq/args.total_random_inputs) * 100
    



    
    
    
    result2info = {}
    result2info["gpu"] = args.gpu
    result2info["Dataset"] = exp2info['data_config']['name']
    result2info['Architecture'] = exp2info['model_config']['model_name']
    result2info['Data Distribution Among Clients'] = exp2info['data_distribution_among_clients']
    result2info['Rounds'] = 1  # exp2info['rounds']
    result2info['Epochs'] = exp2info['epochs']
    result2info['Number of Clients'] = exp2info['num_clients']
    result2info['Number of Faulty Clients'] = exp2info["faulty_clients_ids"]
    result2info['Faulty Clients IDs'] = exp2info["faulty_clients_ids"]
    result2info['Fault Injection'] = noise_rate
    result2info['Number of Random Inputs'] = args.total_random_inputs
    result2info['Neuron Coverage Threshold'] = t
    result2info['Updated Detection Accuracy'] = detection_accuracy
    result2info['FaultyClientLocalization Time'] = faultyclientlocalization_time
    result2info['Random Inputs Generation Time'] = input_gen_time
    result2info['Sequence2Count'] = faultyclientlocalization_clients_combs
    result2info["client2acc"] = client2acc
    result2info["sort_key"] = args.csv_sort_key
    result2info["exp2info"] = exp2info
    result2info["Number of Faulty Clients"] = len(exp2info["faulty_clients_ids"])
    log = f"NA_t {t}, {args.new_key}  Acc: ({true_seq}/{args.total_random_inputs}): {detection_accuracy}, Input Time = {input_gen_time}, Localization Time {faultyclientlocalization_time}"

    print(f"+{log}")
    return result2info
