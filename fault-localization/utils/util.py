
import torch
import torch.nn.functional as F
import itertools
import copy

def makeAllSubsetsofSizeN(s, n):
    assert n < len(s)
    l_of_subsets = list(itertools.combinations(s, n))
    l_of_lists = [set(sub) for sub in l_of_subsets]
    return l_of_lists

def testAccModel(model, valid, use_gpu=True):
    testloader = torch.utils.data.DataLoader(
        valid, batch_size=2048, shuffle=False, num_workers=4)
    
    device = torch.device("cpu")
    if use_gpu:
        device = torch.device("cuda")
    
    model = model.to(device)
    model.eval()
    total = 0
    correct = 0 
    with torch.no_grad():
        for data in testloader:
                
            images, labels = data[0].to(device), data[1].to(device)
            batch_outputs = model(images)

            batch_probs = F.softmax(batch_outputs, dim=1)
            confs_batch, preds_batch = torch.max(batch_probs, dim=1)

            total += labels.size(0)
            correct += (preds_batch == labels).sum().item()
    
    prediction_acc = round((correct/total)*100,2)

    return prediction_acc        

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




def aggToUpdateGlobalModel(clients_models):
    temp_key = list(clients_models.keys())[0]
    global_dict = copy.deepcopy(
        clients_models[temp_key].state_dict())  # due to pytorch lighing
    for k in global_dict.keys():
        l = [clients_models[i].state_dict()[k]for i in range(len(clients_models))]
        s = 0        # print(l[0])
        for t in l:
            s += t
        global_dict[k] = s / len(l)
    # updating the global model with the mean of the client models wieights
    
    global_model = copy.deepcopy(clients_models[0])
    
    global_model.load_state_dict(global_dict)
    global_model.eval()
    # global_model.model.eval()
    return global_model

