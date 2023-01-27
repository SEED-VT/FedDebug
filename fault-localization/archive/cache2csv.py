from ast import arg
from unittest import result
import pandas as pd
import argparse
from diskcache import Index
import pandas as pd


def getAccuracy(all_clients_ids , actual_faulty_clients, faultyclientlocalization_pred_val_clients):
    actual_faulty_clients = set([c for c in actual_faulty_clients])
    detection_acc = 0
    for pred_val_clients in faultyclientlocalization_pred_val_clients:
        prd_crpt_set = all_clients_ids - set(pred_val_clients)
        print(f"+++ {prd_crpt_set} ")
        found_bugs = len(actual_faulty_clients.intersection(prd_crpt_set)) 
        acc =  (found_bugs/len(actual_faulty_clients))*100
        detection_acc += acc
    return detection_acc/len(faultyclientlocalization_pred_val_clients)


def mainCache2CsvFaultyClientLocalization(args):
    print(">> Loading cache")
    cache = Index(args.cache_dir)

    cols = ['Number of Clients', 'Updated Detection Accuracy',  'Fault Injection', "Architecture", "Data Distribution Among Clients", 'Dataset',
            'Neuron Coverage Threshold', "Epochs",  'Number of Random Inputs',  'FaultyClientLocalization Time', 'Random Inputs Generation Time', "Sequence2Count", "Number of Faulty Clients"]
    outputs = []
    for key in cache.keys():
        if key.startswith("Acc_noise_rate"):
            # print(f"Cache Hit: {key}")
            continue
        print(f"\n\n   **** Exp {key}**** ")

        for r in cache[key]:
            d = {c: r[c] for c in cols}
            t = key.split("*** ")[-1]

            d["Random Fun"] = t.split("Rand_")[-1].split("_Transform")[0]
            d["Transform"] = t.split("_")[-1]

            d["Key Start"] = key.split("_")[0]
            # if "iid" in d["Key Start"]: # will also work for niid
            #     d["Key Start"] = "Unequal"

            d["Exp Setting"] = key
            d["Pre Trained"] = key.split("*** ")[0].split("_pre_")[-1]

            if d["Pre Trained"].find("train") == -1:
                d["Pre Trained"] = "trained_True (manual)"

            if d["Dataset"].find("cifar") != -1:
                d["Dataset"] = "CIFAR-10"
            elif d["Dataset"] == "femnist":
                d["Dataset"] = "FEMNIST"

            if "resnet" in d["Architecture"]:
                d["Architecture"] = "ResNet-" + \
                    d["Architecture"].replace("resnet", "")
            elif d["Architecture"] == "densenet121":
                d["Architecture"] = "DenseNet-121"
            elif d["Architecture"] == "vgg16":
                d["Architecture"] = "VGG-16"

            d['FaultyClientLocalization Time'] = d['FaultyClientLocalization Time']/d["Number of Random Inputs"]

            outputs.append(d)

    df = pd.DataFrame(outputs)
    df.rename(columns={'Number of Clients': 'Clients', 'Updated Detection Accuracy': 'Accuracy',
              "Random Inputs Generation Time": "Avg. Input Time (s)", "FaultyClientLocalization Time": "Avg. Detection Time (s)"}, inplace=True)
    df.to_csv(args.csv)
    print(f">> Results saved to {args.csv}")


def mainMultiFaulty(args):
    print(">> Loading cache")
    cache = Index(args.cache_dir)

    cols = ['Number of Clients', "Number of Random Inputs",  'Updated Detection Accuracy',  'Fault Injection', "Architecture", "Data Distribution Among Clients", 'Dataset',
            'Neuron Coverage Threshold', "Epochs",  'Number of Random Inputs',  'FaultyClientLocalization Time', 'Random Inputs Generation Time', "Sequence2Count", "Number of Faulty Clients", "Faulty Clients IDs", "Number of Faulty Clients"]
    outputs = []
    for key in cache.keys():

        if "vgg" in key:
            # print(f"Cache Hit: {key}")
            continue
        print(f"\n\n   **** Exp {key}**** ")

        for r in cache[key]:
            d = {c: r[c] for c in cols}
            t = key.split("*** ")[-1]

            d["Random Fun"] = t.split("Rand_")[-1].split("_Transform")[0]
            d["Transform"] = t.split("_")[-1]

            d["Key Start"] = key.split("_")[0]
            # if "iid" in d["Key Start"]: # will also work for niid
            #     d["Key Start"] = "Unequal"

            d["Exp Setting"] = key
            d["Pre Trained"] = key.split("*** ")[0].split("_pre_")[-1]

            if d["Pre Trained"].find("train") == -1:
                d["Pre Trained"] = "trained_True (manual)"

            if d["Dataset"].find("cifar") != -1:
                d["Dataset"] = "CIFAR-10"
            elif d["Dataset"] == "femnist":
                d["Dataset"] = "FEMNIST"

            if "resnet" in d["Architecture"]:
                d["Architecture"] = "ResNet-" + \
                    d["Architecture"].replace("resnet", "")
            elif d["Architecture"] == "densenet121":
                d["Architecture"] = "DenseNet-121"
            elif d["Architecture"] == "vgg16":
                d["Architecture"] = "VGG-16"

            d['FaultyClientLocalization Time'] = d['FaultyClientLocalization Time']
            d['Random Inputs Generation Time'] = d['Random Inputs Generation Time']
                
            print(f'-- {d["Faulty Clients IDs"]}')
            # d["Faultyion %"] = -1
            # d["Accuracy 2"] = 0
            all_clients_ids = set([c for c in range(d["Number of Clients"])])

            d["Accuracy 2"] = getAccuracy(all_clients_ids, d["Faulty Clients IDs"], d["Sequence2Count"])

            

            outputs.append(d)
        
    df = pd.DataFrame(outputs)
    df.rename(columns={'Number of Clients': 'Clients', 'Updated Detection Accuracy': 'Accuracy',
              "Random Inputs Generation Time": "Avg. Input Time (s)", "FaultyClientLocalization Time": "Avg. Detection Time (s)"}, inplace=True)
    df.to_csv(args.csv)
    print(f">> Results saved to {args.csv}")


def mainAnyCache2Csv(args):
    print(">> Loading cache")
    cache = Index(args.cache_dir)

    outputs = []
    for key in cache.keys():
        if key.startswith("Acc_noise_rate") == False:
            print(f"some other key: {key}")
            continue
        print(f"\n\n   **** Exp {key}**** ")
        d = cache[key]
        outputs.append(d)

    df = pd.DataFrame(outputs)
    df.to_csv(args.csv)
    print(f">> Results saved to {args.csv}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="",
                        help="experiments cache directory", required=True)
    parser.add_argument("--csv", type=str, default="",
                        help="output csv file", required=True)
    args = parser.parse_args()
    assert args.cache_dir.endswith("/"), "cache_dir should end with '/'"
    assert args.csv.endswith(".csv"), "csv file"
    # mainCache2CsvFaultyClientLocalization(args)
    # mainAnyCache2Csv(args)
    mainMultiFaulty(args)