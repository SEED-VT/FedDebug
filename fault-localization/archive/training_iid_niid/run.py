import sys
import os
import argparse
from diskcache import Index
sys.path.append("../")



def commandsToScript(commands, sname):
    with open(sname, "w") as f:
        for c in commands:
            f.write(c)
            f.write("\n")

def mainFaultyion():
    cache_name = "cache/fl_multi_faulty_scale/"
    checkpoints_dir_name = "checkpoints_multi_faulty_scale/"
    noise_rate =",".join(str(f) for f in [10])
    
    samplings = ["iid"]
    sampling = samplings[0]
    model_configs = []
    commands = []
    pretrained = 1
    lr = 0.001
    epochs = 10

    for clients in [10]:                      
        model_configs.append({"model": "resnet50", 'dataset': "femnist", "lr": lr,
                                        'weight_decay': 0.0001, "clients": clients, "batch_size": 512, "epochs": epochs})
        
        
    #     model_configs.append({"model": "vgg16", 'dataset': "femnist", "lr": 0.0001,
    #                                     'weight_decay': 0.0001, "clients": clients, "batch_size": 512, "epochs": 10})
    #     # model_configs.append({"model": "resnet50", 'dataset': "femnist", "lr": lr,
    #     #                                 'weight_decay': 0.0001, "clients": clients, "batch_size": 512, "epochs": epochs})

    # for clients in [50]:                      
    #     model_configs.append({"model": "densenet121", 'dataset': "cifar10", "lr": lr,
    #                                     'weight_decay': 0.0001, "clients": clients, "batch_size": 512, "epochs": epochs})

    # for clients in [30, 50]:                      
    #     model_configs.append({"model": "vgg16", 'dataset': "cifar10", "lr": lr,
    #                                     'weight_decay': 0.0001, "clients": clients, "batch_size": 512, "epochs": 10})
    #     model_configs.append({"model": "vgg16", 'dataset': "femnist", "lr": lr,
    #                                     'weight_decay': 0.0001, "clients": clients, "batch_size": 512, "epochs": 10})
        


            

    # for sampling in samplings:
    for config in model_configs:
        r =   2 # number of faulty clients
        faulty_clients = ",".join(str(p)for p in range(r))
        command = f"python cli_FL_training.py --cache_name {cache_name} --checkpoints_dir_name {checkpoints_dir_name}  --noise_rate {noise_rate} --sampling {sampling} --model {config['model']} --pretrained {pretrained} --epochs {config['epochs']} --lr {config['lr']} --weight_decay {config['weight_decay']} --dataset {config['dataset']} --batch_size {config['batch_size']} --clients {config['clients']} --faulty_clients_ids {faulty_clients} --storage ../storage/"
        commands.append(command)    
   
    commandsToScript(commands=commands, sname = "train_mul_faulty.sh")










# def mainGeneric():
#     faulty_clients_list = ["0"]
#     cache_name = "cache/fl_scale/"
#     checkpoints_dir_name = "checkpoints_scale/"
#     noise_rate =",".join(str(f) for f in [10])    
#     samplings = ["iid"]
#     model_configs = []
#     commands = []
#     pretrained = 1
#     lr = 0.001
#     epochs = 20
    
#     for clients in [200, 100 ,50, 25, 10]:                      
#     # for clients in [400]:    
#         model_configs.append({"model": "resnet18", 'dataset': "femnist", "lr": 0.001,
#                                         'weight_decay': 0.0001, "clients": clients, "batch_size": 512, "epochs": epochs})
#         model_configs.append({"model": "resnet34", 'dataset': "femnist", "lr": 0.001,
#                                         'weight_decay': 0.0001, "clients": clients, "batch_size": 512, "epochs": epochs})
#         model_configs.append({"model": "resnet50", 'dataset': "femnist", "lr": 0.001,
#                                         'weight_decay': 0.0001, "clients": clients, "batch_size": 512, "epochs": epochs})
            
            

#     for sampling in samplings:
#         for config in model_configs:
#             for faulty_clients in faulty_clients_list:
#                 command = f"python cli_FL_training.py --cache_name {cache_name} --checkpoints_dir_name {checkpoints_dir_name}  --noise_rate {noise_rate} --sampling {sampling} --model {config['model']} --pretrained {pretrained} --epochs {config['epochs']} --lr {config['lr']} --weight_decay {config['weight_decay']} --dataset {config['dataset']} --batch_size {config['batch_size']} --clients {config['clients']} --faulty_clients_ids {faulty_clients} --storage ../storage/"
#                 # print(command)
#                 # os.system(command)
#                 commands.append(command)
    
#     commandsToScript(commands=commands, sname = "train200-onwards.sh")

    
if __name__ == "__main__":
    mainFaultyion()
    # mainGeneric()

