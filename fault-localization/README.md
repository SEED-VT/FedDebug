# FedDebug: Faulty Client Localization

<!-- Please run <a target="_blank" href="https://colab.research.google.com/github/warisgill/FedDebug-Artifact/blob/main/artifact.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> to interact with fault localization technique.  -->

See `INSTALL.md` for further instructions on how to setup your environment for this repo.

## Data
CIFAR10 and FEMNIST data are downloaded automatically when running an evaluation script. CIFAR10 is downloaded from `PyTorch` library and FEMNIST is downloaded from `LEAF (cmu.edu)`. 

## FedDebug Performance

We evaluate FEDDEBUG on CIFAR-10 and FEMNIST. Both are considered as gold standard to evaluate both FL frameworks. FEMNIST is a modified version of MNIST presented in the FL LEAF Benchmark and the Non-IID Bench. The FEMNIST dataset contains over 340K training and over 40K testing grayscale, 28x28 images spanning ten different classes. CIFAR-10 contains 50K training 32x32 RGB images that span ten different classes and 10K instances for testing. We adopt popular CNN models i.e., ResNet, VGG, and DenseNet architectures [16], [19], [50]. We set the learning rate between 0.0001 and 0.001, the number of epochs between 10 and 25, the batch size from 512 to 2048, and the weight to 0.0001

These following configurations are available for all architechtures (resnet18, resnet34, resnet50, Desnet121, vgg16) and Dataset (FEMNIST) in `iid` and `niid` settings:
 
| Total Clients       | Dataset       | Architecture     | Faulty Clients             | 
| ------------------- | ------------- | ---------------- | -------------------------- | 
| 30                  | CIFAR10       | ResNET-50        | 2                          | 
| 30                  | CIFAR10       | ResNET-50        | 3                          | 
| 30                  | CIFAR10       | ResNET-50        | 5                          | 
| 30                  | CIFAR10       | ResNET-50        | 7                          | 
| 50                  | CIFAR10       | ResNET-50        | 2                          | 
| 50                  | CIFAR10       | ResNET-50        | 3                          | 
| 50                  | CIFAR10       | ResNET-50        | 5                          | 
| 50                  | CIFAR10       | ResNET-50        | 7                          | 

### Where to edit in code to run these configurations:
#### 1. Total Clients:
It can be either 30 or 50 as mentioned in the paper. In `artifact.ipynb`'s second cell, on `line 61`, you can set its value.
> Keep it under 30 clients and use Resnet18, Resnet34, or Densenet to evaluate on google colab
#### 2. Datasets:
It can be either CIFAR10 or FEMNIST as mentioned in the paper. In `artifact.ipynb`'s second cell, on `line 62`, you can set its value.
> It can be multiple clients separated by comma e.g. "0, 1, 2" but keep under total clients
#### 3. Faulty Clients:
It can be either 2, 3, 5, or 7 as mentioned in the paper. In `artifact.ipynb`'s second cell, on `line 59`, you can set its value.
#### 4. Architechture:
It can be either resnet18, resnet34, resnet50, Desnet121, or vgg16 as mentioned in the paper. In `artifact.ipynb`'s second cell, on `line 56`, you can set its value.
#### 5. Setting:
It can be either `iid`, or `niid` as mentioned in the paper. In `artifact.ipynb`'s second cell, on `line 64`, you can set its value.
#### 6. Epocs:
It can be in range of 10-25 as mentioned in the paper. In `artifact.ipynb`'s second cell, on `line 57`, you can set its value.
#### 6. Noise:
It can be in range of 0 to 1 as mentioned in the paper. In `artifact.ipynb`'s second cell, on `line 63`, you can set its value.


## Evaluation and Results

The results from the above experiment prints FedDebug performance statistics that can be used to generate graphs in the paper.
For example:
1. You can generate the localization performance graph of FedDebug at various noise rates.
2. You can generate the localization performance graph of FedDebug with multiple faulty clients in different settings. etc.
