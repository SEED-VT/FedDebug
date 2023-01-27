# FedDebug: Faulty Client Localization

  

`FedDebug's` novel automated fault localization approach precisely identifies the faulty client without ever needing any test data or labels. To have a measurable impact on the global model, a faulty client’s model must behave differently than the regular clients. Every client in an FL application has the same model architecture, so their internal behaviors are comparable. Based on this insight, generates random inputs adapts differential testing to FL domain. It captures differences in the models’ execution via neuron activations instead of output labels to identify diverging behavior of a faulty client. Note that auto-generated data does not include the class label and thus we cannot use it as an oracle.

  

## Installation

  

### Google Colab Setup

> ***Make sure you configure notebook with GPU: Click Edit->notebook settings->hardware accelerator->GPU***

Copy & paste the following commands in the ***first cell*** of notebook (e.g., `artifact.ipynb`) on `Google Colab`.

```

!pip install pytorch-lightning

!pip install diskcache

!pip install dotmap

!pip install torch torchvision torchaudio

!git clone https://github.com/SEED-VT/FedDebug.git

# appending the path

import sys

sys.path.append("FedDebug/fault-localization/")

```

Now you can run the `artifact.ipynb` (<a  target="_blank"  href="https://colab.research.google.com/github/SEED-VT/FedDebug/blob/main/fault-localization/artifact.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"  alt="Open In Colab"/></a>). You can run notebooks containing `FedDebug` code with the above instructions. **Note:** *You can uncomment the commands instead of copy & pasting if above commands are already in the given notebook.*

  

### Local Setup

1. Create a conda environment for example:

`conda create --name feddebug`

2. Activate environment:

`conda activate feddebug`

3. Install necessary packages:

```

pip install pytorch-lightning

pip install diskcache

pip install dotmap

pip install jupyterlab

pip install torch torchvision torchaudio

pip install jupyterlab

```
4. Clone `FedDebug` repository and switch to it.

```
git clone https://github.com/SEED-VT/FedDebug.git
cd FedDebug/fault-localization
```

>Note: Make sure you are in project directory `~/FedDebug/fault-localization`.

  

5. Run `jupyter-lab` command in `fault-localization` directory. It will open the jupyter-notebook in the project directory. Open and run `artifact.ipynb`. This should work without any errors.

See `INSTALL.md` for further instructions on how to setup your environment for fault localization.

***Note: To run locally, make sure you have an NVIDA GPU in your machine.***

  
  ## Evaluations 
### Datasets
`cifar10` and `feminist` data are downloaded automatically when running an evaluation script. 
 
 ### Models
`resnet18, resnet34, resnet50, vgg16, and Densenet121` are CNN model architectures used in evaluation.  

### To Evaluate Any Experimental Setting.
We provide a single notebook `artifact.ipynb` (<a  target="_blank"  href="https://colab.research.google.com/github/SEED-VT/FedDebug/blob/main/fault-localization/artifact.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"  alt="Open In Colab"/></a>) with all contrling arguments to evaluate any experimental settings.  It can be executed on `Google Colab` but keep clients under 30. Colab has limited computation power and RAM.  

```
args.model = "resnet50"  # [resnet18, resnet34, resnet50, Desnet121, vgg16]

args.epochs = 2  # range 10-25

args.dataset = "cifar10"  # ['cifar10', 'femnist']

args.clients = 5  # keep under 30 clients and use Resnet18, Resnet34, or Densenet to evaluate on Colab

args.faulty_clients_ids = "0"  # can be multiple clients separated by comma e.g. "0,1,2" but keep under args.clients clients and at max less than 7

args.noise_rate = 1  # noise rate 0.1 to 1

args.sampling = "iid"  # [iid, "niid"]
```

## Results:


  