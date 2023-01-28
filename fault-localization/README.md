# FedDebug: Faulty Client Localization
  
At a given faulty round in an FL campaign, `FedDebug` can automatically identify faulty clients without needing test data or labels. The following steps provide a walk-through of this feature using both reconfigurable computational notebooks as well as local python setup. 

## 2.1 Installation

### Google Colab Setup

> ***Make sure you configure the notebook with GPU: Click Edit->notebook settings->hardware accelerator->GPU***


Copy & paste the following commands in the ***first cell*** of notebook (e.g., `artifact.ipynb`) on `Google Colab`.

```
!pip install pytorch-lightning
!pip install matplotlib
!pip install diskcache
!pip install dotmap
!pip install torch torchvision torchaudio
!git clone https://github.com/SEED-VT/FedDebug.git
# appending the path
import sys
sys.path.append("FedDebug/fault-localization/")
```

Now you can run the `artifact.ipynb` (<a  target="_blank"  href="https://colab.research.google.com/github/SEED-VT/FedDebug/blob/main/fault-localization/artifact.ipynb"><img  src="https://colab.research.google.com/assets/colab-badge.svg"  alt="Open In Colab"/></a>). You can run notebooks containing `FedDebug` code with the above instructions. **Note:**  *You can uncomment the commands instead of copy & pasting if above commands are already in the given notebook.*

The notebook is configured to use the following default settings, where the first client (client ID 0) is faulty (with noisy labels that distort the global FL model).  

```
args.model = "resnet50" # [resnet18, resnet34, resnet50, densenet121, vgg16]
args.epochs = 2  # range 10-25
args.dataset = "cifar10" # ['cifar10', 'femnist']
args.clients = 5 # keep under 30 clients and use Resnet18, Resnet34, or Densenet to evaluate on Colab 
args.faulty_clients_ids = "0" # can be multiple clients separated by comma e.g. "0,1,2"  but keep under args.clients clients and at max less than 7 
args.noise_rate = 1  # noise rate 0 to 1 
args.sampling = "iid" # [iid, "niid"] 
```

Once the entire notebook is executed, the logs report the progress and eventually print the faulty client's id. This is printed in the form `+++ Faulty Clients {0}` where 0 is the client ID for each auto-generated input (default is 10). It also reports the final localization accuracy. 

  

### Local Setup

  

  

1. Create a conda environment for example: `conda create --name feddebug`

2. Activate environment: `conda activate feddebug`

3. Install necessary packages:
```
pip install pytorch-lightning
pip install matplotlib
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

***Note: To run locally, make sure you have an NVIDIA GPU in your machine.***

  
## 2.2 Evaluations

### Datasets
`cifar10` and `feminist` data are downloaded automatically when running an evaluation script.

### Models
`resnet18, resnet34, resnet50, vgg16, and Densenet121` are CNN model architectures used in evaluation.

### To Evaluate Any Experimental Setting.
We provide a single notebook `artifact.ipynb` (<a  target="_blank"  href="https://colab.research.google.com/github/SEED-VT/FedDebug/blob/main/fault-localization/artifact.ipynb"><img  src="https://colab.research.google.com/assets/colab-badge.svg"  alt="Open In Colab"/></a>) with all controlling arguments to evaluate any experimental settings. It can be executed on `Google Colab` but keep clients under 30. Colab has limited computation power and RAM.

```
args.model = "resnet50" # [resnet18, resnet34, resnet50, densenet121, vgg16]
args.epochs = 2 # range 10-25
args.dataset = "cifar10" # ['cifar10', 'femnist']
args.clients = 5 # keep under 30 clients and use Resnet18, Resnet34, or Densenet to evaluate on Colab
args.faulty_clients_ids = "0" # can be multiple clients separated by comma e.g. "0,1,2" but keep under args.clients clients and at max less than 7
args.noise_rate = 1 # noise rate 0.1 to 1
args.sampling = "iid" # [iid, "niid"]

```

  

  

## 2.3 Results:

Although `artifact.ipynb` is sufficient to evaluate any configuration of `FedDebug`, we further extended it to reproduce the major results with just a single click on `Google Colab` except the scalability result `Reproduce_Figure9.ipynb`. You can reproduce the scalability result on a local machine which has enough resources to train 400 models.

  

>***Note: We have scaled down the the experiments (e.g., reduce the number of epochs, small number of clients, etc) to assist reviewers to quickly validate the fundamentals of `FedDebug` and fit with `Google Colab` resources. However, you are welcome to test it with any valid FL configuration. Furthermore, if you see an unexpected result, please increase the number of `epochs` to the value mention in `Section V: Evaluations` and read `Section V.D Threat To Validity section`.***

  

-  **`Reproduce_Figure4-Figure7.ipynb` <a  target="_blank"  href="https://colab.research.google.com/github/SEED-VT/FedDebug/blob/main/fault-localization/Reproduce_Figure4-Figure7.ipynb"><img  src="https://colab.research.google.com/assets/colab-badge.svg"  alt="Open In Colab"/></a>:** Figure 4 shows that lower `noise rate` does not impact the performance of the global model and only high `noise rate` degrade its performance. Although low noise rates do not deteriorate global model significantly, `Figure 7` shows that `FedDebug` has the capability to localize the `faulty clients` with low noise rate.

  

-  **`Reproduce_Figure9.ipynb`:** In several prior work simulations, researchers have use a very few clients to participate in a round (usually 10~clients). We challenged our approach to consider hundreds of clients to participate in a round and this notebook shows the result of this evaluation. This notebook cannot run on free available resources of the `Google Colab`. We conducted this experiment on an AMD 16-core processor, with 128 GB RAM and an NVIDIA Tesla T4 GPU.

  

-  **`Reproduce_Figure10.ipynb` <a  target="_blank"  href="https://colab.research.google.com/github/SEED-VT/FedDebug/blob/main/fault-localization/Reproduce_Figure10.ipynb"> <img  src="https://colab.research.google.com/assets/colab-badge.svg"  alt="Open In Colab"/> </a>:** This notebook reproduces the results of `Figure 10` which shows the `FedDebug` performance on different neuron activation thresholds. We found activation threshold less than 0.01 to perform well across all the experiments. By default, `FedDebug` uses neuron activation threshold of `0.003`.

  

-  **`Reproduce_Table1-Table2.ipynb` <a  target="_blank"  href="https://colab.research.google.com/github/SEED-VT/FedDebug/blob/main/fault-localization/Reproduce_Table1-Table2.ipynb"> <img  src="https://colab.research.google.com/assets/colab-badge.svg"  alt="Open In Colab"/> </a>:** Table 1 and Table 2 show the performance of `FedDebug` in IID and Non-IID data distribution settings. This notebook contains the representative settings of these two tables. Feel free to test other configurations as well.
