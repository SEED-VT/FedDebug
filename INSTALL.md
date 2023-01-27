# Local Environment Setup

***To run locally, make sure you have an NVIDA GPU in your machine.***

### Steps to setup locally

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

cd FedDebug
```
>Note: Make sure you are in project directory `~/FedDebug`.

 
5. Run `jupyter-lab` command in `FedDebug` directory. It will open the jupyter-notebook in the project directory.

- Open `artifact.ipynb`

- Run second cell.

This should work without any errors.

  

# Google Colab Setup

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
sys.path.append("/content/FedDebug")
```
Now you can run the second cell of `artifact.ipynb`. You can run notebooks containing  `FedDebug` code with the above instructions. **Note:** *You can uncomment the commands instead of copy & pasting if above commands are already in the given notebook.* 