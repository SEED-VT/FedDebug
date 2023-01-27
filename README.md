# FedDebug: Systematic Debugging for Federated Learning Applications

*Waris Gill, Ali Anwar, and Muhammad Ali Gulzar. Feddebug: Systematic debugging for federated learning applications. In Proceedings of the 45th International Conference on Software Engineering (ICSE '23). Association for Computing Machinery, New York, NY, USA.*

The ArXiv version of the manuscript is avaibable at : [FedDebug](https://arxiv.org/abs/2301.03553)

FedDebug enables interactive and automated fault localization in Federation Learning (FL) applications. It adapts conventional debugging practices in FL with its breakpoint and fix & replay featur and it offers a novel differential testing technique to automatically identify the precise faulty clients.FedDebug tool artifact comprises a two-step process
  - Interactive Debugging Constructs integrated with IBMFL framework via **FL simulation in a Docker Image**  
  - Automated Faulty Client Localization  via **Google Colab Notebooks**


# 1. Interactive debugging of FL Applications in IBMFL with FedDebug
FedDebug's interactive debugging module takes inspiration from traditional debuggers, such as gdb, and enables real-time interactive debugging on a simulation of a live FL application. 

Below, we provide step-by-step walkthrough of FedDebug's interactive debugging features in IBMFL. For ease of installation, we offer pre-configured *Docker Image* of FedDebug enabled IBMFL.  

## Step 1.1: Build FedDebug's Docker Image and Initiate
Go to `debugging constructs` directory (`cd debugging-constructs`). Please type `"docker build -t ibmfl ."`.  It will build docker form the `Dockerfile`. To run and interact with `ibmfl` docker type the following command `docker run -it ibmfl`. It will start the docker and now you can interact with it. Type `ls` in the docker shell to display the current directory content to check whether everything is installed correctly.  You can check more about dockers on the following link [Docker Tutorial](https://docs.docker.com/get-started/).

```
cd debugging-constructs
docker build -t ibmfl .
docker run -it ibmfl
ls
```





![docker](figures/docker.png)



### Tmux Tutorial
In this tutorial, we will be using Tmux to simulate a distributed FL environment in Docker, representing both FL clients and aggregator. Tmux is a terminal multiplexer that allows you to split your terminal screen horizontally. Tmux allows us to interact with the client and aggregator side of IBMFL and seemlesly move between those two interfaces. To start tmux, simply type '`tmux`' in the terminal. You can check more about `tmux` on this link [Tmux Quick Tutorial](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/).
  
### Splitting Screen in Tmux
To split the screen horizontally, type in a tmux session ` Ctrl + b` followed by `shift + " `. It will split the screen into two terminals. 
<!-- To split the screen horizontally, type in a tmux session ` Ctrl + b + " `. It will split the screen into two terminals.  -->

![tmux image](figures/tmux1.png)
 
You can move between terminals (panes) by pressing `Ctrl + b` followed by the `up` and `down` arrows keys. 

  
## Step 1.2: Running the Aggregator in FedDebug enabled IBMFL
In one of the tmux terminals, type `python sim_agg.py` to run the aggregator.
![agg start](figures/python_sim_agg_py.png)

After running this command, you will see the following output: "Press key to continue." Right now, ***do not press any key*** and move to another tmux terminal.

![agg start](figures/agg_start_output.png)

  
## Step 1.3: Running the Parties/Clients in FedDebug enabled IBMFL
In the second terminal, type `python sim_party.py`. This will start the ten parties and connect them to the second aggregator.
![agg start](figures/python_sim_party.png)
After running this command, you will see the following output:
![parties connected](figures/parties_connected.png)

## Step 1.4: Starting the Training
Move back to the aggregator terminal and press `enter key`. This will start the training from the aggregator for `10 rounds with 10 clients`.  Currently, breakpoint is set at round 5, so the terminal will stop displaying logs after `round 5` and you will see the following output:

![breakpoint](figures/breapoint.png). 

***Note: you can change the `breakpoint` round id in `sim_agg.py`*** 

## Step 1.5: FedDebug Interactive Debugging Constructs 

**`help:`** You can type `help` to see all the commands available at the round level in the debugger.

![round help](figures/round-help.png)

  
**`ls:`** The `ls` command at the round level will display generic information about the round.

![ls round](figures/ls.png)

**`step next, step back, and step in:`** You can also use the `step next`, `step back`, and  `step in` commands to navigate through the rounds.

![step next-back](figures/step-next-in-back.png)


## Step 1.6: Navigating inside a Round with FedDebug

After `step in` you can also type `help` to see the commands available inside a round.

![client help command](figures/client-help.png)

 
**`ls:`** You can use the `ls`  command inside a round to display all the clients in the round with their accuracies. 

**`agg:`** Similarly can also use the `agg` to partially aggregate the models from a subset of clients for the partial evaluation to inspect their performance. 

**Note: We have replace `step in` command inside the round with `agg` to avoid any confusion with  `step in` command of  rounds.** 

  

![ls and agg commands](figures/client-ls-agg.png)

 
**`step out:`** To leave a round, you can also use the `step out` command to step out of a round.
![step out command](figures/step-out.png)

 
## Step 1.7: Removing a Client with FedDebug

**`remove client <client id>:`** Suppose that you identify a faulty client in `round 8`, you can remove its contribution from the round using the `remove client <client id>` command. 

![remove client command](figures/remove-client.png)

This will resume the training from `round 9` instead of `round 5`, and the faulty client will not be included from `round 8`.

![Result of remove client command](figures/remove-client-result.png)

**Note: After removing the client, the training is complete. **
 
## Step 1.8: Resume after a debugging session 

To check the functionality of  `resume` command, restart the aggregator (`python sim_agg.py`) and clients ((`python sim_party.py`))  as explained above. Perform some actions (e.g., `step in, step out, ls` etc ), except  `remove client`. Suppose that there is no faulty client and you want resume the training without any further debugging. You can type `resume` to resume training and you will see that the aggregator immediately displays the the output of all the rounds without any retraining.  

![resume](figures/resume.png).


# 2. Automated faulty Client Localization with FedDebug.

WARIS : TODO HEre
GIVe colab link and show steps
## Step  2.1
## Step 2.2 etc. 



