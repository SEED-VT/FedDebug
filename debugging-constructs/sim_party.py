# %%
import sys
import os
from ibmfl.party.party import Party
import time


clients = 10



def startParty(pid):
    party = Party(config_file=f"examples/configs/iter_avg/pytorch/config_party{pid}.yml")
    party.start()
    party.register_party()
    party.proto_handler.is_private = False 




# press key to continue...
# key = input("Press key to register parties...")

# %%
for i in range(clients):
    startParty(i)
    time.sleep(0.1)

# %%



