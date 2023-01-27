import builtins
import os 
import time 
from ibmfl.aggregator.aggregator import Aggregator
from ibmfl.party.party import Party
# from debugger import DebuggerCLI
from queue import Queue
import sys
from diskcache import Index



class FedDebugFL:
    def __init__(self,breakpoint, parties):
        cache  = Index("debug_cache")
        cache.clear()
        cache["breakpoint"] = breakpoint
        self.num_parties = parties
    
    def partiesStart(self):
        # sys.stdout = io.StringIO()
        def startParty(pid):
            party = Party(config_file=f"examples/configs/iter_avg/pytorch/config_party{pid}.yml")
            party.start()
            party.register_party()
            party.proto_handler.is_private = False 

        for i in range(self.num_parties):
            startParty(i)
            time.sleep(0.1)
       
    def aggStart(self):
        pp =  5
        os.system(f'python examples/generate_data.py -n {self.num_parties} -d mnist -pp {pp}')
        os.system(f'python examples/generate_configs.py -f iter_avg -m pytorch -n {self.num_parties} -d mnist -p examples/data/mnist/random')
        aggregator = Aggregator(config_file="examples/configs/iter_avg/pytorch/config_agg.yml")
        aggregator.start()

        user_input = input("Press key to continue...") # wait for clients/parties to connect
        start = time.time()
        aggregator.start_training() 
        print(f">>... Simulation Total Time {time.time()-start}")
  
        

if __name__ == "__main__":    
    breakpoint =  {"round": 5, "status": False}
    parties = 10
    sim = FedDebugFL(breakpoint, parties)
    sim.aggStart()