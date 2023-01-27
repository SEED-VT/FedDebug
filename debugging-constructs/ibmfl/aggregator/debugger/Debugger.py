from diskcache import Index
from subprocess import call
import os
import numpy as np
import time
from termcolor import colored

class Debugger:

    def __init__(self,**kwargs):
        self.cache  = Index("debug_cache")
        breakpoint2info = self.cache.get("breakpoint",None)
        self.current_round = breakpoint2info["round"]
        
        self.round_keys  = [k for k in self.cache.keys() if k.startswith("round")]
        self.total_rounds = len(self.round_keys)
   
        round2info =  self.cache.get(f"round{self.current_round}", {})
        self.parties = round2info.get("parties",[])
        self.party2metric = round2info.get("parties_metrics", {})

        while len(self.parties) <1:
            print("Wait... ", end="\r")
            time.sleep(1)
            round2info =  self.cache.get(f"round{self.current_round}", {})
            self.parties = round2info.get("parties",[])
            self.party2metric = round2info.get("parties_metrics", {})

        self.breakpoint_status = {"retrain_round_id":-1}

    def getTotalTrainingRounds(self): # done
        return  len(self.round_keys)
    
    def _updateRounds(self):
        self.round_keys  = [k for k in self.cache.keys() if k.startswith("round")]
        self.total_rounds = len(self.round_keys)
        round2info =  self.cache.get(f"round{self.current_round}", {})
        self.parties = round2info.get("parties",[])
        self.party2metric = round2info.get("parties_metrics", {})
        
    def nextRound(self): #
        self._updateRounds()        
        self.current_round += 1
        if self.current_round >= self.total_rounds:
            self.current_round = self.total_rounds - 1        
        return self.getRound(self.current_round)
         
    def prevRound(self): #
        self.current_round -= 1
        if self.current_round < 0:
            self.current_round = 0
        return self.getRound(self.current_round)
    
    def isRoundValid(self,round_id):
        return round_id < self.total_rounds 

    def getRound(self, round_id):
        if self.isRoundValid(round_id):
            self.current_round = round_id
            return  self.current_round #self.cache[f"round{self.current_round}"]
        else :
            return None 
    
    def evaluAteModel(self,model,data):
        return None

    def getPartiesInCurrentRound(self):
        return self.cache[f"round{self.current_round}"]['parties']

    def displaySummary(self):
        self._updateRounds()
        print("\n")
        print(f"    -Round: {self.current_round}")
        print(f"    -Clients : {(', ').join(self.parties)}")
        print(f"    -Total Training Rounds : {self.total_rounds}")
        print("\n")
    
    def displayParties(self):
        # print(f"\n    -Clients -> {', '.join(self.parties)} \n")
        print(f"\n   -Clients  & Metrics \n")
        for k,v in self.party2metric.items():
            print(f"    {k} -> {v}")

    def _beforeCommand(self):
        self.cache["command_result"] = None

    def _waitForResults(self):
        while True:
            print("Wait... ", end="\r")
            time.sleep(1)
            if self.cache.get("command_result", None) is not None:
                break 

    def removeParty(self, pid):
        self._beforeCommand()
        self._updateRounds()

        if pid in self.parties:
            print("Removing client", pid)
            faulty_parties = self.cache.get("faulty_parties",[])
            faulty_parties.append(pid)
            self.cache["command"] = "remove_party"
            self.cache["faulty_parties"] = {"faulty_parties": faulty_parties, "round": self.current_round}
            self._waitForResults()
            self.breakpoint_status["retrain_round_id"] = self.current_round
            self.closeDebugger()
        else:
            print(colored(f"Invalid client id {pid} for round {self.current_round}", "red"))
                
    def partialAgg(self):
        self._beforeCommand()
        self._updateRounds()
        pids = input(colored("Enter client ids to aggregate & evaluate, separated by comma:  ", "cyan"))
        pids = pids.split(",")
        

        diff = set(pids) - set(self.parties)
        if len(diff) == 0:
            print("     Aggregating Clients", ', '.join(pids))
            partial_agg = {"pids" : pids, "round": self.current_round}
            self.cache["command"] = "agg"
            self.cache["partial_agg"] = partial_agg
            self._waitForResults()
            # print("Result ()", self.cache["command_result"])
            d = self.cache["command_result"]['result']
            # print(d)
            # print(list(d.keys()))
            cliets_accs = d['accuracies']
            print(f"    Partial Aggregated Models Accuracy on clients")
            for k,v in cliets_accs.items():
                print(f"    Client {k} -> {v}")
        else:
            print(colored(f"Invalid client ids {diff} for round {self.current_round}", "red"))

    
    def stepIn(self):            
        parties = self.getPartiesInCurrentRound()
        print(f"Stepping in: Round {self.current_round}")
        help = f"\n {colored('*agg ->','green')} partial aggregate replies of clients\n {colored('*remove client <client_id> ->','green')} remove client from the given round \n {colored('*step out ->','green')} step out of round\n {colored('*resume ->','green')} resume training\n {colored('*clear ->','green')} clear screen\n {colored('*help ->','green')} display help \n" 
        help_resume = f"\n "  
        note = colored('+Note: remove client <client_id> mimic the fix and replay. Any fault localizatin technique can be integrated here to localize a faulty client.\n','red')
        # help = "test"
        commands = {
            # ('query', 'to'): lambda: print(f"{cli.queryToAPartyInRound(args[2], cli.current_round)}"),
            # ('reply', 'of'): lambda: print(f"{cli.replyofPartyInRound(args[2], cli.current_round)}"),
            ("agg",): lambda: self.partialAgg(),
            ('step', 'out'): lambda: print(f"Stepping out from round {self.current_round}"),
            ("remove", "client"): lambda: self.removeParty(args[2]),
            ("ls"): lambda: self.displayParties(),
            ("resume",): lambda: self.resume(),
            ('help',): lambda: print(help,note),
            ('clear',): lambda: call('clear' if os.name =='posix' else 'cls'),
        }

        while True:
            s = input(colored(f"[Step-in Round-{self.current_round}]> ", "blue"))

            s = s.strip().lower() 
            
            args = tuple([arg for arg in s.split(' ') if arg != ''])
            if len(args) == 0:
                continue
            
            if args[:2] in commands:
                commands[args[:2]]()
                if args[0] == 'step' and args[1] == 'out':
                    break
            elif args[0] in commands:
                commands[args[0]]()
                if args[0] == 'step' and args[1] == 'out':
                    break
            else:
                print(colored(f"    Invalid command.", "red"))

    def closeDebugger(self):
        self.cache["breakpoint_status"] = self.breakpoint_status
        self.cache["breakpoint"] = {"round":-1, "party":-1}
        exit(0)
    
    def resume(self):
        self.cache["command"] = "resume"
        self.closeDebugger()

    def handle_global_commands(self, args):
        help = f"\n {colored('*step in ->','green')} step into round\n {colored('*step next ->', 'green')} - step to next round\n {colored('*step back ->','green')} step to previous round\n {colored('*resume ->','green')} resume debugger\n {colored('*clear ->','green')} clear screen\n {colored('*ls ->', 'green')} display the summary of a round\n {colored('*help ->','green')} display commands and their usage\n "
        # help = "test"
        commands = {
            ('step', 'in'): lambda: self.stepIn(),
            ('step', 'next'): lambda: print(f"Round: {self.nextRound()}"),
            ('step', 'back'): lambda: print(f"Round: {self.prevRound()}"),
            "resume": lambda: self.resume(),
            'ls': lambda: self.displaySummary(),
            "help": lambda : print(help),
            'clear': lambda: call('clear' if os.name =='posix' else 'cls'),

        }
        # print(f"args: {args}, len: {len(args)}")

        if len(args) > 1 and args[:2] in commands:
            commands[args[:2]]()
        elif len(args) == 1  and args[0] in commands:
            commands[args[0]]()
        else:
            print(colored(f"    Invalid command.", "red"))



def runDebugger():
    debugger_interface = Debugger()
    time.sleep(1)
    while 1:
        try:
            s = input(colored(f"[R-{debugger_interface.current_round}]>> ", "green"))
        except EOFError:
            s = "You closed the input stream"
            print(s)
            continue
        s = s.lower().strip()
        args = tuple([arg for arg in s.split(' ') if arg != ''])
        if len(args) == 0:
            continue
        debugger_interface.handle_global_commands(args)

    