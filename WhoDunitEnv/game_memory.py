import pandas as pd
import json, copy, shutil, os
import logging
import pickle
    

class IntelGameMemory:
    def __init__(self, path, env, dump_logprobs=False):
        self.path = path
        self.system_prompts = {'accuser': env.accuser.system_prompt, 'intel': env.intel.system_prompt}
        self.game_flow = []
        self.monitors = []
        self.suspect_knowledge = []
        self.env = env
        self.first_turn = 0
        self.failed = False

        self.dump_logprobs = dump_logprobs
        self.logprobs_dict = dict()
    
    def dump_memory(self, save_path=None):
        if self.failed:
            return
        env_info = self.env.to_dict()
        all_dictionary = {
            'system_prompts': self.system_prompts,
            'game': self.game_flow,
            'monitor': self.monitors,
            'env_info': env_info,
            'blunders': 
            {
                'accuser': self.env.accuser.blunders,
                'intel': self.env.intel.blunders
            },
            'errors':
            {
                'accuser': self.env.accuser.errors,
                'intel': self.env.intel.errors
            },
            "first_turn": self.first_turn
        }
        if save_path is None:
            save_path = self.path
        with open(save_path, 'w') as f:
            try:
                json.dump(all_dictionary, f, indent=4)
            except ValueError as e:
                print(self.agent_replies)
                raise e

        if self.dump_logprobs:
            with open(save_path.replace(".json", ".scores_json"), 'w') as f:
                json.dump(self.logprobs_dict, f)
            

    def delete(self):
        usual_path = self.path.replace('.csv', '.json')
        if os.path.exists(usual_path):
            backup_path = usual_path + '.del'
            shutil.move(usual_path, backup_path)
        else:
            self.dump_memory(save_path=usual_path + '.del')
        self.failed = True
    
            
    def log_turn(self, turn_info):
        self.game_flow.append(turn_info)
        self.suspect_knowledge.append(copy.deepcopy(self.env.suspect_knowledge))
    
    def log_monitor(self, monitor_info):
        self.monitors.append(monitor_info)

class SymmGameMemory:
    def __init__(self, path, env, dump_logprobs=False):
        self.path = path
        self.system_prompts = {i: env.agents[i].system_prompt for i in range(len(env.agents))}
        self.game_flow = []
        self.suspect_knowledge = []
        self.env = env
        self.first_turn = 0
        self.failed = False

        self.dump_logprobs = dump_logprobs
        self.logprobs_dict = dict()
    
    def dump_memory(self, save_path=None):
        if self.failed:
            return
        env_info = self.env.to_dict()
        all_dictionary = {
            'system_prompts': self.system_prompts,
            'game': self.game_flow,
            'env_info': env_info,
            "first_turn": self.first_turn
        }
        if save_path is None:
            save_path = self.path
        with open(save_path, 'w') as f:
            try:
                json.dump(all_dictionary, f, indent=4)
            except ValueError as e:
                print(self.agent_replies)
                raise e

        if self.dump_logprobs:
            with open(save_path.replace(".json", ".scores_json"), 'w') as f:
                json.dump(self.logprobs_dict, f)
            

    def delete(self):
        usual_path = self.path.replace('.csv', '.json')
        if os.path.exists(usual_path):
            backup_path = usual_path + '.del'
            shutil.move(usual_path, backup_path)
        else:
            self.dump_memory(save_path=usual_path + '.del')
        self.failed = True
    
            
    def log_turn(self, turn_info):
        self.game_flow.append(turn_info)
        self.suspect_knowledge.append(copy.deepcopy(self.env.suspect_knowledge))