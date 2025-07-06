from suspect_creator import SuspectCreator
from suspect import Suspect
from variants.symmetric.symm_agent import SymmAgent
from consts import *
import numpy as np
import json
import utils

DEFAULT_TASK_INFO = {
    "general": {
        "suspect_count": 20,
        'max_turn_count': 20
    },
    "message_file": 'json_files/messages_symmetric_positive.json'
}



class SymmEnv:
    def __init__(self, agent_pipe, task_info=DEFAULT_TASK_INFO, attribute_file=DEFAULT_ATTRIBUTE_FILE):
        for k in DEFAULT_TASK_INFO.keys():
            if k not in task_info:
                task_info[k] = DEFAULT_TASK_INFO[k]

        if task_info['general'].get('golden_intel', False):
            self.handle_intel = self.handle_intel_golden
        
        self.attribute_file = attribute_file
        self.message_file = task_info['message_file']
        self.task_info = task_info
        self.knowledge_size = task_info['general']['knowledge_size']

        # Initialize suspects for the environment, and choose culprit.
        self.suspect_creator = SuspectCreator(self.attribute_file)
        self.suspects = self.suspect_creator.create_suspects(self.task_info['general']['suspect_count'])
        self.culprit_index = np.random.randint(self.task_info['general']['suspect_count'])
        self.culprit = self.suspects[self.culprit_index]
        self.suspect_count = len(self.suspects)
        self.possible_suspects = set(range(1, self.suspect_count + 1))

        self.suspect_knowledge = {k: np.zeros(self.suspect_count) for k in self.suspect_creator.attributes}

        # Create Agents..
        self.agent_count = self.task_info['general']['agent_count']
        self.agents = []
        for i in range(self.agent_count):
            agent = SymmAgent(agent_pipe, AGENT_NAMES[i], 'detective', len(self.suspects), self.suspect_creator.attributes)
            self.agents.append(agent)

        self.messages = dict()
        self.load_message_file()
        self.setup(False)
        self.turn_count = 0
        self.cur_player = 0
        self.max_turn_count = task_info['general']['max_turn_count']
        self.previous_run_turns = 0
        self.reset_count = 0

    def reset(self):
        self.suspect_knowledge = {k: np.zeros(self.suspect_count) for k in self.suspect_creator.attributes}
        self.possible_suspects = set(range(1, self.suspect_count + 1))
        for agent in self.agents:
            agent.reset()
        self.turn_count = 0
        self.cur_player = 0
        self.previous_run_turns = 0

    @property
    def game_turn(self):
        return self.turn_count - self.previous_run_turns

    def assign_knowledge(self, agent):
        """
        Assigns facts to the given agent. See suspect.get_fact for more info
        :param agent: An investigator agent
        :return: None
        """
        agent.culprit_knowledge = []
        facts = self.culprit.get_fact(self.knowledge_size)
        agent.add_knowledge(facts)
        for i in range(len(agent.culprit_knowledge)):
            assert self.culprit.is_fact_true(agent.culprit_knowledge[i])

    def get_possible_suspects(self, facts):
        possible_suspects = set(range(self.suspect_count))
        for fact in facts:
            new_possible_suspects = set()
            for suspect_id in possible_suspects:
                if self.suspects[suspect_id].is_fact_true(fact):
                    new_possible_suspects.add(suspect_id)
            possible_suspects = new_possible_suspects
        assert self.culprit_index in possible_suspects
        return possible_suspects

    def add_to_communication(self, message):
        for agent in self.agents:
            agent.comm_channel.append(message)

    def get_eliminated_suspects(self):
        return set(range(1, self.suspect_count + 1)) - self.possible_suspects

    def set_from_dict(self, info_dictionary):
        self.reset()
        self.suspect_count = info_dictionary['suspect_count']
        self.suspects = []
        for i in range(self.suspect_count):
            self.suspects.append(Suspect(i, info_dictionary['suspects'][i]))
        self.culprit_index = info_dictionary['culprit']
        self.culprit = self.suspects[self.culprit_index]

        for i in range(len(info_dictionary['agents'])):
            self.agents[i].reset() 
            self.agents[i].culprit_knowledge = info_dictionary['agents'][str(i)]
        self.load_message_file()
        self.setup(False)

    def to_dict(self):
        info = dict()
        info['suspect_count'] = self.suspect_count
        info['suspects'] = [self.suspects[i].suspect_info for i in range(self.suspect_count)]
        info['culprit'] = self.culprit_index
        info['task_info'] = self.task_info
        info['previous_turns'] = self.previous_run_turns
        info['agents'] = {i: self.agents[i].culprit_knowledge for i in range(len(self.agents))}
        return info

    def get_agent_names(self):
        names = ""
        for agent in self.agents[:-1]:
            names += agent.name + ', '
        names = names + 'and ' + self.agents[-1].name
        return names

    def setup(self, spread_facts):
        starting_msg = self.get_message('starting').replace("SUSPECT_COUNT", str(self.suspect_count))
        starting_msg = starting_msg.replace("AGENT_COUNT", str(len(self.agents)))
        starting_msg = starting_msg.replace("AGENT_NAMES", self.get_agent_names())
        starting_msg += "\n"
        starting_msg += self.get_message('env_info')
        for agent in self.agents:
            agent.set_system_prompt(starting_msg)

        if spread_facts:
            # Spread facts such that they can guess who the culprit is with all information
            self.agent_to_possible_suspects = []
            maximum_share_suspect_len = -1
            while maximum_share_suspect_len != 1:
                print("Spreading facts")
                all_knowledge = []
                for agent in self.agents:
                    agent.reset()
                    self.assign_knowledge(agent)
                    self.agent_to_possible_suspects.append(self.get_possible_suspects(agent.culprit_knowledge))
                    all_knowledge += agent.culprit_knowledge
                maximum_share_suspect_len = len(self.get_possible_suspects(all_knowledge))

    def get_communication_message(self, agent):
        communication_message = "The current communication channel between you and the other players is:\n"
        if agent.comm_channel:
            communication_message += ''.join(agent.comm_channel)
        else:
            communication_message += 'Currently empty, as this is the first turn.'
        communication_message = f"{communication_message}\n\n" \
        + f"The current turn is {self.game_turn + 1} out of {self.max_turn_count - self.previous_run_turns}.\n" \
        + "It\'s your turn. What would you do?\n"
        return communication_message

    def step(self, agent_answer):
        answer = utils.load_message(agent_answer)
        action = int(answer['action'])
        if action == 1:
            info = int(answer['fact'])
            # Share a fact
            chosen_fact = self.agents[self.cur_player].culprit_knowledge[info - 1]

            remaining_suspects = set()
            for i in self.possible_suspects:
                suspect = self.suspects[i - 1]
                if suspect.is_fact_true(chosen_fact):
                    remaining_suspects.add(i)
            self.possible_suspects = remaining_suspects

            message = self.get_message('share').replace("AGENT_NAME", self.agents[self.cur_player].name)
            message += get_share_message(chosen_fact[0], chosen_fact[1]) + '.\n'
            self.add_to_communication(message)
            return False, action
        
        elif action == 2:
            correct_accusation = (int(answer['character']) - 1) == self.culprit_index
            return True, action
        
        elif action == 3:
            # Do nothing
            return False, action
        
        raise ValueError(f"Invalid action: {action}")

    def load_message_file(self):
        with open(self.message_file, 'r') as f:
            messages = json.load(f)
        
        for k in messages.keys():
            self.messages[k] = '\n'.join(messages[k])
        
        self.messages['env_info'] = self.get_env_description(verbose=False)

    def get_message(self, key):
        return self.messages[key]
    
    def get_env_description(self, verbose=False):
        description = ""
        for i in range(len(self.suspects)):
            description += f"{self.messages['character_name']} number {i + 1}: "
            description += str(self.suspects[i])
            if verbose:
                if i == self.culprit_index:
                    description += " <------ Culprit"
            description += '\n'
        if verbose:
            description += f"{self.messages['character_name']} with (X) can be eliminated with perfect usage of facts.\n"
        return description


 