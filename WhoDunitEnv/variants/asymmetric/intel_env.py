from suspect_creator import SuspectCreator
from suspect import Suspect
from variants.asymmetric.intel_agent import IntelAgent
from consts import *
import numpy as np
import json
import utils


DEFAULT_TASK_INFO = {
    "general": {
        "suspect_count": 6,
        "golden_intel": False,
        'max_turn_count': 31
    },
    "message_file": 'json_files/messages_asymmetric_positive.json'
}


class IntelEnv:
    def __init__(self, agent_pipe, task_info=DEFAULT_TASK_INFO, attribute_file=DEFAULT_ATTRIBUTE_FILE):
        for k in DEFAULT_TASK_INFO.keys():
            if k not in task_info:
                task_info[k] = DEFAULT_TASK_INFO[k]

        if task_info['general'].get('golden_intel', False):
            self.handle_intel = self.handle_intel_golden
        
        self.agent_count = 2
        self.attribute_file = attribute_file
        self.message_file = task_info['message_file']
        self.task_info = task_info

        # Initialize suspects for the environment, and choose culprit.
        self.suspect_creator = SuspectCreator(self.attribute_file)
        self.suspects = self.suspect_creator.create_suspects(self.task_info['general']['suspect_count'])
        self.culprit_index = np.random.randint(self.task_info['general']['suspect_count'])
        self.culprit = self.suspects[self.culprit_index]
        self.suspect_count = len(self.suspects)
        self.possible_suspects = set(range(1, self.suspect_count + 1))

        self.suspect_knowledge = {k: np.zeros(self.suspect_count) for k in self.suspect_creator.attributes}

        # Create Agents..
        self.intel = IntelAgent(agent_pipe, AGENT_NAMES[0], 'intel', len(self.suspects), self.suspect_creator.attributes)
        self.accuser = IntelAgent(agent_pipe, AGENT_NAMES[1], 'accuser', len(self.suspects), self.suspect_creator.attributes)

        self.agents = [self.accuser, self.intel]

        self.messages = dict()
        self.load_message_file()
        self.setup()
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

    def set_from_dict(self, info_dictionary):
        self.reset()
        self.suspect_count = info_dictionary['suspect_count']
        self.suspects = []
        for i in range(self.suspect_count):
            self.suspects.append(Suspect(i, info_dictionary['suspects'][i]))
        self.culprit_index = info_dictionary['culprit']
        self.culprit = self.suspects[self.culprit_index]
        self.load_message_file()
        self.setup()

    def add_to_communication(self, message):
        for agent in self.agents:
            agent.comm_channel.append(message)

    def get_eliminated_suspects(self):
        return set(range(1, self.suspect_count + 1)) - self.possible_suspects

    def to_dict(self):
        info = dict()
        info['suspect_count'] = self.suspect_count
        info['suspects'] = [self.suspects[i].suspect_info for i in range(self.suspect_count)]
        info['culprit'] = self.culprit_index
        info['task_info'] = self.task_info
        info['previous_turns'] = self.previous_run_turns
        return info

    def setup(self):
        chosen_name = self.get_message('chosen_name')
        accuser_msg = self.get_message('accuser')
        character_name = self.get_message('character_name')
        accuser_msg = accuser_msg.replace("SUSPECT_COUNT", str(self.suspect_count)).replace("AGENT_NAME", self.accuser.name).replace("PARTNER_NAME", self.intel.name)
        accuser_msg = accuser_msg + f"\nPossible attributes for {character_name}s are: " + ', '.join(self.intel.possible_properties) + '.\n'
        accuser_msg = accuser_msg + f'\nThe {chosen_name} is described as:\n' + str(self.culprit)
        self.accuser.set_system_prompt(accuser_msg)

        intel_msg = self.get_message('intel').replace("AGENT_NAME", self.intel.name).replace("PARTNER_NAME", self.accuser.name)
        intel_msg = intel_msg + f"Possible attributes for {character_name}s are: " + ', '.join(self.intel.possible_properties) + '.\n'
        intel_msg = intel_msg + f'\nFollowing is information about the {character_name}s:\n' + self.get_message('env_info')
        self.intel.set_system_prompt(intel_msg)

    def get_communication_message(self, agent):
        communication_message = "The current communication channel between you and the other players is:\n"
        if agent.comm_channel:
            communication_message += '\n'.join(agent.comm_channel)
        else:
            communication_message += 'Currently empty, as this is the first turn.'
        communication_message = f"{communication_message}\n\n" \
        + f"The current turn is {self.game_turn + 1} out of {self.max_turn_count - self.previous_run_turns}.\n" \
        + "It\'s your turn. What would you do?\n"
        return communication_message

    def monitor_accuser(self, turn_count):
        agent_answers = {"turn_count": turn_count,
                         "answers": dict()}
        
        temperatures = [0.5, 0]
        keys = ['analysis_accuser_which', 'analysis_accuser_next_action', 'analysis_accuser_best_request', 'analysis_accuser_action_reward']
        prompts = []
        for k in keys:
            key_prompt = self.get_communication_message(self.accuser) + '\n' + self.get_message(k)
            prompts.append(key_prompt)
        
        for temp in temperatures:
            temp_name = 'temp' + str(temp)
            agent_answers['answers'][temp_name] = dict()
            full_prompts, answers, _ = self.accuser.batch_prompt(prompts, validate=False, temperature=temp)
            for i in range(len(keys)):
                agent_answers['answers'][temp_name][keys[i]] = [{'prompt': full_prompts[i], 'answer': answers[i]}]

        return agent_answers
            

    def handle_accuser(self, accuser_answer):
        answer = utils.load_message(accuser_answer)
        action = int(answer['action'])
        request_info = {'action': action}
        impact = -1
        best_impact = max([len(self.possible_suspects) - sum(self.suspect_knowledge[k]) for k in self.suspect_knowledge.keys()])
         
        if action in [1, 2]:  # Request information
            is_done = False
            should_switch = True
            if action == 1:
                impact = 1
                request_info['property'] = answer['property'].replace(' ', '_')
                request_info['value'] = answer['value']
                request_info['requested_id'] = answer['character']
                actual_id = int(request_info['requested_id']) - 1
                if request_info['requested_id'] not in self.possible_suspects:
                     # Blunder: Request info about eliminated suspect, or accuse eliminated suspect
                    self.accuser.blunders.append({'possible_suspects': list(self.possible_suspects), 'culprit_id': self.culprit_index, 'request_info': request_info, 'answer': answer, 'blunder': 'irrelevant character', "turn_count": self.turn_count})
                    impact = 0
                
                if self.suspect_knowledge[request_info['property']][actual_id] == 1:
                    self.accuser.blunders.append({'possible_suspects': list(self.possible_suspects), 'culprit_id': self.culprit_index, 'request_info': request_info, 'answer': answer, 'blunder': 'already known', "turn_count": self.turn_count})
                    impact = 0
                
                if len(self.possible_suspects) == 1:
                    impact = 0

                request = self.get_message('request').replace("AGENT_NAME", self.accuser.name)
                request = request.replace("PROPERTY", str(request_info['property']))
                request = request.replace("ID", str(request_info['requested_id']))
                request = request.replace("VALUE", str(request_info['value']))
            else:  # action == 2: Request broad message
                request = self.get_message('broad_request').replace("AGENT_NAME", self.accuser.name)
                impact = best_impact  # Assuming optimal intel
                if len(self.possible_suspects) == 2:
                    # Correct choice is to request specific information (check in 2)
                    best_impact = 1
                    impact = 0
                if len(self.possible_suspects) == 1:
                    # Correct choice is to accuse, no information can be gathered.
                    impact = 0
            
            self.add_to_communication(request)

        elif action == 3:  # Accuse
            correct_accusation = (answer['character'] - 1) == self.culprit_index
            if correct_accusation:
                self.add_to_communication("Successfully accused: " + str(answer['character']))
            else:
                self.accuser.blunders.append({'possible_suspects': list(self.possible_suspects), 'culprit_id': self.culprit_index, 'request_info': answer['character'], 'answer': answer, 'blunder': 'bad accuse', "turn_count": self.turn_count})

                self.add_to_communication(self.get_message('accuse').replace('ID', str(answer['character'])).replace("AGENT_NAME", self.accuser.name))
                self.possible_suspects = self.possible_suspects - {answer['character']}
            is_done = True
            should_switch = False
        else:
            raise ValueError("Bad action " + str(answer['action']))
        request_info['message'] = self.accuser.comm_channel[-1]
        return is_done, should_switch, action, request_info, impact, best_impact

    def handle_intel_golden(self, intel_answer, request_info):
        answer = utils.load_message(intel_answer)
        action = int(answer['action'])
        value = str(answer['value'])
        impact = -1
        best_impact = -1

        if action == 1:
            if request_info['action'] != 1:
                self.intel.blunders.append({'action': action, 'request_info': request_info, 'answer': intel_answer, 'characters': [], 'property': "Unknown", 'value': "Unknown", 'blunder': "specific with no character", "turn_count": self.turn_count})
                # We want to keep the game going - so we continue to generate the intel message.
                fulfillment_message = self.get_message('fulfill').replace('AGENT_NAME', self.intel.name)
                fulfillment_message += "Agent has been unable to answer the query."
                self.self.add_to_communication(fulfillment_message)
                return action, impact, best_impact # Cannot progress without known property, value or character.

            shared_property = str(request_info['property'])
            shared_value = str(request_info['value'])
            requested_id = request_info['requested_id']
            # Since this is golden, we swap out the value for the actual value for the requested character:
            boolean_answer = self.suspects[int(requested_id) - 1].is_fact_true((shared_property, shared_value))
            character_set = {int(requested_id)}

            impact = 1 - self.suspect_knowledge[shared_property][int(requested_id) - 1]
            self.suspect_knowledge[shared_property][int(requested_id) - 1] = 1
                
            fulfillment_message = self.get_message('fulfill').replace('AGENT_NAME', self.intel.name)
            fulfillment_message += get_response_message(shared_property, shared_value, requested_id, not boolean_answer)
            self.add_to_communication(fulfillment_message)

            if int(requested_id) in self.possible_suspects:
                if self.culprit.is_fact_true((shared_property, shared_value)) != boolean_answer:
                    self.possible_suspects -= character_set
                
        elif action == 2:
            value = value.split('-')
            shared_property, shared_value = value[0], value[1]
            boolean_answer = True

            # Since this is golden answer, we swap the character set for the exact character set that fits the property + value combo
            character_set = set()
            characters = []
            for character in range(len(self.suspects)):
                if self.suspects[character].is_fact_true((shared_property, shared_value)):
                    character_set.add(character + 1)
                    characters.append(character + 1)
                
            broadcast_message = self.get_message('broadcast').replace("AGENT_NAME", self.intel.name)
            broadcast_message += f"For characters: {characters} the property {shared_property} is {shared_value}"
            self.add_to_communication(broadcast_message)

            impact = self.suspect_count - sum(self.suspect_knowledge[shared_property])
            best_impact = max([self.suspect_count - sum(self.suspect_knowledge[k]) for k in self.suspect_knowledge.keys()])
            self.suspect_knowledge[shared_property][:] = 1

            does_culprit_fit_answer = self.culprit.is_fact_true((shared_property, shared_value))
            new_possible_suspects = set()
            intel_correct_answer = True
            for suspect_id in self.possible_suspects:
                # Suspect and culprit agree about property value.
                if self.suspects[suspect_id - 1].is_fact_true((shared_property, shared_value)) == does_culprit_fit_answer:
                    new_possible_suspects.add(suspect_id)
                    if not ((suspect_id in character_set) == does_culprit_fit_answer):
                        # Means the intel answer was wrong!
                        intel_correct_answer = False
            self.possible_suspects = new_possible_suspects
            if not intel_correct_answer:
                self.intel.blunders.append({'action': action, 'accuser_action': request_info['action'], 'message': request_info['message'], 'answer': intel_answer, 'characters': list(character_set), 'property': shared_property, 'value': shared_value, 'blunder': "incorrect output action 2", "turn_count": self.turn_count})

            if request_info['action'] != 2:
                # Blunder: Return broad message when asked for specific information
                self.intel.blunders.append({'action': action, 'accuser_action': request_info['action'], 'message': request_info['message'], 'answer': intel_answer, 'characters': list(character_set), 'property': shared_property, 'value': shared_value, 'blunder': "action mismatch 2", "turn_count": self.turn_count})

        else:
            raise ValueError("Bad action " + str(answer['action']))

        valid_answer = np.array([self.suspects[index - 1].is_fact_true((shared_property, shared_value)) == boolean_answer for index in character_set]).all()
        if not valid_answer:
            self.intel.blunders.append({'characters': list(character_set), 'property': shared_property, 'value': value, 'action': action, "accuser_action": request_info['action'], "message": request_info['message'], 'blunder': "Wrong info", "turn_count": self.turn_count})

        return action, impact, best_impact

    def handle_intel(self, intel_answer, request_info):
        answer = utils.load_message(intel_answer)
        action = int(answer['action'])
        # shared_property = answer['property'].replace(' ', '_')
        value = str(answer['value'])
        impact = -1
        best_impact = -1

        if action == 1:
            if request_info['action'] != 1:
                # Blunder: Return requested information when asked for broad message
                self.intel.blunders.append({'action': action, 'request_info': request_info, 'answer': intel_answer, 'characters': [], 'property': "Unknown", 'value': "Unknown", 'blunder': "specific with no character", "turn_count": self.turn_count})
                # We want to keep the game going - so we continue to generate the intel message.
                fulfillment_message = self.get_message('fulfill').replace('AGENT_NAME', self.intel.name)
                # fulfillment_message += get_response_message("Unknown", "Unknown", "Unknown", not utils.try_bool(value))
                fulfillment_message += "Agent has been unable to answer the query."
                self.add_to_communication(fulfillment_message)
                return action, impact, best_impact # Cannot progress without known property, value or character.

            shared_property = str(request_info['property'])
            shared_value = str(request_info['value'])
            requested_id = request_info['requested_id']
            boolean_answer = utils.try_bool(value)
            character_set = {int(requested_id)}

            impact = 1 - self.suspect_knowledge[shared_property][int(requested_id) - 1]
            self.suspect_knowledge[shared_property][int(requested_id) - 1] = 1
                
            fulfillment_message = self.get_message('fulfill').replace('AGENT_NAME', self.intel.name)
            fulfillment_message += get_response_message(shared_property, shared_value, requested_id, not boolean_answer)
            self.add_to_communication(fulfillment_message)

            if int(requested_id) in self.possible_suspects:
                if self.culprit.is_fact_true((shared_property, shared_value)) != boolean_answer:
                    self.possible_suspects -= character_set
                
        elif action == 2:
            characters = answer['character']
            boolean_answer = True
            try:
                eval_characters = eval(characters)
                if isinstance(eval_characters, int):
                    character_set = {eval_characters}
                else:
                    character_set = set(eval_characters)
            except Exception as e:
                print(characters)
                print(e)
                print(eval_characters)
                raise e
            bad_characters = character_set - set(range(1, self.suspect_count + 1))
            if len(bad_characters) > 0:
                # Intel returned a list with invalid characters
                self.intel.blunders.append(
                    {'action': action, 'accuser_action': request_info['action'], 'message': request_info['message'], 'answer': intel_answer, 'characters': list(character_set), 'property': shared_property, 'value': shared_value, 'blunder': "bad character set", "turn_count": self.turn_count})
                # Clean the set
                character_set = character_set - bad_characters
                
            
            value = value.split('-')
            shared_property, shared_value = value[0], value[1]
            broadcast_message = self.get_message('broadcast').replace("AGENT_NAME", self.intel.name)
            broadcast_message += "For characters: " + characters + ", the property " + shared_property + " is " + shared_value
            self.add_to_communication(broadcast_message)

            impact = self.suspect_count - sum(self.suspect_knowledge[shared_property])
            best_impact = max([self.suspect_count - sum(self.suspect_knowledge[k]) for k in self.suspect_knowledge.keys()])
            self.suspect_knowledge[shared_property][:] = 1

            culprit_answer = self.culprit.is_fact_true((shared_property, shared_value))
            new_possible_suspects = set()
            intel_correct_answer = True
            for suspect_id in self.possible_suspects:
                # Suspect and culprit agree about property value.
                if self.suspects[suspect_id - 1].is_fact_true((shared_property, shared_value)) == culprit_answer:
                    new_possible_suspects.add(suspect_id)
                    if not ((suspect_id in character_set) == culprit_answer):
                        # Means the intel answer was wrong!
                        intel_correct_answer = False
            self.possible_suspects = new_possible_suspects
            if not intel_correct_answer:
                self.intel.blunders.append({'action': action, 'accuser_action': request_info['action'], 'message': request_info['message'], 'answer': intel_answer, 'characters': list(character_set), 'property': shared_property, 'value': shared_value, 'blunder': "incorrect output action 2", "turn_count": self.turn_count})

            if request_info['action'] != 2:
                # Blunder: Return broad message when asked for specific information
                self.intel.blunders.append({'action': action, 'accuser_action': request_info['action'], 'message': request_info['message'], 'answer': intel_answer, 'characters': list(character_set), 'property': shared_property, 'value': shared_value, 'blunder': "action mismatch 2", "turn_count": self.turn_count})

        else:
            raise ValueError("Bad action " + str(answer['action']))

        valid_answer = np.array([self.suspects[index - 1].is_fact_true((shared_property, shared_value)) == boolean_answer for index in character_set]).all()
        if not valid_answer:
            self.intel.blunders.append({'characters': list(character_set), 'property': shared_property, 'value': value, 'action': action, "accuser_action": request_info['action'], "message": request_info['message'], 'blunder': "Wrong info", "turn_count": self.turn_count})

        return action, impact, best_impact


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

    def get_agent_names(self):
        names = ""
        for agent in self.agents[:-1]:
            names += agent.name + ', '
        names = names + 'and ' + self.agents[-1].name
        return names

    def set_summary(self, summary):
        summary = f"Parts of the communication channel were summarized by a helpful agent as:\n{summary}"
        for agent in self.agents:
            agent.comm_channel = []
        self.add_to_communication(summary.split('\n'))
    


 