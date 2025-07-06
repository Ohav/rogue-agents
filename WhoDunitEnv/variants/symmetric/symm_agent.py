import numpy as np
from consts import get_share_message, MAX_FORMAT_ATTEMPTS, DEFAULT_TEMPERATURE
import utils
import itertools

from variants.symmetric.errors import Errors, error_to_text, BadAnswerException
DEBUG = False

class SymmAgent:
    def __init__(self, llm_pipe, name, role, suspect_count, possible_properties, intervention_manager=None):
        self.pipe = llm_pipe
        self.system_prompt = {"role": "system",
                              "content": ""
                              }
        
        self.name = name
        self.role = 'detective'

        self.culprit_knowledge = []

        self.suspect_count = suspect_count
        self.possible_properties = sorted(list(possible_properties))
        self.comm_channel = None
        self.errors = None
        self.blunders = None
        self.last_answer_monitor_token = {k: None for k in ['entropy', 'varentropy', 'kurtosis']}
        self.last_answer_entropy = -2
        self.last_answer_varentropy = -2
        self.last_answer_kurtosis = 0
        
        # Defines logic for agent actions, other than LLM.
        self.intervention_manager = intervention_manager

        self.reset()

    def reset(self):
        self.comm_channel = []
        self.errors = dict()
        self.blunders = []
        self.last_answer_entropy = -2
        self.last_answer_varentropy = -2
        self.last_answer_kurtosis = 0

    def calc_agent_entropy(self):
        entropy = -1
        logprobs = self.pipe.get_last_logprobs()
        if logprobs is None:
            self.last_answer_entropy = -2
            self.last_answer_varentropy = -2
            self.last_answer_kurtosis = 0
        else:
            self.max_agent_entropy()
        
        return self.last_answer_entropy, self.last_answer_varentropy, self.last_answer_kurtosis

    def features_for_char_times_action(self, limit=5):
        relevant_features = {'entropy': {'scores': [], 'tokens': []},
                             'varentropy': {'scores': [], 'tokens': []},
                             'kurtosis': {'scores': [], 'tokens': []}}
        collected = {'character': [], 'action': []} #, 'fact': []}
        logprobs = self.pipe.get_last_logprobs()
        # Logprobs is a list. Each element is a generation, i.e. a list of (token, prob) pairings for that specific location
        for i in range(len(logprobs)):
            for text in collected.keys():
                if text in logprobs[i][0][0].lower(): # Get first token
                    index = i + 1
                    found = utils.try_int(logprobs[index][0][0].strip()) >= 0
                    while not found and index < i + limit:
                        index = index + 1
                        if index >= len(logprobs):
                            break
                        found = utils.try_int(logprobs[index][0][0].strip()) >= 0
                    if found:
                        tokens = [p[0] for p in logprobs[index]]
                        index_logprobs = [p[1] for p in logprobs[index]]
                        collected[text].append({'tokens': tokens, 'logprobs': index_logprobs})

        collected['info'] = collected['character']
        del collected['character']
        # del collected['fact']

        if len(collected['info']) == 0 or (len(collected['action']) == 0):
            if len(collected['info']) == 0 and (len(collected['action']) == 0):
                relevant_features['entropy']['scores'] = [-2]
                relevant_features['varentropy']['scores'] = [-2]
                relevant_features['kurtosis']['scores'] = [0]
                for k in relevant_features.keys():
                    relevant_features[k]['tokens'] = ["None"] 

            else:
                for k in relevant_features.keys():
                    relevant_features[k]['scores'] = [utils.get_score(np.array(logs['logprobs']), k) for logs in (collected['action'] + collected['info'])]
                    relevant_features[k]['tokens'] = [logs['tokens'] for logs in (collected['action'] + collected['info'])]
            
            return relevant_features

        for i,j in itertools.product(range(len(collected['info'])), range(len(collected['action']))):
            char_location = collected['info'][i]
            act_location = collected['action'][j]
            combined = np.array([p1 + p2 for p1 in char_location['logprobs'] for p2 in act_location['logprobs']])
            for k in relevant_features.keys():
                relevant_features[k]['scores'].append(utils.get_score(combined, k))
                relevant_features[k]['tokens'].append(char_location['tokens'] + act_location['tokens'])
        return relevant_features

    def max_agent_entropy(self):
        feature_dictionary = self.features_for_char_times_action()
        self.last_answer_entropy = max(feature_dictionary['entropy']['scores'])
        self.last_answer_varentropy = max(feature_dictionary['varentropy']['scores'])
        self.last_answer_kurtosis = max(feature_dictionary['kurtosis']['scores'])
        for k in feature_dictionary.keys():
            self.last_answer_monitor_token[k] = feature_dictionary[k]['tokens'][np.argmax(feature_dictionary[k]['scores'])]
        
    def get_knowledge_description(self):
        description = f"You received the following starting facts about the Winner (these are the facts you may share):\n"
        for i in range(len(self.culprit_knowledge)):
            fact = self.culprit_knowledge[i]
            description += f"{i + 1}. {get_share_message(fact[0], fact[1])}.\n"
        return description

    def act(self, env):
        if self.intervention_manager is None:
            prompt = '\n'.join([self.get_knowledge_description(),  env.get_communication_message(self)])
            return self.prompt(prompt), "None"
        return self.intervention_manager.act(self)

    def add_knowledge(self, facts):
        for fact in facts:
            self.culprit_knowledge.append(fact)

    def prompt(self, message, validate=True, temperature=DEFAULT_TEMPERATURE):
        prompts, answers, validations = self.batch_prompt([message], validate, temperature=temperature)
        return prompts[0], answers[0], validations[0]
    
    def batch_prompt(self, messages, validate=True, temperature=DEFAULT_TEMPERATURE):
        prompts = []
        for message in messages:
            prompt = [self.system_prompt,
                     {'role': 'user', 'content': message}]
            prompts.append(prompt)

        answers = []
        validations = []
        batch_size = 4 if self.pipe.can_batch() else 1

        i = 0
        while i < len(prompts):
            batch_answers = self.pipe.run_batch(prompts[i:i+batch_size], temperature=temperature)
            if i + batch_size > len(prompts):
                # Left with less than batch size prompts
                batch_size = len(prompts) - i
            if not validate:
                answers += batch_answers
                validations += [True] * batch_size
            else:
                for j in range(min(batch_size, len(prompts) - i)):
                    validated, answer = self.test_answer(batch_answers[j], prompts[i + j], temperature)
                    answers.append(answer)
                    validations.append(validated)
            i += batch_size
        return prompts, answers, validations
    
    def test_answer(self, answer, prompt, temperature):
        value = utils.load_message(answer)
        if value is not None:
            validate_res = self.validate_answer(value)
        else:
            validate_res = Errors.INVALID_FORMAT
        validated = (validate_res == Errors.NO_ERROR)

        attempt_count = 0
        while (not validated) and (attempt_count < MAX_FORMAT_ATTEMPTS):
            if DEBUG:
                print(f"Validation failed. Attempt count {attempt_count} with error {error_to_text.get(validate_res)}")   
            # Update error counter
            self.errors[validate_res] = self.errors.get(validate_res, 0) + 1

            # Attach error helper to prompt
            new_prompt = [prompt[0], {'role': 'user', 'content': prompt[1]['content']}]
            notes = error_to_text.get(validate_res, "Missing error text")
            new_prompt[1]['content'] += '\n' + notes
            answer = self.pipe(new_prompt, temperature=temperature)
            attempt_count += 1

            value = utils.load_message(answer)
            if value is not None:
                validate_res = self.validate_answer(value)
            else:
                validate_res = Errors.INVALID_FORMAT
            validated = (validate_res == Errors.NO_ERROR)

        if not validated:
            err_str = f"Prompt {new_prompt}\nFailed after {attempt_count} attempts on temperature {temperature} with final answer:\n{answer}\n\nAgent is {self.role}, {self.name}"
            raise BadAnswerException(err_str)
        return validated, answer

    def validate_answer(self, answer_dict):
        if 'thoughts' not in answer_dict:
            return Errors.MISSING_THOUGHTS
        if 'action' not in answer_dict:
            return Errors.MISSING_ACTION
        action = utils.try_int(answer_dict['action'])

        if action not in [1, 2, 3]: # Share, Accuse or Skip
            return Errors.BAD_ACTION
    
        if action == 1:
            # Request(1) requires a fact number
            if 'fact' not in answer_dict:
                return Errors.MISSING_FACT
            fact_index = utils.try_int(answer_dict['fact'])
            if not (0 < fact_index <= len(self.culprit_knowledge)):
                return Errors.BAD_FACT
                
        if action == 2:
            if 'character' not in answer_dict:
                return Errors.MISSING_CHARACTER
            accused_suspect = utils.try_int(answer_dict['character'])
            if not (0 < accused_suspect <= self.suspect_count):
                return Errors.BAD_CHARACTER
            
        return Errors.NO_ERROR

    def set_system_prompt(self, system_prompt):
        self.system_prompt['content'] = system_prompt.replace("AGENT_NAME", self.name)