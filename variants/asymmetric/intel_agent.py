import numpy as np
from consts import MAX_FORMAT_ATTEMPTS, DEFAULT_TEMPERATURE
import utils
import itertools

from variants.asymmetric.errors import Errors, error_to_text, BadAnswerException
DEBUG = False

class IntelAgent:
    def __init__(self, llm_pipe, name, role, suspect_count, possible_properties, intervention_manager=None):
        self.pipe = llm_pipe
        self.system_prompt = {"role": "system",
                              "content": ""
                              }
        
        self.name = name
        self.role = role

        self.suspect_count = suspect_count
        self.possible_properties = sorted(list(possible_properties))
        self.comm_channel = None
        self.errors = None
        self.blunders = None
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
        relevant_features = {'entropy': [], 'varentropy': [], 'kurtosis': []}
        collected = {'character': [], 'action': []}
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
                        collected[text].append([p[1] for p in logprobs[index]])

        if len(collected['character']) == 0 or (len(collected['action']) == 0):
            if len(collected['character']) == 0 and (len(collected['action']) == 0):
                relevant_features['entropy'] = [-2]
                relevant_features['varentropy'] = [-2]
                relevant_features['kurtosis'] = [0]

            else:
                for k in relevant_features.keys():
                    relevant_features[k] = [utils.get_score(np.array(logs), k) for logs in (collected['action'] + collected['character'])] 
            
            return relevant_features

        for i,j in itertools.product(range(len(collected['character'])), range(len(collected['action']))):
            char_logprobs = collected['character'][i]
            act_logprobs = collected['action'][j]
            combined = np.array([p1 + p2 for p1 in char_logprobs for p2 in act_logprobs])
            for k in relevant_features.keys():
                relevant_features[k].append(utils.get_score(combined, k))
        return relevant_features

    def max_agent_entropy(self):
        feature_dictionary = self.features_for_char_times_action()
        self.last_answer_entropy = max(feature_dictionary['entropy'])
        self.last_answer_varentropy = max(feature_dictionary['varentropy'])
        self.last_answer_kurtosis = max(feature_dictionary['kurtosis'])
        
    def act(self, env):
        if self.intervention_manager is None:
            prompt = env.get_communication_message()
            return self.prompt(prompt), "None"
        return self.intervention_manager.act(self)

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

        # Accuser segment
        if self.role == 'accuser':
            if action not in [1, 2, 3]:
                return Errors.BAD_ACTION
        
            if action in [1, 3]:
                # Request(1) and accuse(3) require a character
                if 'character' not in answer_dict:
                    return Errors.MISSING_CHARACTER
                character = utils.try_int(answer_dict['character'])
                if not (0 < character <= self.suspect_count):
                    return Errors.BAD_CHARACTER
                    
                if action == 1:
                    if 'property' not in answer_dict:
                        return Errors.MISSING_PROPERTY
                    if answer_dict['property'] not in self.possible_properties:
                        return Errors.BAD_PROPERTY
                    if 'value' not in answer_dict:
                        return Errors.MISSING_VALUE
                    # TODO: Check value?
            # else action == 2 -> no checks needed
                
        # Intel segment
        elif self.role == 'intel':
            # If the action is not the correct one asked for, that is a "mistake" and not a format error
            if action not in [1, 2]:
                return Errors.BAD_ACTION
            if 'value' not in answer_dict:
                return Errors.MISSING_VALUE

            if action == 1:
                try:
                    value = utils.try_bool(answer_dict['value'])
                except ValueError as e:
                    return Errors.BAD_VALUE_1
            
            if action == 2:  # Broad message
                value = answer_dict['value']
                if '-' not in value:
                    return Errors.BAD_VALUE_2
                value = value.split('-')
                prop, val = value[0], value[1]
                if prop not in self.possible_properties:
                    return Errors.BAD_PROPERTY
                # TODO: Check value?

                if 'character' not in answer_dict:
                    return Errors.MISSING_CHARACTER_LIST
                try:
                    characters = eval(answer_dict['character'])
                except Exception as e:
                    return Errors.BAD_CHARACTER_LIST
            
        return Errors.NO_ERROR

    
    def set_system_prompt(self, system_prompt):
        self.system_prompt['content'] = system_prompt