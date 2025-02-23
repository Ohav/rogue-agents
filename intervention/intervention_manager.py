import numpy as np
from intervention.exceptions import ResetGameException
from intervention.intervention_models import FullyConnected
import pickle

DEFAULT_INTERVENTION_ACTION = 'reset'

class InterventionManager:
    def __init__(self, env, classifier=None, entropy_threshold=0, reset_game_count=0, intervention_action=None, verbose=False):
        self.reset_game_count = reset_game_count
        
        self.entropy_reset_threshold = entropy_threshold
        self.entropy_classifier_function = self.get_entropy_classifier(classifier)

        if intervention_action is None:
            self.intervention_action = DEFAULT_INTERVENTION_ACTION
        else:
            self.intervention_action = intervention_action

        self.env = env

        self.verbose = verbose

        if self.verbose:
            print(f"""Intervention set: classifier={classifier}, entropy_threshold={entropy_threshold}, reset_game_count={reset_game_count}""")


    def act(self, agent):
        # Uses agent memory and env to generate current prompt
        prompt = self.env.get_communication_message(agent)
        ret, res_name = self.decide_on_intervention(agent, prompt)
        return ret, res_name 


    def decide_on_intervention(self, agent, message):
        res_name = 'None'
        response = None

        if self.reset_game_count > 0:
            agent.pipe.set_save_logprobs(True)
            response = agent.prompt(message)
            entropy, varentropy, kurtosis = agent.calc_agent_entropy()

            if entropy > 0: # If entropy is less than 0, there's a problem while calculating.
                classifier_score = self.entropy_classifier_function([self.env.game_turn, entropy, varentropy, kurtosis])
                if classifier_score > self.entropy_reset_threshold:
                    # If we got here, we can still reset
                    self.reset_game_count -= 1
                    if self.intervention_action == 'reset':
                        raise ResetGameException(f'Reset with score {classifier_score}. Response: {response}')
                    elif self.intervention_action == 'retry':
                        # Communication analysis - just prompt the message again
                        res_name = 'retry'
                        self.env.previous_run_turns += 1
                        response = agent.prompt(message)
                    else:
                        raise Exception(f"Invalid intervention type {self.intervetion_action}")
                    
                if self.verbose:
                    print(f"Turn {self.env.game_turn}. Didn't reset: {classifier_score} <= {self.entropy_reset_threshold}")
            elif self.verbose:
                print(f"Invalid entropy for agent {self.env.cur_player}, turn: {self.env.game_turn}")
        
        if response is None:
            # No intervention
            response = agent.prompt(message)
        
        return response, res_name


    def get_entropy_classifier(self, model_path):
        if (model_path is None) or (model_path == 'None'):
            # Random
            print("loading random")
            def random_model(features):
                return np.random.random()
            return random_model

        import torch

        if model_path.endswith('.pkl'):
            # Is an sklearn polynomial fit
            with open(model_path, 'rb') as f:
                classifier = pickle.load(f)
        else:
            # Is a pytorch model
            classifier = FullyConnected(2)
            classifier.load_state_dict(torch.load(model_path, weights_only=True))
        
        def filter_by_model(features):
            turn, entropy, varentropy, kurtosis = features
            entropy = np.log(max(entropy, 0.0000001))
            varentropy = np.log(max(varentropy, 0.0000001))
            features = torch.Tensor([turn, entropy, varentropy, kurtosis]).reshape(1, -1)
            return classifier(features)
        return filter_by_model