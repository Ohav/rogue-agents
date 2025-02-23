import numpy as np
from intervention.exceptions import ResetGameException
from intervention.intervention_models import FullyConnected
from intervention.intervention_manager import InterventionManager
import pickle

MINIMUM_RESET_TURN = 0

class InterventionManagerSymmetric(InterventionManager):
    def act(self, agent):
        # Uses agent memory and env to generate current prompt
        prompt = '\n'.join([agent.get_knowledge_description(),  self.env.get_communication_message(agent)])
        ret, res_name = self.decide_on_intervention(agent, prompt)
        return ret, res_name 


    def decide_on_intervention(self, agent, message):
        res_name = 'None'
        response = None

        if self.env.reset_count > 0:
            agent.pipe.set_save_logprobs(True)
            response = agent.prompt(message)
            entropy, varentropy, kurtosis = agent.calc_agent_entropy()

            if entropy > 0: # If entropy is less than 0, there's a problem while calculating.
                classifier_score = self.entropy_classifier_function([self.env.game_turn, entropy, varentropy, kurtosis])
                if classifier_score > self.entropy_reset_threshold:
                    # If we got here, we can still reset
                    if self.intervention_action == 'reset':
                        print(f"Reset with score {classifier_score}")
                        raise ResetGameException(f'Reset with score {classifier_score}. Response: {response}')
                    elif self.intervention_action == 'retry':
                        # Just retry the same prompt
                        response = agent.prompt(message)
                        self.env.reset_count -= 1
                        self.env.previous_run_turns += 1
                    else:
                        raise Exception()
                print(f"Turn {self.env.game_turn}. Didn't reset: {classifier_score} <= {self.entropy_reset_threshold}")
            else:
                print(f"Invalid entropy for agent {self.env.cur_player}, turn: {self.env.game_turn}")
        else: 
            print(f"Out of resets: {self.env.reset_count}")
        
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
            entropy = np.log(entropy)
            varentropy = np.log(varentropy)
            features = torch.Tensor([turn, entropy, varentropy, kurtosis]).reshape(1, -1)
            return classifier(features)
        return filter_by_model