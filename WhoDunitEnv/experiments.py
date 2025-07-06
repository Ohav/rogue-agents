from consts import *
import os
import argparse
import logging
logger = logging.getLogger(__name__)


def run_asymmetric(name, model, experiment_output, **kwargs):
    from agent_pipes import TransformersPipe
    from game_memory import IntelGameMemory
    from variants.asymmetric.intel_game import game_loop
    from variants.asymmetric.intel_env import IntelEnv

    if model == 'random':
        raise ValueError("Unsupported model: random")
        # llm_pipe = RandomBaseline(suspect_count.value, 3)
    else:
        llm_pipe = TransformersPipe(model, **kwargs)

    base_path = experiment_output
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    print("Output path: " + experiment_output)
    random_envs = kwargs.get('game_count', 50)
    
    message_file = 'json_files/messages_asymmetric_positive.json' if kwargs.get('positive', False) else 'json_files/messages_asymmetric.json'
    task_info = {
        "general": {
            "suspect_count": kwargs.get('suspect_count', 6)
        },
        'message_file': message_file
    }
    monitor = 'gpt' not in model

    for i in range(random_envs):
        print(f"Starting env {i}")
        env = IntelEnv(llm_pipe, attribute_file=DEFAULT_ATTRIBUTE_FILE, task_info=task_info)

        path = base_path + f'env_{i}.json'
        memory = IntelGameMemory(path, env)

        for v in game_loop(env, memory, monitor=monitor, max_turns=kwargs['game_length']):
            continue
        memory.dump_memory()
        env.reset()


DEFAULT_AGENT_COUNT = 6
KNOWLEDGE_SIZE = 3
def define_games_symmetric(name, output, **kwargs):
    from variants.symmetric.symm_env import SymmEnv
    from game_memory import SymmGameMemory
    message_file = 'json_files/messages_facts_positive.json'
    task_info = {
        "general": {
            "suspect_count": kwargs.get('suspect_count', 6),
            'max_turn_count': kwargs['game_length'],
            'agent_count': kwargs.get('agent_count', DEFAULT_AGENT_COUNT),
            'knowledge_size': kwargs.get('knowledge_size', KNOWLEDGE_SIZE)
        },
        'message_file': message_file
    }
    
    index = kwargs.get('index', 0)
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)
    
    game_count = kwargs.get('game_count', 50)

    print(f"Defining {game_count} envs for future use")
    for i in range(game_count):
        env = SymmEnv(None, attribute_file='json_files/attributes_long.json', task_info=task_info)
        env.setup(True)
        path = output + f'{name}_env_{index * game_count + i}.json'
        memory = SymmGameMemory(path, env)
        memory.dump_memory()
        env.reset()

def load_and_run_symmetric(input_json, name, model, experiment_output, **kwargs):
    from variants.symmetric.symm_game import run_from_file

    message_file = 'json_files/messages_facts_positive.json'
    task_info = {
        "general": {
            "suspect_count": kwargs.get('suspect_count', 6),
            'max_turn_count': kwargs['game_length'],
            'agent_count': kwargs.get('agent_count', DEFAULT_AGENT_COUNT),
            'knowledge_size': kwargs.get('knowledge_size', KNOWLEDGE_SIZE)
        },
        'message_file': message_file
    }
    if not os.path.exists(experiment_output):
        os.makedirs(experiment_output, exist_ok=True)

    output_path = experiment_output + f'/{name}.json'
    
    complete = kwargs.get('complete', False)
    print(output_path)
    if complete and os.path.exists(output_path):
        print("skip")
        # If flag is complete, don't overwrite existing games.
        return

    run_from_file(input_json, model, int(kwargs['load_turn_count']), task_info, output_path, **kwargs)

def load_and_run_asymmetric(input_json, name, model, experiment_output, **kwargs):
    from variants.asymmetric.intel_game import run_from_file

    message_file = 'json_files/messages_asymmetric_positive.json' if kwargs.get('positive', True) else 'json_files/messages_asymmetric.json'
    task_info = {
        "general": {
            "suspect_count": kwargs.get('suspect_count', 6),
            'golden_intel': kwargs.get('golden_intel', False),
            'max_turn_count': kwargs['game_length']
        },
        'message_file': message_file
    }
    if not os.path.exists(experiment_output):
        os.makedirs(experiment_output, exist_ok=True)

    output_path = experiment_output + f'/{name}.json'
    
    complete = kwargs.get('complete', False)
    complete = True
    print(output_path)
    if complete and os.path.exists(output_path):
        print("skip")
        # If flag is complete, don't overwrite existing games.
        return

    run_from_file(input_json, model, int(kwargs['load_turn_count']), task_info, output_path, **kwargs)
    
def define_games(name, output, **kwargs):
    from variants.asymmetric.intel_env import IntelEnv
    from game_memory import IntelGameMemory
    message_file = 'json_files/messages_asymmetric_positive.json' if kwargs.get('positive', True) else 'json_files/messages_asymmetric.json'
    task_info = {
        "general": {
            "suspect_count": kwargs.get('suspect_count', 6),
            'golden_intel': kwargs.get('golden_intel', False),
            'max_turn_count': 0
        },
        'message_file': message_file
    }
    
    index = kwargs.get('index', 0)
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)
    
    game_count = kwargs.get('game_count', 50)

    print(f"Defining {game_count} envs for future use")
    for i in range(game_count):
        env = IntelEnv(None, attribute_file=DEFAULT_ATTRIBUTE_FILE, task_info=task_info)
        path = output + f'{name}_env_{index * game_count + i}.json'
        memory = IntelGameMemory(path, env)
        memory.dump_memory()
        env.reset()


