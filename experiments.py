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


# def launch_slurm(name, count, run_type, slurm_output, model, experiment_output, gpu_count=0):
#     slurm_template = "#! /bin/sh\n " \
#                     "#SBATCH --job-name={job_name}_{i} \n" \
#                     "#SBATCH --output={slurm_output}/{job_name}_{i}.out\n" \
#                     "#SBATCH --error={slurm_output}/{job_name}_{i}.err\n" \
#                     "#SBATCH --account=gpu-research\n" \
#                     "#SBATCH --partition=gpu-h100-killable\n" \
#                     "#SBATCH --time=1440\n" \
#                     "#SBATCH --nodes=1\n"
#     if gpu_count != 0:
#         slurm_template += f"#SBATCH --gpus={gpu_count}\n"

#     slurm_template += "python /home/morg/students/ohavbarbi/multiAgent/experiments.py --name {job_name}_{i} --type {type} --model {model} --output {experiment_output}/{job_name}_{i}/ --gpu_count {gpu_count}\n"
    
    
#     if not os.path.exists(slurm_output):
#         os.makedirs(slurm_output)
#     for i in range(1, count + 1):
#         cur_template = slurm_template.format(job_name=name, i=i, slurm_output=slurm_output, type=run_type, model=model, experiment_output=experiment_output, gpu_count=gpu_count)
#         file_name = f"{slurm_output}/{name}_{i}.slurm"
#         with open(file_name, 'w') as f:
#             f.write(cur_template)
#         os.chmod(file_name, 0o777)
#         print(f"Launching: {name}_{i}")
#         os.system(f"sbatch {file_name}")



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--name", type=str, default="")
#     parser.add_argument("--type", type=int, choices=EXP_NAMES.keys(), default=1)
#     parser.add_argument('--count', type=int, default=0)
#     parser.add_argument("--model", type=str)
#     parser.add_argument('--gpu_count', type=int, default=0)
#     parser.add_argument("--output", default=None)
#     parser.add_argument('--suspect_count', type=int, default=6)
#     parser.add_argument('--positive', action='store_true')
#     parser.add_argument('--json_file', type=str, default="")
#     parser.add_argument('--load_turn_count', type=int, default=0)
#     parser.add_argument('--game_length', type=int, default=MAX_TURN_COUNT)
#     args = parser.parse_args()

#     experiment_output = args.output
#     if experiment_output is None:
#         experiment_output = f'run_results/{args.name}/output/'
#     elif args.count > 0:
#         experiment_output = experiment_output + '/{args.name}'
    
#     slurm_output = f"run_results/{args.name}/slurm_meta/"
#     if args.count > 0:
#         print("Launching on slurm")
#         launch_slurm(args.name, args.count, args.type, slurm_output, args.model, experiment_output, args.gpu_count)
#         print("Done! Use: | squeue --state all --me | to inspect")
#         exit()


#     print(f"Running experiment: {experiment_output} ({args.type})")
#     if experiment_output.endswith('/'):
#         url_path = '/'.join(experiment_output.split('/')[:-2]) + '/server/url_path.txt'
#     else:
#         url_path = '/'.join(experiment_output.split('/')[:-1]) + '/server/url_path.txt'
#     print(f"Loading url from: {url_path}")
#     if os.path.exists(url_path):
#         print("path exists")
#         with open(url_path, 'r') as f:
#             url = f.readlines()[0].strip()
#         print(f"Loaded url: {url}")
#     else:
#         print("invalid path")
#         url = None
#     other_values = {'suspect_count': args.suspect_count, 'url': url, 'positive': args.positive, 'gpu_count': args.gpu_count, 
#                     'json_file': args.json_file, 'load_turn_count': args.load_turn_count, 'game_length': args.game_length}
#     experiment = EXP_FUNCS[args.type]
#     experiment(args.name, args.model, experiment_output, **other_values)
#     print("Done")
