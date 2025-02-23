import sys
sys.path.append('/home/morg/students/ohavbarbi/multiAgent/')
import argparse, glob, os, json
from experiments import run_asymmetric, load_and_run_asymmetric, define_games, load_and_run_symmetric, define_games_symmetric
import itertools
import shutil


def copy_classifier(classifier_path, destination_dir, name):
    if (classifier_path is None) or (classifier_path == "None"):
        return None
    os.makedirs(destination_dir, exist_ok=True)
    file_name = os.path.basename(classifier_path)
    destination_file = destination_dir + '/' + name + '_' + file_name
    shutil.copy(classifier_path, destination_file)
    return destination_file

def launch_on_slurm(name, original_output, job_count, gpu_count=1, **kwargs):
    output = original_output

    slurm_output = output + name + '/slurm/'
    if not os.path.exists(slurm_output):
        os.makedirs(slurm_output)

    if job_count is None or job_count == 0:
        launch_job(name, None, output, gpu_count, **kwargs)
    else:
        launch_job_tasks(name, output, job_count, gpu_count)
        # for i in range(job_count):
            # launch_job(name, i, output, gpu_count, **kwargs)

def launch_job_tasks(name, output, job_count, gpu_count):
    job_name = name
    slurm_output = output + name + '/slurm/'
    partition = 'gpu-h100-killable' if gpu_count > 0 else 'killable'
    slurm_template = "#! /bin/sh\n" \
                "#SBATCH --job-name={job_name} \n" \
                "#SBATCH --output={slurm_output}{job_name}.out\n" \
                "#SBATCH --error={slurm_output}{job_name}.err\n" \
                "#SBATCH --account=gpu-research\n" \
                "#SBATCH --partition={partition}\n" \
                "#SBATCH --time=1440\n" \
                "#SBATCH --nodes=1\n" \
                "#SBATCH --gpus={gpu_count}\n" \
                "#SBATCH --ntasks={job_count}\n" \
                "#SBATCH --cpus-per-task=4\n" \
                "#SBATCH --exclude=rack-gww-dgx1\n" \
                "cd /home/morg/students/ohavbarbi/multiAgent/\n" \
                "for i in $(seq 1 {job_count}); do\n" \
                "  srun -n1 -c1 --mem-per-cpu=2000M {python_command} --id $((i-1)) &\n"\
                "done\n" \
                "\nwait"
    python_command = 'python ' + ' '.join([v for v in sys.argv if v != '--slurm'])
    
    print(python_command)

    slurm_command = slurm_template.format(job_name=job_name, python_command=python_command, slurm_output=slurm_output, gpu_count=gpu_count, partition=partition, job_count=job_count)
    with open(f'{slurm_output}/{job_name}.slurm', 'w') as f:
        f.write(slurm_command)
    os.system(f"sbatch {slurm_output}/{job_name}.slurm")

def launch_job(name, index, output, gpu_count, **kwargs):
    job_name = f"{name}_{index}" if index is not None else name
    slurm_output = output + name + '/slurm/'
    partition = 'gpu-h100-killable' if gpu_count > 0 else 'killable'
    slurm_template = "#! /bin/sh\n " \
                "#SBATCH --job-name={job_name} \n" \
                "#SBATCH --output={slurm_output}{job_name}.out\n" \
                "#SBATCH --error={slurm_output}{job_name}.err\n" \
                "#SBATCH --account=gpu-research\n" \
                "#SBATCH --partition={partition}\n" \
                "#SBATCH --time=1440\n" \
                "#SBATCH --nodes=1\n" \
                "#SBATCH --gpus={gpu_count}\n" \
                "#SBATCH --exclude=rack-gww-dgx1\n" \
                "source /a/home/cc/students/cs/ohavbarbi/.bashrc\n" \
                "cd /home/morg/students/ohavbarbi/multiAgent/\n" \
                "{python_command}\n"
    python_command = 'python ' + ' '.join([v for v in sys.argv if v != '--slurm'])

    if index is not None:
        python_command = python_command + f' --id {index}'
    
    print(python_command)
    slurm_command = slurm_template.format(job_name=job_name, python_command=python_command, slurm_output=slurm_output, gpu_count=gpu_count, partition=partition)
    with open(f'{slurm_output}/{job_name}.slurm', 'w') as f:
        f.write(slurm_command)
    os.system(f"sbatch {slurm_output}/{job_name}.slurm")


def load_and_run_symmetric_games(game_list, output_path, model_info, run_parameters):
    rerun_mapping = dict()
    args_dict = {'url': model_info.get('url', ''), 'gpu_count': model_info.get('gpu_count', 0), 
                 'agent': dict(), 'load_turn_count': 0}

    for k in run_parameters:
        if k.startswith('accuser'):
            split_key = k.split('_')
            agent = split_key[0]
            new_key = '_'.join(split_key[1:])
            args_dict['agent'][new_key] = run_parameters[k]
        else:  
            args_dict[k] = run_parameters[k]
    
    index = args_dict.get('index', 0)
    assert 'index' in args_dict

    args_dict['intervention_config'] = output_path + f'intervention_config_{index}.conf'
    os.makedirs(output_path, exist_ok=True)

    with open(args_dict['intervention_config'], 'w') as f:
        json.dump(args_dict, f, indent=4)
    
    print("Argument dictionary: " + ', '.join([f'{k}={args_dict[k]}' for k in args_dict.keys()]))

    stage_name = run_parameters.get('stage_name', '')

    for game_file in game_list:
        game_name = game_file.split('/')[-1].split('.')[0]
        experiment_name = f'ind_{index}_{game_name}'
        load_and_run_symmetric(game_file, experiment_name, model_info['model_name'], output_path, **args_dict)


def load_and_run_games(game_list, output_path, model_info, run_parameters):
    if run_parameters['symmetric']:
        return load_and_run_symmetric_games(game_list, output_path, model_info, run_parameters)
    rerun_mapping = dict()
    args_dict = {'url': model_info.get('url', ''), 'gpu_count': model_info.get('gpu_count', 0), 
                 'intel': dict(), 'accuser': dict(), 'load_turn_count': 0}

    for k in run_parameters:
        if k.startswith('intel') or k.startswith('accuser'):
            split_key = k.split('_')
            agent = split_key[0]
            new_key = '_'.join(split_key[1:])
            args_dict[agent][new_key] = run_parameters[k]
        else:  
            args_dict[k] = run_parameters[k]
    
    index = args_dict.get('index', 0)
    assert 'index' in args_dict

    args_dict['intervention_config'] = output_path + f'intervention_config_{index}.conf'
    os.makedirs(output_path, exist_ok=True)

    with open(args_dict['intervention_config'], 'w') as f:
        json.dump(args_dict, f, indent=4)
    
    print("Argument dictionary: " + ', '.join([f'{k}={args_dict[k]}' for k in args_dict.keys()]))

    stage_name = run_parameters.get('stage_name', '')

    for game_file in game_list:
        game_name = game_file.split('/')[-1].split('.')[0]
        experiment_name = f'ind_{index}_{game_name}'
        load_and_run_asymmetric(game_file, experiment_name, model_info['model_name'], output_path, **args_dict)
    

def main(name, model, output_dir, **kwargs):
    model_info = {'url': kwargs.get('url', ''), 
                  'model_name': model,
                  'gpu_count': kwargs.get('gpu_count', 0)
    }

    if kwargs['game_list_dir'] is None:
        if kwargs['symmetric']:
            define_games_symmetric(name, output_dir + f'env_list/', **kwargs)
        else:
            define_games(name, output_dir + f'env_list/', **kwargs)
        return

    index = kwargs['index']
    if 'index' in kwargs:
        kwargs['game_list_dir'] = kwargs['game_list_dir'].replace('#', str(index))
    game_list = glob.glob(kwargs['game_list_dir'] + '*.json') # Collect games
    game_list = sorted(game_list)
    if len(game_list) > kwargs['game_count']:
        game_list = game_list[index*kwargs['game_count']:(index + 1)*kwargs['game_count']]
    assert len(game_list) == kwargs['game_count'], f"{len(game_list)} vs {kwargs['game_count']}"
    print(f"Total of {len(game_list)} games")

    for reset_count in kwargs['reset_game_count']:
        intel_entropy_thresholds = kwargs['intel_entropy_threshold']
        accuser_entropy_thresholds = kwargs['accuser_entropy_threshold']
        intel_entropy_thresholds = intel_entropy_thresholds if intel_entropy_thresholds is not None else [1]
        accuser_entropy_thresholds = accuser_entropy_thresholds if accuser_entropy_thresholds is not None else [1]

        for accuser_th, intel_th in itertools.product(accuser_entropy_thresholds, intel_entropy_thresholds):
            accuser_reset_count = reset_count if accuser_th < 1 else 0
            intel_reset_count = reset_count if intel_th < 1 else 0
            output_path = output_dir + f'th_{accuser_th}_{intel_th}_reset_{reset_count}/'

            kwargs['accuser_classifier'] = copy_classifier(kwargs['accuser_classifier'], output_path + '/classifiers/', 'accuser')
            kwargs['intel_classifier'] = copy_classifier(kwargs['intel_classifier'], output_path + '/classifiers/', 'intel')

            run_parameters = prepare_run_parameters({'accuser_entropy_threshold': accuser_th, 'accuser_reset_game_count': accuser_reset_count,
                                                        'intel_entropy_threshold': intel_th, 'intel_reset_game_count': intel_reset_count,
                                                    'stage_name': 'entropy_classifier'},
                                                        **kwargs)
            load_and_run_games(game_list, output_path, model_info,
                            run_parameters=run_parameters)
            

            

def prepare_run_parameters(arguments, **kwargs):
    run_parameters = arguments
    run_parameters['game_length'] = kwargs['game_length']
    run_parameters['suspect_count'] = kwargs['suspect_count']
    run_parameters['golden_intel'] = kwargs.get('golden_intel', False)
    run_parameters['positive'] = kwargs['positive']
    run_parameters['complete'] = kwargs['complete']
    run_parameters['index'] = kwargs['index']
    run_parameters['intel_classifier'] = kwargs['intel_classifier']
    run_parameters['accuser_classifier'] = kwargs['accuser_classifier']
    run_parameters['symmetric'] = kwargs['symmetric']
    return run_parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--model", type=str, default='llama3.1-70b')
    parser.add_argument("--output", type=str)
    parser.add_argument('--gpu_count', type=int, default=0)
    parser.add_argument('--node', type=str)
    parser.add_argument('--url', type=str)
    parser.add_argument('--slurm', action='store_true')
    parser.add_argument('--count', type=int)
    parser.add_argument('--base_path', type=str, default='')
    parser.add_argument('--game_count', type=int, default=30)
    parser.add_argument('--game_length', type=int, default=31)
    parser.add_argument('--suspect_count', type=int)

    parser.add_argument('--accuser_entropy_threshold', type=float, nargs='+')
    parser.add_argument('--accuser_classifier', type=str, required=True)
    parser.add_argument('--intel_entropy_threshold', type=float, nargs='+')
    parser.add_argument('--intel_classifier', type=str, required=False)
    parser.add_argument('--reset_game_count', type=float, nargs='+')
    
    parser.add_argument('--sessions', type=int, default=0)
    parser.add_argument('--id', type=int, required=False, default=0)

    parser.add_argument('--new_name', type=str, required=False)
    parser.add_argument('--golden_intel', action='store_true', default=False)
    parser.add_argument('--game_list_dir', type=str)
    parser.add_argument('--complete', action='store_true')

    parser.add_argument('--symmetric', action='store_true', default=False)

    args = parser.parse_args()

    if args.slurm:
        other_params = {'game_list_dir': args.game_list_dir}
        if args.sessions > 0:
            name_index = sys.argv.index('--name')
            output_index = sys.argv.index('--output')
            for i in range(args.sessions):
                i = i + 2
                sess_name = args.name + f'_{i}'
                sys.argv[name_index + 1] = sess_name
                output = args.output + args.name + '/'
                sys.argv[output_index + 1] = output
                launch_on_slurm(sess_name, output, args.count, args.gpu_count, **other_params)
        else:
            launch_on_slurm(args.name, args.output, args.count, args.gpu_count, **other_params)
    else:
        argument_dict = {'positive': True, 'game_length': args.game_length,
            'base_path': args.base_path,
            'game_count': args.game_count,
            'gpu_count': args.gpu_count,

            'accuser_entropy_threshold': args.accuser_entropy_threshold,
            'accuser_classifier': args.accuser_classifier,
            'intel_entropy_threshold': args.intel_entropy_threshold,
            'intel_classifier': args.intel_classifier,
            'reset_game_count': args.reset_game_count,

            'game_list_dir': args.game_list_dir,
            'complete': args.complete,
            'index': args.id,
            'golden_intel': args.golden_intel,
            'symmetric': args.symmetric
        }
        if args.url is not None or args.node is not None:
            assert (args.url is not None) != (args.node is not None), f"Can't have both node input {args.node} and url {args.url} - they override each other."
            if args.url:
                argument_dict['url'] = args.url
            else:
                argument_dict['url'] = f'http://{args.node}.cs.tau.ac.il:8000/v1/'
        if args.suspect_count is not None:
            argument_dict['suspect_count'] = args.suspect_count
        if args.new_name is not None:
            argument_dict['new_name'] = args.new_name
        output = args.output + args.name + '/'

        main(args.name, args.model, output, **argument_dict)
