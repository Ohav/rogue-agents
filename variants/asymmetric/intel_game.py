from variants.asymmetric.intel_env import IntelEnv
from game_memory import IntelGameMemory
from consts import *
from agent_pipes import TransformersPipe
import argparse, tqdm, json, sys, traceback
from variants.asymmetric.intervention_manager_asymmetric import InterventionManagerAsymmetric
from intervention.exceptions import ResetGameException

def run_from_file(json_file, model, turn, task_info, path, **kwargs):
    llm_pipe = TransformersPipe(model, **kwargs)
    assert 'max_turn_count' in task_info['general']
    original_max_turn_count = task_info['general']['max_turn_count']

    env = IntelEnv(llm_pipe, attribute_file=DEFAULT_ATTRIBUTE_FILE, task_info=task_info)
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    env.set_from_dict(json_data['env_info'])
    turn = min(turn, len(json_data['game']))
    if turn % 2 == 1:
        turn -= 1
    
    if turn > 0:
        chosen_turn = json_data['game'][turn]
        comm_lines = chosen_turn['user_prompt'].split('\n')[1:]
        for line in comm_lines:
            if line.startswith('The current turn is'):
                break
            if line == '':
                continue
            env.add_to_communication(line) 
        env.possible_suspects = set(chosen_turn['remaining_suspects'])
    else:
        assert len(env.accuser.comm_channel) == 0
        assert len(env.possible_suspects) == env.suspect_count
    
    env.turn_count = turn
    env.cur_player = turn % 2

    memory = IntelGameMemory(path, env)
    memory.game_flow = json_data['game'][:turn]
    memory.first_turn = turn

    if 'intervention_config' in kwargs:
        config_json = kwargs['intervention_config']
        with open(config_json, 'r') as f:
            configs = json.load(f)
    else:
        configs = {'accuser': dict(), 'intel': dict()}
    accuser_intervention_args = configs.get('accuser', dict())
    intel_interventions_args = configs.get('intel', dict())
    for k in kwargs.keys():
        if k.startswith('intel_'):
            new_key = '_'.join(k.split('_')[1:])
            intel_interventions_args[new_key] = kwargs[k]
        elif k.startswith('accuser_'):
            new_key = '_'.join(k.split('_')[1:])
            accuser_intervention_args[new_key] = kwargs[k]

    env.agents[0].intervention_manager = InterventionManagerAsymmetric(env, **accuser_intervention_args, verbose=False)
    env.agents[1].intervention_manager = InterventionManagerAsymmetric(env, **intel_interventions_args, verbose=False)

    finished = False
    while not finished:
        try:
            for v in game_loop(env, memory, dump_features=kwargs.get('dump_features', False)):
                pass
            finished = True
        except ResetGameException as e:
            print(f"Resetting game! Turn: {env.turn_count + 1}. Game turn: {env.game_turn}", file=sys.stderr)
            turns_env_ran = env.turn_count + 1
            # Clean agents, reset knowledge. Update max turn count, but leave current turn as is.
            env.reset()
            env.turn_count = turns_env_ran
            env.max_turn_count = original_max_turn_count + turns_env_ran
            env.previous_run_turns = turns_env_ran
            assert env.max_turn_count - env.turn_count == original_max_turn_count, f"From {env.turn_count} to {env.max_turn_count}"
            env.reset_count += 1

    memory.dump_memory()


def game_loop(env, game_memory, dump_features, **kwargs):
    game_memory.dump_logprobs = dump_features
    is_done = False
    forced = False
    request_info = dict()
    next_player = 0
    env.agents[0].pipe.set_save_logprobs(True)

    # Start game loop
    for turn_count in tqdm.tqdm(range(env.turn_count, env.max_turn_count)):
        agent = env.agents[env.cur_player]
        answer = None

        try:
            (full_prompt, answer, validity), intervention_name = agent.act(env)
            forced = intervention_name != 'None' 

            if env.cur_player == 0:
                is_done, should_switch, action, request_info, impact, best_impact = env.handle_accuser(answer)
                next_player = 1 if should_switch else 0
            else:
                # Intel always switches back to accuser, and cannot win the game.
                action, impact, best_impact = env.handle_intel(answer, request_info)
                next_player = 0
            
            turn_result = {
                'cur_player': env.cur_player,
                'user_prompt': full_prompt[1]['content'], 
                'answer': answer,
                'turn_count': env.turn_count,
                'action': action,
                'is_done': is_done,
                'next_player': next_player,
                'remaining_suspects': list(env.possible_suspects),
                'impact': impact,
                'best_impact': best_impact,
                'validity': validity,
                'forced': forced,
                'forced_name': intervention_name,
                'errors': agent.errors,
                'entropy': agent.calc_agent_entropy()
            }
            agent.errors = dict()
        except ResetGameException as e:
            # Log current turn, then raise the same exception
            next_player = 0
            turn_result = {
                'cur_player': env.cur_player,
                'user_prompt': 'None', 
                'answer': '{"thoughts": "Reset intervention", "action": -1}',
                'turn_count': env.turn_count,
                'action': -1,
                'is_done': False,
                'next_player': next_player,
                'remaining_suspects': list(env.possible_suspects),
                'impact': 0,
                'best_impact': 0,
                'validity': True,
                'forced': True,
                'forced_name': str(e),
                'errors': agent.errors,
                'entropy': agent.calc_agent_entropy()
            }
            game_memory.log_turn(turn_result)
            raise e

        except Exception as e:
            print(f"Turn {env.turn_count} failed. Returned answer (if exists) was:\n{answer}\n\nError:\n{e}\n", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            print(f"Cancelling game with {e}")
            failed_turn_result = {
                'cur_player': env.cur_player,
                'user_prompt': 'See error', 
                'answer': answer,
                'turn_count': env.turn_count,
                'action': -1,
                'is_done': True,
                'next_player': next_player,
                'remaining_suspects': list(env.possible_suspects),
                'impact': -1,
                'best_impact': -1,
                'validity': False,
                'forced': False,
                'forced_name': "None",
                'error': str(e)
            }
            game_memory.log_turn(failed_turn_result)
            game_memory.delete()
            return

        game_memory.log_turn(turn_result)

        if game_memory.dump_logprobs:
            game_memory.logprobs_dict[env.turn_count] = agent.pipe.get_last_logprobs()

        yield turn_result
        if is_done:
            break
        env.cur_player = next_player
        env.turn_count += 1

    game_memory.dump_memory()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribute_file", type=str, default=DEFAULT_ATTRIBUTE_FILE)
    parser.add_argument('--message_file', type=str, default="json_files/messages_asymmetric_positive.json")
    parser.add_argument('--suspect_count', type=int, default=10)
    parser.add_argument('--turn_count', type=int, default=31)
    parser.add_argument('--model', type=str)
    parser.add_argument('--model_url', type=str)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--name", type=str, default='')
    args = parser.parse_args()


    llm_pipe = TransformersPipe(args.model, **{'url': args.model_url})
    task_info = {
       "general": {
            "suspect_count": args.suspect_count,
            "golden_intel": False,
            'max_turn_count': args.turn_count
        },
        "message_file": args.message_file
    }
    env = IntelEnv(llm_pipe, attribute_file=args.attribute_file, task_info=task_info)
    memory = IntelGameMemory(args.output_dir + args.name, env)
    game_loop(env, memory, False)