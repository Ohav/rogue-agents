from variants.symmetric.symm_env import SymmEnv
from game_memory import SymmGameMemory
from consts import *
from agent_pipes import TransformersPipe
import tqdm, json, sys
import traceback

from intervention.exceptions import ResetGameException
from variants.symmetric.intervention_manager_symmetric import InterventionManagerSymmetric

def run_from_file(json_file, model, turn, task_info, path, **kwargs):
    llm_pipe = TransformersPipe(model, **kwargs)
    assert 'max_turn_count' in task_info['general']
    original_max_turn_count = task_info['general']['max_turn_count']

    env = SymmEnv(llm_pipe, attribute_file=DEFAULT_ATTRIBUTE_FILE, task_info=task_info)
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    env.set_from_dict(json_data['env_info'])
    turn = min(turn, len(json_data['game']))
    if turn % 2 == 1:
        turn -= 1
    
    if turn > 0:
        chosen_turn = json_data['game'][turn]
        comm_lines = chosen_turn['user_prompt'].split('\n')[1:]
        for l in comm_lines:
            if l.startswith('The current turn is'):
                break
            if l == '':
                continue
            env.add_to_communication(l) 
        env.possible_suspects = set(chosen_turn['remaining_suspects'])
    else:
        assert len(env.agents[0].comm_channel) == 0
        assert len(env.possible_suspects) == env.suspect_count
    
    env.turn_count = turn
    env.cur_player = turn % len(env.agents)

    memory = SymmGameMemory(path, env)
    memory.game_flow = json_data['game'][:turn]
    memory.first_turn = turn

    if 'intervention_config' in kwargs:
        config_json = kwargs['intervention_config']
        with open(config_json, 'r') as f:
            configs = json.load(f)
    else:
        configs = {'agent': dict()}
    
    agent_intervention_args = configs.get('agent', dict())
    for k in kwargs.keys():
        if k.startswith('agent_'):
            new_key = '_'.join(k.split('_')[1:])
            agent_intervention_args[new_key] = kwargs[k]

    for i in range(len(env.agents)):
        env.agents[i].intervention_manager = InterventionManagerSymmetric(env, **agent_intervention_args, verbose=False)
    env.reset_count = agent_intervention_args['reset_game_count']

    finished = False
    while not finished:
        try:
            for v in game_loop(env, memory, dump_features=kwargs.get('dump_features', False)):
                pass
            finished = True
        except ResetGameException as e:
            print(f"Resetting game! Turn: {env.turn_count + 1}. Game turn: {env.game_turn}. Resets: {env.reset_count}", file=sys.stderr)
            turns_env_ran = env.turn_count + 1
            # Clean agents, reset knowledge. Update max turn count, but leave current turn as is.
            env.reset()
            env.turn_count = turns_env_ran
            max_turn_count = (env.max_turn_count - env.previous_run_turns) + turns_env_ran
            env.max_turn_count = original_max_turn_count + turns_env_ran
            if env.max_turn_count != max_turn_count:
                print("Max turn count mismatch!", file=sys.stderr)
                print(env.turn_count, env.previous_run_turns, env.max_turn_count, max_turn_count, file=sys.stderr)
            env.previous_run_turns = turns_env_ran
            assert (env.max_turn_count - env.turn_count) == original_max_turn_count, f"From {env.turn_count} to {env.max_turn_count}"
            env.reset_count -= 1

    memory.dump_memory()


def game_loop(env, game_memory, dump_features, **kwargs):
    game_memory.dump_logprobs = dump_features
    is_done = False
    forced = False
    env.agents[0].pipe.set_save_logprobs(True)

    all_logprobs_dict = dict()

    # Start game loop
    for turn_count in tqdm.tqdm(range(env.turn_count, env.max_turn_count)):
        agent = env.agents[env.cur_player]
        answer = None

        try:
            (full_prompt, answer, validity), intervention_name = agent.act(env)
            forced = intervention_name != 'None' 
            is_done, action = env.step(answer)
            
            turn_result = {
                'cur_player': env.cur_player,
                'user_prompt': full_prompt[1]['content'], 
                'answer': answer,
                'turn_count': env.turn_count,
                'action': action,
                'is_done': is_done,
                'remaining_suspects': list(env.possible_suspects),
                'validity': validity,
                'forced': forced,
                'forced_name': intervention_name,
                'entropy': agent.calc_agent_entropy()
            }
            agent.errors = dict()
        except ResetGameException as e:
            # Log current turn, then raise the same exception
            turn_result = {
                'cur_player': env.cur_player,
                'user_prompt': "None", 
                'answer': '{"thoughts": "Reset intervention", "action": -1}',
                'turn_count': env.turn_count,
                'action': -1,
                'is_done': False,
                'remaining_suspects': list(env.possible_suspects),
                'validity': True,
                'forced': True,
                'forced_name': str(e),
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
                'remaining_suspects': list(env.possible_suspects),
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
        env.cur_player = (env.cur_player + 1) % len(env.agents)
        env.turn_count += 1

    game_memory.dump_memory()