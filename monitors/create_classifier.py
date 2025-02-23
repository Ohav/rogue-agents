import sys
sys.path.append('/home/morg/students/ohavbarbi/multiAgent')
import glob, tqdm, json, pickle
import os
import pandas as pd
import numpy as np
import plotly.express as px
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

from monitors.metric_utils import *
from monitors import train_classifier
from intervention.intervention_models import PolynomialModel, FullyConnected
from variants.asymmetric.intel_agent import IntelAgent
from variants.symmetric.symm_agent import SymmAgent

from graphs import graph_utils


class EmptyPipe:
    def __init__(self):
        self.logprobs = None
        pass

    def get_last_logprobs(self):
        return self.logprobs


def load_metrics(files, env, collect):
    games_arr = []

    accuse_action = 2 if env == 'symmetric' else 3
    agent_cls = SymmAgent if env == 'symmetric' else IntelAgent

    for cur_file in tqdm.tqdm(files):
        loaded_game = graph_utils.load_asymmetric_data({'data': cur_file}, load_game_list=False, verbose=False)['data']
        loaded_game['exp_name'] += '_' + cur_file.split('/')[-3].split('_')[-1]
        scores = cur_file.replace('.json', '.scores_json')
        with open(cur_file, 'r') as f:
            game = json.load(f)['game']

        if collect:
            with open(scores, 'r') as f:
                game_scores = json.load(f)
            json_pipe = EmptyPipe()
            agent = agent_cls(json_pipe, 'James', 'None', 6, {}, None)
    
        turns = sorted([int(t) for t in range(len(game))])

        entropies = []
        for t in turns:
            turn_values = {'turn_count': t}
            if collect:
                scores = game_scores[str(t)]
                json_pipe.logprobs = scores
                _ = agent.calc_agent_entropy()
                turn_values['max entropy'] = agent.last_answer_entropy
                turn_values['max varentropy'] = agent.last_answer_varentropy
                turn_values['max kurtosis'] = agent.last_answer_kurtosis

                if agent.last_answer_varentropy <= 0:
                    raise Exception()

            else:
                scores = game[t]['entropy']
                turn_values['max entropy'] = scores[0]
                turn_values['max varentropy'] = scores[1]
                turn_values['max kurtosis'] = scores[2]

            if len(entropies) > 2:
                turn_values['entropy delta'] = turn_values['max entropy'] - entropies[-2]['max entropy']
            else:
                turn_values['entropy delta'] = 0

            turn_values['max entropy'] = max(turn_values['max entropy'], 0.0000001)
            turn_values['max varentropy'] = max(turn_values['max varentropy'], 0.0000001)
            # turn_values['max entropy'] = max(turn_values['max entropy'], 0.00001)
            # turn_values['max varentropy'] = max(turn_values['max varentropy'], 0.00001)
            entropies.append(turn_values)
        score_df = pd.DataFrame.from_dict(entropies)
        score_df = score_df.set_index('turn_count')
        
        loaded_game = loaded_game.join(score_df)
        loaded_game['is_game_successful'] = ((loaded_game.action == accuse_action) & (loaded_game['info'] == loaded_game.culprit)).any()
        if loaded_game['is_game_successful'].iloc[0]:
            loaded_game['Will game fail?'] = 'Accuser Falses'
            loaded_game.loc[loaded_game['turn_count'] % 2 == 1, 'Will game fail?'] = 'Intel Falses'
        else:
            loaded_game['Will game fail?'] = 'Accuser Trues'
            loaded_game.loc[loaded_game['turn_count'] % 2 == 1, 'Will game fail?'] = 'Intel Trues'
        loaded_game['Agent Turn Count'] = loaded_game['turn_count'] if env == 'symmetric' else loaded_game['turn_count'] // 2 
        games_arr.append(loaded_game)
    games_df = pd.concat(games_arr)
    games_df['entropy delta absolute'] = np.abs(games_df['entropy delta'])
    games_df['max entropy'] = np.log(games_df['max entropy'])
    games_df['max varentropy'] = np.log(games_df['max varentropy'])
    return games_df


def draw_boxplots(name, games, path):
    fig, axes = plt.subplots(len(METRIC_COLUMNS), 1, figsize=(27, 5 * len(METRIC_COLUMNS)))
    if len(METRIC_COLUMNS) == 1:
        axes = [axes]
    fig.suptitle('Will game fail?\nModel game comparison, with different metrics')
    # hue_order = ['Accuser Trues', 'Accuser Falses', 'Intel Trues', 'Intel Falses']
    hue_order = sorted(list(games['Will game fail?'].unique()))

    LOG = False

    for i, c in enumerate(METRIC_COLUMNS):
        if LOG:
            games[c + '_log'] = np.log(games[c])
            sns.boxplot(ax=axes[i], data=games[~games[c].isna()], x='Agent Turn Count', y=c + '_log', hue='Will game fail?', hue_order=hue_order, showfliers=False).set(title=f'{name} Character*Action Metric')
        else:
            sns.boxplot(ax=axes[i], data=games[~games[c].isna()], x='Agent Turn Count', y=c, hue='Will game fail?', hue_order=hue_order, showfliers=False).set(title=f'{name} Character*Action Metric')
    if LOG:
        plt.savefig(f'{path}{name}_metrics_log.png')
    else:
        plt.savefig(f'{path}{name}_metrics.png')
    plt.show()


def train_all_columns(name, games, output_path, validation_ratio=3, verbose=True):
    all_games = list(games.exp_name.unique())
    np.random.seed(777)
    np.random.shuffle(all_games)
    if validation_ratio > 0:
        valid_len = (len(all_games) // 10) * validation_ratio
        validation_games = games[games.exp_name.apply(lambda x: x in all_games[:valid_len])]
        all_valid, _ = train_classifier.load_training_data(validation_games, METRIC_COLUMNS)
    else:
        valid_len = 0
        all_valid = None
    train_games = games[games.exp_name.apply(lambda x: x in all_games[valid_len:])]


    all_train, norm_params = train_classifier.load_training_data(train_games, METRIC_COLUMNS)
    flat_train = train_classifier.flatten_data(all_train)

    col_names = np.array(['None', 'ent', 'var', 'kur'])
    os.makedirs(output_path + name, exist_ok=True)
    
    trained_models = dict()
    vals = []
    for i in range(1, len(METRIC_COLUMNS) + 1):
        for cols in itertools.combinations(range(1, len(METRIC_COLUMNS) + 1), i):
            cols = list(cols)
            feature_indices = [0] + cols
            for degree in [1, 2, 3, 4, 5]:
                polynomial_model = PolynomialModel(feature_indices=feature_indices, normalization_params=norm_params, degree=degree)
                # polynomial_model = FullyConnected(feature_indices=feature_indices, normalization_params=norm_params, degree=degree)
                
                # Features are selected on init, so the model gets all_train
                polynomial_model.fit(flat_train)
                model_name = f'{name}/poly_' + '_'.join(col_names[cols]) + f'_deg_{degree}'
                with open(f'{output_path}{model_name}.pkl', 'wb') as f:
                    pickle.dump(polynomial_model, f)

                if all_valid is not None:
                    performance = train_classifier.test_model_theory(polynomial_model, all_valid)
                    performance['name'] = model_name
                else:
                    performance = pd.DataFrame.from_dict([{'name': model_name, 'intervention_gain':0, 'th':0}])
                vals.append(performance)
        
    random_model = lambda x: np.random.random(size=len(x))
    model_name = f'{name}/random_model'
    if all_valid is not None:
        performance = train_classifier.test_model_theory(random_model, all_valid)
        performance['name'] = model_name
    else:
        performance = pd.DataFrame.from_dict([{'name': model_name, 'intervention_gain':0, 'th':0}])
    vals.append(performance)
    
    
    vals = pd.concat(vals)
    hyper_parameter_search_lines = []
    
    if all_valid is not None:
        title = name.split('_')[0] + ' ' + name.split('_')[1].replace('acc', 'Accuser').replace('intel', 'Intel') + ' Classifiers'
        fig = px.line(vals, x='precision', y='recall', color='name', title=title) 
        fig.write_image(f'{output_path}{name}/{name}_models.png')
    
        for sub_name in vals['name'].unique():
            cur = vals[vals['name'] == sub_name]
            best = cur[cur.intervention_gain == cur.intervention_gain.max()].iloc[0]
            hyper_parameter_search_lines.append(f"Best for {sub_name}: gain={best.intervention_gain:.3f}, f1={best.f1_score:.3f} (r={best.recall:.3f}, p={best.precision:.3f} th={best.threshold:.2f})")
    
        # real_best = vals[vals.intervention_gain == vals.intervention_gain.max()].iloc[0]
        real_best = vals[vals.intervention_gain == vals.intervention_gain.min()].iloc[0]
        hyper_parameter_search_lines.append(f"Absolute best {name}: {real_best['name']} gain={real_best.intervention_gain:.3f}, f1={real_best.f1_score:.3f} (r={real_best.recall:.3f}, p={real_best.precision:.3f} th={real_best.threshold:.2f})")
        with open(f'{output_path}{name}/res.txt', 'w') as f:
            for l in hyper_parameter_search_lines:
                f.write(l + '\n')
        vals.to_csv(f'{output_path}{name}/model_res.csv')
    if verbose:
        print('\n'.join(hyper_parameter_search_lines))
    return vals

def main(name, paths, output, collect):
    if not paths.endswith('*.json'):
        paths = paths + '*.json'
    files = glob.glob(paths)

    games = load_metrics(files, env='asymmetric', collect=collect)

    draw_boxplots(name, games, output)

    res_acc = train_all_columns(name + '_acc', games[games.cur_player == 0], output, validation_ratio=3)
    res_intl = train_all_columns(name + '_intel', games[games.cur_player == 1], output, validation_ratio=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--sigmoid', action='store_true')
    args = parser.parse_args()

    main(args.name, args.input_path, args.output_path, args.collect)

    

    

    