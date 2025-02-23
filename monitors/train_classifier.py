import torch
from torch import optim
import numpy as np
import os
import copy
import pickle
import pandas as pd

from env_intel.intervention_models import PolynomialModel, normalize
from classifiers.metric_utils import METRIC_COLUMNS
MINIMUM_GAME_TURNS = 1

def load_training_data(game_df, columns=None, do_normalize=True):
    if columns is None:
        columns = METRIC_COLUMNS
    for col in columns:
        # No matter what columns were chosen - filter the default columns for invalids.
        game_df = game_df[~game_df[col].isna()]

    turn_input = game_df['turn_count'].values
    input_columns = [turn_input]
    for col in columns:
        input_columns.append(game_df[col].values)
    
    normalization_params = [(0, 15)]
    if do_normalize: 
        for col in input_columns[1:]:
            params = col.min(), col.max()
            normalization_params.append(params)

    data = []    
    for exp in game_df.exp_name.unique():
        cur_game = game_df.exp_name == exp
        game = game_df[cur_game]

        row = [col[cur_game] for col in input_columns]
        features = torch.Tensor(np.stack(row).T)

        labels = (1 - game.is_game_successful.values.astype(int)).reshape(-1, 1)
        labels = torch.Tensor(labels)
        data.append((features, labels))
    return data, normalization_params

def flatten_data(data):
    all_turns = []
    all_labels = []
    for g in data:
        all_turns.append(g[0])
        all_labels.append(g[1])

    all_turns = np.vstack(all_turns)
    all_labels = np.vstack(all_labels)
    indices = np.arange(len(all_turns))
    all_turns = torch.Tensor(all_turns[indices])
    all_labels = torch.Tensor(all_labels[indices])
    return all_turns, all_labels


def test_model(net, test_data, threshold=0.5):
    """
    Tests a classifier model vs the test data. 
    Test is done by iterating over each game, and checking whether it is reset. 
    A game is reset if it has even one turn where the classifier predicts True.
    Finally, the recall and precision are according to games reset and not turn based.
    """
    resets = 0
    game_will_fail = 0
    tp = 0
    signal_count = 0
    for i, (features, labels) in enumerate(test_data):       
        prediction = net(features) > threshold
        will_reset = prediction.any().numpy()
        label = labels.all()
        signal_count += sum(prediction) / len(prediction)

        tp += label & will_reset
        resets += will_reset
        game_will_fail += label

    recall = tp / game_will_fail
    if resets != 0:
        precision = tp / resets
    else:
        precision = 1
    ratio = resets / len(test_data)

    if (recall + precision) != 0:
        f1_score = 2 * recall * precision / (recall + precision)
    else:
        f1_score = 0
        
    return recall, precision, f1_score, ratio


def test_model_theory(net, test_data):
    label_vector = np.zeros(len(test_data))
    pred_vector = np.zeros(len(test_data))
    
    for i, (features, labels) in enumerate(test_data):       
        pred_vector[i] = max(net(features))
        label_vector[i] = labels.all()

    label_vector = label_vector.astype(bool)
    p_err = sum(label_vector) / len(label_vector)
    p_succ = 1 - p_err

    assert len(label_vector) > 0
    assert 0 < sum(label_vector)
    assert sum(label_vector) < len(label_vector)

    performances = []
    for threshold in np.arange(0, 1.01, 0.01):
        will_reset = (pred_vector > threshold).astype(bool)
        tp = sum(will_reset & label_vector)
        fp = sum(will_reset & (~label_vector))
        fn = sum((~will_reset) & label_vector)
        tn = sum((~will_reset) & (~label_vector))

        recall = tp / sum(label_vector)
        if sum(will_reset) != 0:
            precision = tp / sum(will_reset)   
            
            p_int_given_error = tp / sum(label_vector)
            p_int_given_success = fp / sum(~label_vector)
            
        else:
            precision = 0
            p_int_given_error = 0
            p_int_given_success = 0
        
        predicted_new_success = p_succ * (p_int_given_success) * p_succ \
                                + p_succ * (1 - p_int_given_success) \
                                + p_err * p_int_given_error * p_succ
        
        gain = predicted_new_success - p_succ

        predicted_new_err = p_err * p_int_given_error * p_err \
                                + p_err * (1-p_int_given_error) \
                                + p_succ * (p_int_given_success) * p_err

        intervention_gain = p_err - predicted_new_err
        assert np.isclose(gain, intervention_gain)
        # intervention_gain = gain
        assert (not np.isnan(intervention_gain))
        if recall + precision > 0:
            f1_score = 2 * recall * precision / (recall + precision)
        else:
            f1_score = 0

        performance = {'recall': recall, 'precision': precision, 'p_int_given_error': p_int_given_error,
                       'p_int_given_success': p_int_given_success, 'predicted_new_err': predicted_new_err,
                       'intervention_gain': intervention_gain, 'f1_score': f1_score,
                       'p_err': p_err, 'p_succ': p_succ,
                       'threshold': threshold}
        performances.append(performance)
    data = pd.DataFrame.from_dict(performances)
    assert ((~data.intervention_gain.isna()).all())
    return data


def train_game(net, train_games, test_games, name, num_epochs=10, learning_rate=0.1, verbose=False):
    criterion = nn.BCELoss()  # Binary Cross-Entropy loss
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=30)
    last_total_loss = 999 * len(train_games)

    best_model = None
    best_f1_score = 0

    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, (features, labels) in enumerate(train_games):
            labels = labels.float()
            if len(labels) <= 10:
                continue
            features = features[10:, :]
            labels = labels[10:]
            # Forward pass
            
            optimizer.zero_grad()
            outputs = net(features)
            if labels[0]:
                for i, output in enumerate(outputs[:, 0]):
                    if output > 0.5: 
                        loss = criterion(outputs[i:, :], labels[i:, :]) # Calculate loss at the point of failure prediction
                        break
                else: 
                    # If no failure prediction exceeded threshold, use final prediction
                    loss = criterion(outputs[-1, :], labels[0, :])
                loss = loss
            else:
                loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 100 == 99:
                scheduler.step()
        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_games)}")
        last_total_loss = total_loss
        recall, precision, f1_score, ratio = test_model(net, test_games)
        if verbose:
            print(f'Test performance: Recall: {recall:.3f} -- Precision {precision:.3f} -- Reset ratio: {ratio:.3f} -- F1 Score {f1_score}')
        if f1_score > best_f1_score:
            best_model = copy.deepcopy(net.state_dict())
            best_f1_score = f1_score

    if best_model is None:
        print("Failed Training")
        return None
    net.load_state_dict(best_model)
    recall, precision, f1_score, ratio = test_model(net, test_games)
    print(f'Recall: {recall:.3f} -- Precision {precision:.3f} -- Reset ratio: {ratio:.3f} -- F1 Score {f1_score}')
    torch.save(best_model, f'/home/morg/students/ohavbarbi/multiAgent/monitors/models/{name}_best_input_{net.input_size}.pth')

    return net



