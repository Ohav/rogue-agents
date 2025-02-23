from scipy import stats
import itertools
import utils
import numpy as np

METRIC_COLUMNS = ['max entropy', 'max varentropy', 'max kurtosis'] 

def get_score(logprobs, score_method):
    if not isinstance(logprobs, np.ndarray):
        logprobs = np.array([p[1] for p in logprobs])
    if score_method == 'entropy':
        return get_entropy(logprobs)
    elif score_method == 'varentropy':
        return get_varentropy(logprobs)
    elif score_method == 'kurtosis':
        return stats.kurtosis(logprobs)
    else:
        raise Exception()

def get_entropy(logprobs):
    return -np.sum(np.exp(logprobs) * logprobs, axis=-1)

def get_varentropy(logprobs):
    entropy = get_entropy(logprobs)
    probs = np.exp(logprobs)
    logprobs = -logprobs
    deviation = (logprobs - entropy) ** 2
    varentropy = np.sum(deviation * probs, axis=-1)
    return varentropy
    
def entropy_for_number_after_text(scores, text, limit=5):
    entropies = []
    for i in range(len(scores)):
        if text in scores[i][0][0].lower():
            index = i + 1
            found = utils.try_int(scores[index][0][0].strip()) >= 0
            while not found and index < i + 5:
                index = index + 1
                if index >= len(scores):
                    break
                found = utils.try_int(scores[index][0][0].strip()) >= 0
            if not found:
                # Just a mention of the text instead of a specific usage
                continue
            entropy = get_score(scores[index])
            entropies.append(entropy)
    # assert len(entropies) > 0
    return entropies

def arrange_entropies(entropy, name, as_dict=True):
    if len(entropy) == 0:
        return {f'{name}_avg': np.nan, f'{name}_max': np.nan, f'{name}_first': np.nan, f'{name}_last': np.nan, f'{name}_median': np.nan}
    avg = np.mean(entropy)
    max_val = max(entropy)
    first = entropy[0]
    last = entropy[-1]
    median = np.median(entropy)
    if as_dict:
        return {f'{name}_avg': avg, f'{name}_max': max_val, f'{name}_first': first, f'{name}_last': last, f'{name}_median': median, f'{name}_min': min(entropy)}
    return avg, max_val, first, last, median
        
def x_token_entropy(scores, token_count):
    return [get_score(scores[i]) for i in range(token_count)]

def get_x_token_entropy(scores, skip_first=False, as_dict=True):
    if skip_first:
        scores = scores[1:]
    top_5_tokens = x_token_entropy(scores, 5)
    max_5_token = max(top_5_tokens)
    first_token = top_5_tokens[0]
    mean_5_token = sum(top_5_tokens) / len(top_5_tokens)
    diff_token = max_5_token - mean_5_token
    if as_dict:
         return {'first_5_avg': mean_5_token, 'first_5_max': max_5_token, 'first_5_first': first_token, 'first_5_diff': diff_token}
    return  mean_5_token, max_5_token, first_token, diff_token

def get_conversation_entropy(scores, skip_first=True, as_dict=True):
    if skip_first:
        scores = scores[1:]
    entire_conversation = x_token_entropy(scores, len(scores))
    
    max_entire = max(entire_conversation)
    avg_entire = np.mean(entire_conversation)
    median_entire = np.median(entire_conversation)
    ninth_percentile = np.percentile(entire_conversation, 90)
    if as_dict:
        return {'entire_avg': avg_entire, 'entire_max': max_entire, 'entire_median': median_entire, 'entire_9th_percentile': ninth_percentile}
    return max_entire, avg_entire, median_entire, ninth_percentile

def get_char_times_action(scores, score_method='entropy'):
    logprobs = {'character': [], 'action': []}
    for i in range(len(scores)):
        for text in logprobs.keys():
            if text in scores[i][0][0].lower():
                index = i + 1
                found = utils.try_int(scores[index][0][0].strip()) >= 0
                while not found and index < i + 5:
                    index = index + 1
                    if index >= len(scores):
                        break
                    found = utils.try_int(scores[index][0][0].strip()) >= 0
                if found:
                    logprobs[text].append([p[1] for p in scores[index]])

    if (len(logprobs['character']) == 0) or (len(logprobs['action']) == 0):
        res = [get_score(np.array(logs), score_method) for logs in logprobs['action']]
        res += [get_score(np.array(logs), score_method) for logs in logprobs['character']]
        return res

    scores = []
    for i,j in itertools.product(range(len(logprobs['character'])), range(len(logprobs['action']))):
        char_logprobs = logprobs['character'][i]
        act_logprobs = logprobs['action'][j]
        combined = np.array([p1 + p2 for p1 in char_logprobs for p2 in act_logprobs])
        score = get_score(combined, score_method)
        scores.append(score)
    return scores