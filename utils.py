import re
import json
import numpy as np
from scipy import stats

def load_message(msg, verbose=False):
    msg = re.sub(r'("value":\s*)(True|False)', lambda match: f'{match.group(1)}{match.group(2).lower()}', msg)
    cur_value = msg
    cur_value = cur_value.replace('\n', '')
    cur_value = '{' + cur_value.split('{')[-1].split('}')[0] + '}'
    cur_value = re.sub("//([a-zA-Z \\d]*)\n", '', cur_value)  # remove any comment strings in the json
    try:
        loaded = json.loads(cur_value)
        keys = list(loaded.keys())
        loaded = {k.strip(): loaded[k] for k in loaded.keys()}
        return loaded
    except json.JSONDecodeError as e:
        match = re.search("\"action\": (.*), \"info\": (.*)}", cur_value)
        if match:
            return {"thoughts": "invalid", "action": match.group(1), "info": match.group(2)}
        if verbose:
            print(cur_value)
            raise e
        return None


def try_int(x):
    try:
        return int(x)
    except Exception as e:
        return -1

def try_bool(val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'. 
    Raises ValueError if 'val' is anything else.
    """
    if isinstance(val, bool):
        return val
    if val is None:
        raise ValueError(f"invalid truth value None")
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError(f"invalid truth value {val}")


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
        raise Exception("Invalid scoring used")


def get_entropy(logprobs):
    return -np.sum(np.exp(logprobs) * logprobs, axis=-1)


def get_varentropy(logprobs):
    entropy = get_entropy(logprobs)
    probs = np.exp(logprobs)
    logprobs = -logprobs
    deviation = (logprobs - entropy) ** 2
    varentropy = np.sum(deviation * probs, axis=-1)
    return varentropy