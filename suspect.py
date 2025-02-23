import numpy as np
from consts import get_attribute_message


class Suspect:
    def __init__(self, suspect_id, suspect_dictionary):
        self.suspect_id = suspect_id
        self.suspect_info = suspect_dictionary

    def generate_suspect_description(self):
        special_keys = {'pants', 'pants_color', 'shirt_color', 'shirt'}
        normal_keys = set(self.suspect_info.keys()) - special_keys
        normal_keys = sorted(list(normal_keys))  # Remove randomization from suspect info
        description = ', '.join([get_attribute_message(k, self.suspect_info[k]) for k in normal_keys])
        description += f", they have {self.suspect_info['pants']}, {self.suspect_info['pants_color']} pants"
        description += f" and a {self.suspect_info['shirt_color']}, {self.suspect_info['shirt']} shirt."
        return description

    def get_fact(self, count=1):
        attrs = np.random.choice(list(self.suspect_info), count, replace=False)
        results = []
        for attr in attrs:
            results.append((attr, self.suspect_info[attr]))
        return results

    def is_fact_true(self, fact):
        attr, value = fact
        return value == self.suspect_info[attr]
    
    def __str__(self):
        return self.generate_suspect_description()



