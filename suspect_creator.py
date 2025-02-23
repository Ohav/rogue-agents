import json
import numpy as np
from suspect import Suspect

class SuspectCreator:
    def __init__(self, attribute_file):
        with open(attribute_file, 'r') as json_file:
            self.attribute_file = json.load(json_file)
        self.attributes = list(self.attribute_file.keys())
        self.attribute_sizes = [len(self.attribute_file[k]) for k in self.attributes]

    def index_vector_to_dictionary(self, vector):
        suspect_dictionary = dict()
        for j in range(len(self.attributes)):
            attr_name = self.attributes[j]
            suspect_dictionary[attr_name] = self.attribute_file[attr_name][vector[j]]
        return suspect_dictionary

    def create_suspects(self, suspect_count):
        np_suspects = []
        i = 0
        while i < suspect_count:
            suspect = np.random.randint(0, self.attribute_sizes)
            duplicate = False
            for other_suspect in np_suspects:
                if np.all(other_suspect == suspect):
                    # They're the same
                    duplicate = True
                    break
            if not duplicate:
                np_suspects.append(suspect)
                i += 1

        suspects = []
        for i in range(len(np_suspects)):
            suspect = np_suspects[i]
            suspect_dictionary = self.index_vector_to_dictionary(suspect)
            suspect_object = Suspect(i, suspect_dictionary)
            suspects.append(suspect_object)

        return suspects


