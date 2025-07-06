from enum import IntEnum, auto

class BadAnswerException(Exception):
    pass

class Errors(IntEnum):
    MISSING_THOUGHTS = auto()
    MISSING_ACTION = auto()
    BAD_ACTION = auto()
    MISSING_FACT = auto()
    BAD_FACT = auto()
    MISSING_CHARACTER = auto()
    BAD_CHARACTER = auto()
    INVALID_FORMAT = auto() # Completely invalid json format
    NO_ERROR = auto()


error_to_text = {
    Errors.INVALID_FORMAT: "Please make sure your answer is in the correct json format.",
    Errors.MISSING_ACTION: "Please make sure your answer is in the correct json format.",
    Errors.MISSING_THOUGHTS: "Please make sure your answer is in the correct json format.",
    Errors.BAD_ACTION: "Please make sure to select a valid action.",
    Errors.MISSING_FACT: "When action is 1, share a fact, answer must have a fact field with the relevant fact index.",
    Errors.BAD_FACT: "When action is 1, share a fact, answer must have a fact field with the relevant fact index in the correct range.",
    Errors.MISSING_CHARACTER: "When action is accuse, answer must have a character field with the relevant character index.",
    Errors.BAD_CHARACTER: "Character must be a number between 1 and the number of characters.",
    


    # Errors.BAD_PROPERTY: 'Note property must be a valid property from the list of properties you have of the culprit.',
    # Errors.MISSING_VALUE: "Answer must have a value field, indicating the value for the action taken",
    # Errors.BAD_VALUE_1: "When action is 1, value must be a boolean value True/False.",
    # Errors.BAD_VALUE_2: "When action is 2, value must be in the form property-value",
    # Errors.MISSING_CHARACTER_LIST: "When action is 2, share a broad message, you must add a character property with a comma seperated list of characters that have this property and value combination.",
    # Errors.BAD_CHARACTER_LIST: 'When action is 2, share a broad message, you must add a character property with a comma seperated list of characters that have this property and value combination.'

}