from enum import IntEnum, auto

class BadAnswerException(Exception):
    pass

class Errors(IntEnum):
    MISSING_THOUGHTS = auto()
    MISSING_ACTION = auto()
    BAD_ACTION = auto()
    MISSING_CHARACTER = auto()
    BAD_CHARACTER = auto()
    MISSING_PROPERTY = auto()
    BAD_PROPERTY = auto()
    MISSING_VALUE = auto()
    BAD_VALUE_1 = auto()
    BAD_VALUE_2 = auto()
    MISSING_CHARACTER_LIST = auto()
    BAD_CHARACTER_LIST = auto()
    INVALID_FORMAT = auto() # Completely invalid json format
    NO_ERROR = auto()


error_to_text = {
    Errors.INVALID_FORMAT: "Please make sure your answer is in the correct json format.",
    Errors.MISSING_THOUGHTS: "Please make sure your answer is in the correct json format.",
    Errors.MISSING_ACTION: "Please make sure your answer is in the correct json format.",
    Errors.BAD_ACTION: "Please make sure to select a valid action.",
    Errors.MISSING_CHARACTER: "When action is request or accuse, answer must have a character field. If action is request, it should be the character you want to ask about. If action is accuse, it should be the accused character.",
    Errors.BAD_CHARACTER: "Character must be a number between 1 and the number of characters.",
    Errors.MISSING_PROPERTY: 'When action is 1, request, property field is mandatory.',
    Errors.BAD_PROPERTY: 'Note property must be a valid property from the list of properties you have of the culprit.',
    Errors.MISSING_VALUE: "Answer must have a value field, indicating the value for the action taken",
    Errors.BAD_VALUE_1: "When action is 1, value must be a boolean value True/False.",
    Errors.BAD_VALUE_2: "When action is 2, value must be in the form property-value",
    Errors.MISSING_CHARACTER_LIST: "When action is 2, share a broad message, you must add a character property with a comma seperated list of characters that have this property and value combination.",
    Errors.BAD_CHARACTER_LIST: 'When action is 2, share a broad message, you must add a character property with a comma seperated list of characters that have this property and value combination.'

}