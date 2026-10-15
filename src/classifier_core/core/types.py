from enum import Enum


class ReviewLabelType(str, Enum):
    SPAM = "Spam"
    HIGH_UTILITY = "High Utility"
    LOW_UTILITY = "Low Utility"


class BouncerResultType(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
