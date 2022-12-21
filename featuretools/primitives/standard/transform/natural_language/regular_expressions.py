# Define regular expressions to be used by NatLang primitives
from string import punctuation

DELIMITERS = " \n\t"
DELIMITERS = f"[{DELIMITERS}]"

PUNCTUATION_AND_WHITESPACE = f"[{punctuation}\n\t ]"
