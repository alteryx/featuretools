# Define regular expressions to be used by NatLang primitives

from string import punctuation
import re

DELIMITERS = re.escape(r"[-.!?]") + " \n\t"
DELIMITERS = f"[{DELIMITERS}]"