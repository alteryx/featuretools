# Define regular expressions to be used by NatLang primitives

import re
from string import punctuation

DELIMITERS = re.escape(r"[-.!?]") + " \n\t"
DELIMITERS = f"[{DELIMITERS}]"
