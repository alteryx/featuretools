# Define regular expressions to be used by NatLang primitives

from string import punctuation
import re

DELIMITERS = set(punctuation) - {
    '"',
    ".",
    "'",
    ",",
    "-",
    ":",
    "@",
    "/",
    "\\",
}
DELIMITERS = "".join(list(DELIMITERS))
DELIMITERS = re.escape(f" {DELIMITERS}\n\t")