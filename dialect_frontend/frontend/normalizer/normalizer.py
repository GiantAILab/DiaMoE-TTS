import re
import unicodedata
from builtins import str as unicode

from frontend.normalizer.numbers import normalize_numbers


def remove_duplicate_symbols(input_string, symbols):
    pattern = re.compile(rf"[{re.escape(symbols)}]+")
    return pattern.sub(lambda x: x.group(0)[0], input_string)


def normalize(sentence):
    """Normalize English text."""
    # preprocessing
    sentence = re.sub(r"[——▁《》【】<=>{}()「」（）『』#&@“”^_|…\\/]", "", sentence)
    sentence = remove_duplicate_symbols(sentence, ",.?!，。？！")
    sentence = unicode(sentence)
    sentence = normalize_numbers(sentence)
    sentence = "".join(
        char for char in unicodedata.normalize("NFD", sentence) if unicodedata.category(char) != "Mn"
    )  # Strip accents
    sentence = sentence.lower()
    sentence = re.sub(r"[^ a-z'.,?!\-]", "", sentence)
    sentence = sentence.replace("i.e.", "that is")
    sentence = sentence.replace("e.g.", "for example")
    return sentence
