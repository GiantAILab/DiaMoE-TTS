from frontend.en_frontend import English as EnFrontend
from frontend.normalizer import normalize
import re


def g2p_splited_wrapper(frontend: EnFrontend, text: str):
    # text = re.sub(r"[——▁《》【】<=>{}()（）#&@“”^_|\\/]", "", text)
    norm_text = normalize(text)
    phones_list, tones_list, word2ph = frontend.get_splited_phonemes_tones(norm_text)
    phones_list = ["_"] + phones_list + ["_"]
    tones_list = [0] + tones_list + [0]
    word2ph = [1] + word2ph + [1]

    return norm_text, phones_list, tones_list, word2ph
