import re
from typing import List


def g2p_wrapper(frontend, text: str) -> List[str]:
    pinyins = frontend.get_phonemes(text)
    return pinyins


def g2p_splited_wrapper(frontend, text: str) -> List[str]:
    # text = re.sub("[a-zA-Z]+", "", text)
    # text = re.sub(r"[——▁《》【】<=>{}()「」（）『』#&@“”^_|…\\/]", "", text)
    norm_texts = frontend.text_normalizer.normalize(text)
    phones_list, word2ph, tones_list = frontend.get_splited_phonemes_tones(norm_texts)
    phones_list = ["_"] + phones_list + ["_"]
    tones_list = [0] + tones_list + [0]
    word2ph = [1] + word2ph + [1]
    return "".join(norm_texts), phones_list, tones_list, word2ph


def zh_from_pinyin(frontend, text: str, ppinyin: str) -> List[str]:
    # text = re.sub("[a-zA-Z]+", "", text)
    # text = re.sub(r"[——▁《》【】<=>{}()「」（）『』#&@“”^_|…\\/]", "", text)
    # norm_texts = frontend.text_normalizer.normalize(text)
    norm_texts = text.replace("《", "").replace("》", "").replace("“", "").replace("”", "").replace(" ", "").replace("：", ",").replace("；", ",")
    ppinyin_list = ppinyin.replace(":", ",").replace(";", ",").split(" ")
    assert len(norm_texts) == len(ppinyin_list), print(f"error in {text}")
    phones_list, word2ph, tones_list = frontend.get_splited_phonemes_tones_ppinyin(norm_texts, ppinyin_list)
    phones_list = ["_"] + phones_list + ["_"]
    tones_list = [0] + tones_list + [0]
    word2ph = [1] + word2ph + [1]
    return "".join(norm_texts), phones_list, tones_list, word2ph


def get_modal(mm: str, mms: dict):
    try:
        MP = [mms[mm[1:-1]]]
    except:
        MP = ["MM0"]
    return (mm[1], MP, [6], [1])


def g2p_splited_modal(frontend, text: str, modal: dict) -> List[str]:
    pattern = re.compile(r"(<[^>]+>)")
    norm_text = ""
    phones = []
    word2phs = []
    tones = []

    parts = pattern.split(text)
    parts = [part for part in parts if part]
    for idx, part in enumerate(parts):
        if pattern.match(part):
            n1, n2, n3, n4 = get_modal(part, modal)
        else:
            n1, n2, n3, n4 = g2p_splited_wrapper(frontend, part)
            n2 = n2[1:-1]
            n3 = n3[1:-1]
            n4 = n4[1:-1]
        if idx == 0:
            n2 = ["_"] + n2
            n3 = [0] + n3
            n4 = [1] + n4
        elif idx == len(parts) - 1:
            n2 = n2 + ["_"]
            n3 = n3 + [0]
            n4 = n4 + [1]
        norm_text += n1
        phones += n2
        tones += n3
        word2phs += n4
    return norm_text, phones, tones, word2phs
