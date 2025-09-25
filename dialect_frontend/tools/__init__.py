import torch

from frontend.symbols import phonemes, language_tone_start_map, language_id_map, mms

from .chinese_wrapper import g2p_splited_modal as chinese
from .chinese_wrapper import zh_from_pinyin
from .dialect_wrapper import g2p_splited_wrapper as dialect
from .english_wrapper import g2p_splited_wrapper as english

# _language_cleaner = {"ZH": chinese, "EN": english}
_phonemes_to_id = {s: i for i, s in enumerate(phonemes)}


def text_cleaner(text: str, language: str, frontend, from_pinyin: bool=False, ppinyin: str=None):
    if language == "ZH":
        if from_pinyin:
            return zh_from_pinyin(frontend, text, ppinyin)
        return chinese(frontend, text, mms)
    elif language == "EN":
        return english(frontend, text)
    else:  # 方言
        return dialect(frontend, text, from_pinyin, ppinyin)

        


def cleaned_text_to_sequence(phones, tones, language):
    if language == "HN":
        language = "ZH"
    phone_ids = [_phonemes_to_id[symbol] for symbol in phones]
    tone_start = language_tone_start_map[language]
    tone_ids = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    lang_ids = [lang_id for i in phone_ids]
    return phone_ids, tone_ids, lang_ids


def merge_text(texts: list, min_len: int):
    merged = []
    buffer = ""
    if len(texts) == 1:
        return texts
    for s in texts:
        if (len(buffer) + len(s)) < min_len:
            buffer += s
        else:
            buffer += s
            merged.append(buffer)
            buffer = ""
    if buffer != "":
        merged[-1] = merged[-1] + buffer
    return merged


def get_bert_feature(text, word2ph, model, tokenizer):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to("cuda")
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
    word2phone = word2ph
    phone_level_feature = []
    try:
        for i in range(len(word2phone)):
            repeat_feature = res[i].repeat(word2phone[i], 1)
            phone_level_feature.append(repeat_feature)
    except:
        raise RuntimeError(text)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T
