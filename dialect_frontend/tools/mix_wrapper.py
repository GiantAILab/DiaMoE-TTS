import torch

import tools.commons as commons
from tools import get_bert_feature
from tools.sentence import split_by_language
from tools import text_cleaner, cleaned_text_to_sequence

from frontend.symbols import zh_phonemes, en_phonemes
# from frontend.en_frontend import English as english_frontend
from frontend.zh_frontend import Frontend as chinese_frontend
# from frontend.henan_frontend import HenanFrontend as henan_frontend
# from frontend.yunbai_frontend import YunbaiFrontend as yunbai_frontend


def process_auto(text):
    _text, _lang = [], []
    sentences_list = split_by_language(text, target_languages=["zh", "en"])
    for sentence, lang in sentences_list:
        if sentence == "":
            continue
        _text.append(sentence)
        _lang.append(lang.upper())
    return _text, _lang


class Preprocessor:
    def __init__(self) -> None:
        self.frontend = {
            # "EN": english_frontend(vocab_phones=en_phonemes),
            "ZH": chinese_frontend(vocab_phones=zh_phonemes),
            # "HN": henan_frontend(vocab_phones=zh_phonemes),
            # "YB": yunbai_frontend(vocab_phones=zh_phonemes),
        }

    def get_text(self, text: str, lang: str, from_pinyin: bool=False, ppinyin: str=None,):
        output_ppinyin = ""
        oop = []
        dialect_phone = None
        if lang == "ZH" or lang == "EN":
            norm_text, phones, tones, word2ph = text_cleaner(text, lang, self.frontend[lang], from_pinyin, ppinyin)
        else:
            norm_text, phones, tones, word2ph, output_ppinyin, oop = text_cleaner(text, lang, self.frontend[lang], from_pinyin, ppinyin)
            dialect_phone = self.frontend[lang].vocab_phones
        phone_ids, tone_ids, lang_ids = cleaned_text_to_sequence(phones, tones, lang, dialect_phone)

        phone_ids = commons.intersperse(phone_ids, 0)
        tone_ids = commons.intersperse(tone_ids, 0)
        lang_ids = commons.intersperse(lang_ids, 0)
        for i in range(len(word2ph)):
            word2ph[i] = int(word2ph[i] * 2)
        word2ph[0] = int(word2ph[0] + 1)

        return norm_text, phones, word2ph, phone_ids, tone_ids, lang_ids, output_ppinyin, oop

    def get_text_bert(self, text: str, lang: str, berts, tokenizers):
        norm_text, phones, tones, word2ph = text_cleaner(text, lang, self.frontend[lang])
        phone_ids, tone_ids, lang_ids = cleaned_text_to_sequence(phones, tones, lang)

        phone_ids = commons.intersperse(phone_ids, 0)
        tone_ids = commons.intersperse(tone_ids, 0)
        lang_ids = commons.intersperse(lang_ids, 0)

        for i in range(len(word2ph)):
            word2ph[i] = int(word2ph[i] * 2)
        word2ph[0] = int(word2ph[0] + 1)

        if lang == "EN":
            bert_zh = torch.zeros(1024, len(phone_ids))
            bert_en = get_bert_feature(norm_text, word2ph, berts[lang], tokenizers[lang])
        else:
            bert_zh = get_bert_feature(norm_text, word2ph, berts["ZH"], tokenizers[lang])
            bert_en = torch.zeros(1024, len(phone_ids))

        assert bert_zh.shape[-1] == len(phone_ids), phone_ids
        assert bert_zh.shape[-1] == len(phone_ids), f"Bert seq len {bert_zh.shape[-1]} != {len(phone_ids)}"

        return norm_text, phones, word2ph, phone_ids, tone_ids, lang_ids, bert_zh, bert_en

    def single_lang_func(self, text, lang, from_pinyin: bool=False, ppinyin: str=None,):
        return self.get_text(text, lang, from_pinyin, ppinyin)

    def multi_lang_func(self, texts, berts, tokenizers):
        # norm_text, phones, phone_ids, word2ph, tone_ids, lang_ids, bert_zh, bert_en
        texts, langs = process_auto(texts)
        norm_text = ""
        bert_zh, bert_en = [], []
        phones, word2ph, phone_ids, tone_ids, lang_ids = [], [], [], [], []
        for idx, (_text, _lang) in enumerate(zip(texts, langs)):
            (
                temp_norm_text,
                temp_phones,
                temp_word2ph,
                temp_phone_ids,
                temp_tone_ids,
                temp_lang_ids,
                temp_bert_zh,
                temp_bert_en,
            ) = self.get_text_bert(_text, _lang, berts, tokenizers)
            if idx == 0:
                temp_phones = temp_phones[:-1]
                temp_phone_ids = temp_phone_ids[:-3]
                temp_tone_ids = temp_tone_ids[:-3]
                temp_lang_ids = temp_lang_ids[:-3]
                temp_bert_zh = temp_bert_zh[:, :-3]
                temp_bert_en = temp_bert_en[:, :-3]
            elif idx != len(texts) - 1:
                temp_phones = temp_phones[1:-1]
                temp_phone_ids = temp_phone_ids[2:-3]
                temp_tone_ids = temp_tone_ids[2:-3]
                temp_lang_ids = temp_lang_ids[2:-3]
                temp_bert_zh = temp_bert_zh[:, 2:-3]
                temp_bert_en = temp_bert_en[:, 2:-3]
            else:
                temp_phones = temp_phones[1:]
                temp_phone_ids = temp_phone_ids[2:]
                temp_tone_ids = temp_tone_ids[2:]
                temp_lang_ids = temp_lang_ids[2:]
                temp_bert_zh = temp_bert_zh[:, 2:]
                temp_bert_en = temp_bert_en[:, 2:]
            norm_text += temp_norm_text
            phones += temp_phones
            word2ph += temp_word2ph
            phone_ids += temp_phone_ids
            tone_ids += temp_tone_ids
            lang_ids += temp_lang_ids

            bert_zh.append(temp_bert_zh)
            bert_en.append(temp_bert_en)

        bert_zh = torch.concatenate(bert_zh, dim=1)
        bert_en = torch.concatenate(bert_en, dim=1)

        return norm_text, phones, word2ph, phone_ids, tone_ids, lang_ids, bert_zh, bert_en
