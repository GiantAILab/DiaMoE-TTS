from abc import ABC
from abc import abstractmethod
from typing import List

import numpy as np

import torch
from frontend.g2p_en import G2p
from g2pM import G2pM

import numpy as np
import re

from frontend.normalizer.normalizer import normalize
from frontend.punctuation import get_punctuations
from frontend.vocab import Vocab
from frontend.zh_normalization.text_normlization import TextNormalizer
from frontend.symbols import en_phonemes


__all__ = ["Phonetics", "English", "EnglishCharacter", "Chinese"]


def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for _ in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word


class Phonetics(ABC):
    @abstractmethod
    def __call__(self, sentence):
        pass

    @abstractmethod
    def phoneticize(self, sentence):
        pass

    @abstractmethod
    def numericalize(self, phonemes):
        pass


class English(Phonetics):
    """Normalize the input text sequence and convert into pronunciation id sequence.

    https://github.com/Kyubyong/g2p/blob/master/g2p_en/g2p.py

    phonemes = ["<pad>", "<unk>", "<s>", "</s>"] + [
        'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
        'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH',
        'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
        'EY2', 'F', 'G', 'HH',
        'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L',
        'M', 'N', 'NG', 'OW0', 'OW1',
        'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH',
        'UH0', 'UH1', 'UH2', 'UW',
        'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
    """

    LEXICON = {
        # key using lowercase
        "AI".lower(): [["EY0", "AY1"]],
    }

    def __init__(self, vocab_phones=None):
        self.backend = G2p()
        self.backend.cmu.update(English.LEXICON)
        self.phonemes = list(self.backend.phonemes)
        self.punctuations = get_punctuations("en")
        self.vocab = Vocab(self.phonemes + self.punctuations)
        self.vocab_phones = vocab_phones
        self.punc = "、：，；。？！“”‘’':,;.?!…"
        self.text_normalizer = TextNormalizer()

    def phoneticize(self, sentence):
        """Normalize the input text sequence and convert it into pronunciation sequence.
        Args:
            sentence (str): The input text sequence.
        Returns:
            List[str]: The list of pronunciation sequence.
        """
        start = self.vocab.start_symbol
        end = self.vocab.end_symbol

        _phonemes, _word2ph = self.backend(sentence)
        _phonemes = [item for item in _phonemes if not item.isspace()]
        _phonemes = ([] if start is None else [start]) + _phonemes + ([] if end is None else [end])
        _word2ph = ([] if start is None else [1]) + _word2ph + ([] if end is None else [1])

        phonemes = []
        word2ph = []
        idx = 0
        for w in _word2ph:
            flag = True
            for i in range(idx, idx + w):
                p = _phonemes[i]
                if p in self.vocab.stoi and not p.isspace():
                    phonemes.append(p)
                    continue
                flag = False
            idx += w

            if flag:
                word2ph.append(w)

        return phonemes, word2ph

    def get_splited_phonemes_tones(self, sentence, return_other: bool=False):
        sentences = [sentence.strip() for sentence in re.split(r"\n+", sentence)]
        phonemes_list = []
        word2phs_list = []
        for sentence in sentences:
            sentence = normalize(sentence)
            phonemes, word2phs = self.phoneticize(sentence)

            phonemes = phonemes[1:-1]
            word2phs = word2phs[1:-1]
            # phn if (phn in self.vocab_phones and phn not in self.punc) else "sp" for phn in phonemes
            phonemes = [phn for phn in phonemes]
            if len(phonemes) != 0:
                phonemes_list += phonemes
                word2phs_list += word2phs

        phonemes_wo_tone = []
        tones = []
        for ph in phonemes_list:
            if ph in en_phonemes:
                tn = 0
                if re.search(r"\d$", ph):
                    tn = int(ph[-1]) + 1
                    ph = ph[:-1]
                phonemes_wo_tone.append(ph)
                tones.append(tn)
            else:
                phonemes_wo_tone.append(ph)
                tones.append(0)
        if return_other:
            return phonemes_wo_tone, tones, word2phs_list, "", []
        return phonemes_wo_tone, tones, word2phs_list

    def _p2id(self, phonemes: List[str]) -> np.array:
        phone_ids = [self.vocab_phones[item] for item in phonemes]
        return np.array(phone_ids, np.int64)

    def get_input_ids(
        self, sentence: str, merge_sentences: bool = False, to_tensor: bool = True
    ) -> torch.Tensor:
        sentences = self.text_normalizer._split(sentence, lang="en")

        phones_list = []
        temp_phone_ids = []
        for sentence in sentences:
            sentence = normalize(sentence)
            phones = self.phoneticize(sentence)
            # remove start_symbol and end_symbol
            phones = phones[1:-1]
            phones = [phn for phn in phones if not phn.isspace()]
            # replace unk phone with sp
            phones = [phn if (phn in self.vocab_phones and phn not in self.punc) else "_" for phn in phones]
            if len(phones) != 0:
                phones_list.append(phones)

        if merge_sentences:
            merge_list = sum(phones_list, [])
            # rm the last 'sp' to avoid the noise at the end
            # cause in the training data, no 'sp' in the end
            if merge_list[-1] == "sp":
                merge_list = merge_list[:-1]
            phones_list = []
            phones_list.append(merge_list)

        for part_phones_list in phones_list:
            phone_ids = self._p2id(part_phones_list)
            if to_tensor:
                phone_ids = torch.tensor(phone_ids)
            temp_phone_ids.append(phone_ids)

        result = {}
        result["phone_ids"] = temp_phone_ids

        return result

    def numericalize(self, phonemes):
        """Convert pronunciation sequence into pronunciation id sequence.
        Args:
            phonemes (List[str]): The list of pronunciation sequence.
        Returns:
            List[int]: The list of pronunciation id sequence.
        """
        ids = [self.vocab.lookup(item) for item in phonemes if item in self.vocab.stoi]
        return ids

    def reverse(self, ids):
        """Reverse the list of pronunciation id sequence to a list of pronunciation sequence.
        Args:
            ids (List[int]): The list of pronunciation id sequence.
        Returns:
            List[str]: The list of pronunciation sequence.
        """
        return [self.vocab.reverse(i) for i in ids]

    def __call__(self, sentence):
        """Convert the input text sequence into pronunciation id sequence.
        Args:
            sentence(str): The input text sequence.
        Returns:
            List[str]: The list of pronunciation id sequence.
        """
        return self.numericalize(self.phoneticize(sentence))

    @property
    def vocab_size(self):
        """Vocab size."""
        return len(self.vocab)


class EnglishCharacter(Phonetics):
    """Normalize the input text sequence and convert it into character id sequence."""

    def __init__(self):
        self.backend = G2p()
        self.graphemes = list(self.backend.graphemes)
        self.punctuations = get_punctuations("en")
        self.vocab = Vocab(self.graphemes + self.punctuations)

    def phoneticize(self, sentence):
        """Normalize the input text sequence.
        Args:
            sentence(str): The input text sequence.
        Returns:
            str: A text sequence after normalize.
        """
        words = normalize(sentence)
        return words

    def numericalize(self, sentence):
        """Convert a text sequence into ids.
        Args:
            sentence (str): The input text sequence.
        Returns:
            List[int]:
                List of a character id sequence.
        """
        ids = [self.vocab.lookup(item) for item in sentence if item in self.vocab.stoi]
        return ids

    def reverse(self, ids):
        """Convert a character id sequence into text.
        Args:
            ids (List[int]): List of a character id sequence.
        Returns:
            str: The input text sequence.
        """
        return [self.vocab.reverse(i) for i in ids]

    def __call__(self, sentence):
        """Normalize the input text sequence and convert it into character id sequence.
        Args:
            sentence (str): The input text sequence.
        Returns:
            List[int]: List of a character id sequence.
        """
        return self.numericalize(self.phoneticize(sentence))

    @property
    def vocab_size(self):
        """Vocab size."""
        return len(self.vocab)


class Chinese(Phonetics):
    """Normalize Chinese text sequence and convert it into ids."""

    def __init__(self):
        # self.opencc_backend = OpenCC('t2s.json')
        self.backend = G2pM()
        self.phonemes = self._get_all_syllables()
        self.punctuations = get_punctuations("cn")
        self.vocab = Vocab(self.phonemes + self.punctuations)

    def _get_all_syllables(self):
        all_syllables = set([syllable for k, v in self.backend.cedict.items() for syllable in v])
        return list(all_syllables)

    def phoneticize(self, sentence):
        """Normalize the input text sequence and convert it into pronunciation sequence.
        Args:
            sentence(str): The input text sequence.
        Returns:
            List[str]: The list of pronunciation sequence.
        """
        # simplified = self.opencc_backend.convert(sentence)
        simplified = sentence
        phonemes = self.backend(simplified)
        start = self.vocab.start_symbol
        end = self.vocab.end_symbol
        phonemes = ([] if start is None else [start]) + phonemes + ([] if end is None else [end])
        return self._filter_symbols(phonemes)

    def _filter_symbols(self, phonemes):
        cleaned_phonemes = []
        for item in phonemes:
            if item in self.vocab.stoi:
                cleaned_phonemes.append(item)
            else:
                for char in item:
                    if char in self.vocab.stoi:
                        cleaned_phonemes.append(char)
        return cleaned_phonemes

    def numericalize(self, phonemes):
        """Convert pronunciation sequence into pronunciation id sequence.
        Args:
            phonemes(List[str]): The list of pronunciation sequence.
        Returns:
                List[int]: The list of pronunciation id sequence.
        """
        ids = [self.vocab.lookup(item) for item in phonemes]
        return ids

    def __call__(self, sentence):
        """Convert the input text sequence into pronunciation id sequence.
        Args:
            sentence (str): The input text sequence.
        Returns:
            List[str]: The list of pronunciation id sequence.
        """
        return self.numericalize(self.phoneticize(sentence))

    @property
    def vocab_size(self):
        """Vocab size."""
        return len(self.vocab)

    def reverse(self, ids):
        """Reverse the list of pronunciation id sequence to a list of pronunciation sequence.
        Args:
        ids (List[int]): The list of pronunciation id sequence.
        Returns:
            List[str]: The list of pronunciation sequence.
        """
        return [self.vocab.reverse(i) for i in ids]
