import os

import yaml


class Polyphonic:
    def __init__(self):
        with open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "polyphonic.yaml"), "r", encoding="utf-8"
        ) as polyphonic_file:
            # 解析yaml
            polyphonic_dict = yaml.load(polyphonic_file, Loader=yaml.FullLoader)
        self.polyphonic_words = polyphonic_dict["polyphonic"]

    def correct_pronunciation(self, word, pinyin):
        # 词汇被词典收录则返回纠正后的读音
        if word in self.polyphonic_words.keys():
            pinyin = self.polyphonic_words[word]
        # 否则返回原读音
        return pinyin
