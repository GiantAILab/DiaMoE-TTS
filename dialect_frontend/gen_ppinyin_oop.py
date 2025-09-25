from tools.mix_wrapper import Preprocessor
from tqdm import tqdm
import argparse

def replace_english_punctuation_with_chinese(text: str) -> str:
    en_to_zh_punct = {
        ",": "，",
        ".": "。",
        "?": "？",
        "!": "！",
        ":": "：",
        ";": "；",
        "(": "（",
        ")": "）",
        "[": "【",
        "]": "】",
        "{": "｛",
        "}": "｝",
        "<": "《",
        ">": "》",
        "\"": "“",
        "'": "‘",
        "-": "－",
        "_": "＿",
        "&": "＆",
        "@": "＠",
        "/": "／",
        "\\": "、",
        "|": "｜",
        "`": "｀",
        "~": "～",
        "^": "＾"
    }

    for en, zh in en_to_zh_punct.items():
        text = text.replace(en, zh)
    return text


def get_ppinyin(infile, outfile=None, lang: str='ZH'):

    preprocessor = Preprocessor()
    frontend = preprocessor.frontend[lang]
    print(f"Load {lang} frontend success!")
    data_pinyin_lines = []
    changed_lines = []
    oop_set = set()
    with open(infile, "r", encoding="utf-8") as fr:
        for line in tqdm(fr.readlines()):
            # utt, text, ppinyin = line.strip().split("\t")
            line_list = line.strip().split("\t")
            if line_list==[] or line_list==['']:
                continue
            idx = line_list[0]
            text = line_list[1]
            phones_list, word2ph, tones_list, ppinyins, oop, zhongwens = frontend.get_splited_phonemes_tones([text])
            zhongwens = replace_english_punctuation_with_chinese(zhongwens)
            ppinyins = replace_english_punctuation_with_chinese(ppinyins)
            if text != zhongwens:
                changed_lines.append(f"{idx}\t{text}\t{zhongwens}")
            oop_set.update(oop)
            line_list.append(ppinyins)
            line_list[1] = zhongwens
            data_pinyin_lines.append(line_list)
    if outfile is not None:
        with open(outfile, "w", encoding="utf-8") as fw:
            for line_list in data_pinyin_lines:
                fw.write("\t".join(line_list) + "\n")
    log_file = outfile + ".changed_log.txt"
    oop_set = list(oop_set)
    if oop_set:
        changed_lines.append('Below are OOP items:')
        changed_lines.append(' '.join(oop_set))
    with open(log_file, "w", encoding="utf-8") as f:
        for log in changed_lines:
            f.write(log + "\n")
    print(f"Recorded {len(changed_lines)} inconsistent text entries, log written to: {log_file}")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="input data.list")
    parser.add_argument("-o", "--output", type=str, help="generate data_pinyin.list")
    parser.add_argument("--lang", type=str, default="ZH", help="language")
    args = parser.parse_args()
    infile = args.input
    outfile = args.output
    lang = args.lang
    get_ppinyin(infile=infile, outfile=outfile, lang=lang)


