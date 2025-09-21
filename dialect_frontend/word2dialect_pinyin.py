# TODO: Word matching currently does not support standard_chinese_pinyin-related polyphonic word functionality
# ✅ Complete integrated script: includes step 1 adding #, step 2 replacing pinyin, integrated 3-step processing

import pandas as pd
import re
from tqdm import tqdm
import argparse
def load_word_dicts_v4(excel_path):
    '''Read word dictionary'''
    df = pd.read_excel(excel_path)

    # Blocked judgment
    def is_blocked(val):
        return str(val).strip().lower() == "true"

    df = df[~df['blocked'].apply(is_blocked)]

    text_map = {}
    hash_only_map = {}

    for _, row in df.iterrows():
        word = str(row["word"]).strip()
        pinyin = str(row["dialect_pinyin"]).strip()

        text_map[word] = pinyin
        if "#" in word:
            hash_only_map[word] = pinyin

    sorted_keys = sorted(text_map.keys(), key=lambda x: -len(x.replace("#", "")))
    return text_map, hash_only_map, sorted_keys


def insert_hash_marks_aligned(text: str, pinyin: str, hash_only_map: dict):
    '''Insert possible new #liaison rules#'''
    text_tokens = tokenize_text(text)
    pinyin_tokens = tokenize_pinyin(pinyin)
    assert len(text_tokens) == len(pinyin_tokens), f"Character count and pinyin count mismatch: {len(text_tokens)} vs {len(pinyin_tokens)}"

    sorted_keys = sorted(hash_only_map.keys(), key=lambda x: -len(x.replace("#", "")))
    i = 0
    while i < len(text_tokens):
        for word in sorted_keys:
            word_plain = word.replace("#", "")
            l = len(word_plain)
            if i + l > len(text_tokens):
                continue
            segment = ''.join(text_tokens[i:i + l])
            if segment == word_plain:
                text_tokens[i:i + l] = [f"#{segment}#"] + [""] * (l - 1)
                joined_pinyin = ' '.join(pinyin_tokens[i:i + l])
                pinyin_tokens[i:i + l] = [f"#{joined_pinyin}#"] + [""] * (l - 1)
                # print(f"[Add #] Matched word: {word} → Add #{segment}#, pinyin #{joined_pinyin}#")
                i += l - 1
                break
        i += 1

    new_text = ''.join([t for t in text_tokens if t])
    new_pinyin = ' '.join([p for p in pinyin_tokens if p])
    return new_text, new_pinyin


def tokenize_text(text):
    result = []
    pos = 0
    pattern = re.finditer(r"#([^#]+)#", text)
    for match in pattern:
        start, end = match.span()
        if start > pos:
            result.extend(list(text[pos:start]))
        result.append(text[start:end])
        pos = end
    if pos < len(text):
        result.extend(list(text[pos:]))
    return result


def tokenize_pinyin(pinyin_line):
    tokens = pinyin_line.strip().split()
    result = []
    buffer = []
    in_hash = False
    for token in tokens:
        if token.startswith("#") and not token.endswith("#"):
            in_hash = True
            buffer = [token]
        elif in_hash:
            buffer.append(token)
            if token.endswith("#"):
                result.append(" ".join(buffer))
                in_hash = False
        else:
            result.append(token)
    if in_hash:
        raise ValueError("Pinyin contains unclosed #")
    return result


def apply_replacements(text_units, pinyin_units, replacements):
    '''Word-level dialect pinyin replacement'''
    i = 0
    while i < len(text_units):
        for word, rep_pinyin in replacements:
            word_units = tokenize_text(word)
            match_len = len(word_units)
            if i + match_len <= len(text_units) and text_units[i:i + match_len] == word_units:
                # print(f"[Replace] Matched word: {word} → Replace pinyin: {rep_pinyin}")
                pinyin_units[i:i + match_len] = [rep_pinyin] + [""] * (match_len - 1)
                i += match_len - 1
                break
        i += 1
    return [p for p in pinyin_units if p != ""]


def process_file_all_steps(input_path, word_excel_path, output_path):
    text_map, hash_only_map, _ = load_word_dicts_v4(word_excel_path)
    replacements = [(k, v) for k, v in text_map.items()]
    replacements.sort(key=lambda x: -len(x[0].replace("#", "")))

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin):
            line = line.strip()
            if not line:
                continue
            try:
                index, text, pinyin_line = line.split("\t")
            except ValueError:
                print(f"Skip format exception line: {line}")
                continue

            # Step 1: Add hash marks based on hash_only_map
            text_with_hash, pinyin_with_hash = insert_hash_marks_aligned(text, pinyin_line, hash_only_map)

            # Step 2: Tokenize
            text_units = tokenize_text(text_with_hash)
            pinyin_units = tokenize_pinyin(pinyin_with_hash)
            if len(text_units) != len(pinyin_units):
                print(f"[Warning] Text and pinyin not aligned: {index}")
                continue

            # Step 3: Replace pinyin
            new_pinyin_units = apply_replacements(text_units, pinyin_units, replacements)

            fout.write(f"{index}\t{text_with_hash}\t{' '.join(new_pinyin_units)}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="input data.list")
    parser.add_argument("-o", "--output", type=str, help="generate data_pinyin.list")
    parser.add_argument("-d", "--dialect", type=str, help="which dialect")
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    dialect = args.dialect

    # Example paths:
    # input_path = f"./output/{dialect}_hanzi.txt"
    # output_path = f"./output/{dialect}_word.txt"


    if dialect == 'putonghua':
        word_excel_path = f"frontend/dialect/{dialect}/base_word.xlsx"
    else:
        word_excel_path = f"frontend/dialect/{dialect}/word.xlsx"
    process_file_all_steps(input_path, word_excel_path, output_path)
