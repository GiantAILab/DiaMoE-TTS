import pandas as pd
import  re
from tqdm import tqdm
import argparse
def load_word_pinyin_map(excel_path):
    df = pd.read_excel(excel_path)
    mapping = {}
    for _, row in df.iterrows():
        word = str(row['word']).strip()
        std_pinyin = str(row['standard_chinese_pinyin']).strip()
        dialect_pinyin = str(row['dialect_pinyin']).strip()
        mapping[(word, std_pinyin)] = dialect_pinyin
    return mapping

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

def match_and_replace(text_units, pinyin_units, word_pinyin_map):
    result = []
    i = 0
    while i < len(text_units):
        matched = False
        for (word, std_pinyin), dialect_pinyin in word_pinyin_map.items():
            word_units = tokenize_text(word)
            pinyin_seq = tokenize_pinyin(std_pinyin)
            L = len(word_units)
            if i + L <= len(text_units):
                if text_units[i:i+L] == word_units and pinyin_units[i:i+L] == pinyin_seq:
                    result.extend([dialect_pinyin])
                    i += L
                    matched = True
                    break
        if not matched:
            result.append(pinyin_units[i])
            i += 1
    return result

def process_file(input_path, excel_path, output_path):
    word_pinyin_map = load_word_pinyin_map(excel_path)

    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin):
            line = line.strip()
            if not line:
                continue
            try:
                index, text, pinyin_line = line.split('\t')
            except ValueError:
                print(f"[Format error] Skip line: {line}")
                continue

            text_units = tokenize_text(text)
            pinyin_units = tokenize_pinyin(pinyin_line)

            if len(text_units) != len(pinyin_units):
                print(f"[Alignment failed] {index}")
                fout.write(f"{index}\t{text}\t{pinyin_line}\n")
                continue

            replaced_pinyin = match_and_replace(text_units, pinyin_units, word_pinyin_map)
            fout.write(f"{index}\t{text}\t{' '.join(replaced_pinyin)}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="input data.list")
    parser.add_argument("-o", "--output", type=str, help="generate data_pinyin.list")

    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    excel_path = "frontend/dialect/putonghua/base_word.xlsx"  # Contains word / standard_chinese_pinyin / dialect_pinyin

    # Example paths:
    # input_path = f"./input/{dialect}_txt.txt"
    # output_path = f"./output/{dialect}_fix_pinyin.txt"


    process_file(input_path, excel_path, output_path)
