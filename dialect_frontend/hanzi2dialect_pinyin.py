import pdb

import pandas as pd
import re
import argparse
def load_mapping(excel_path):
    df = pd.read_excel(excel_path)
    simple_map = {}
    polyphonic_map = {}

    for _, row in df.iterrows():
        hanzi = str(row["hanzi"]).strip()
        std_pinyin = str(row["standard_chinese_pinyin"]).strip() if pd.notna(row["standard_chinese_pinyin"]) else ""
        dialect = str(row["dialect_pinyin"]).strip()

        if std_pinyin == "":
            simple_map[hanzi] = dialect
        else:
            if ',' in std_pinyin:
                std_pinyins = std_pinyin.split(',')
                for std_py in std_pinyins:
                    polyphonic_map[(hanzi, std_py)] = dialect
            else:
                polyphonic_map[(hanzi, std_pinyin)] = dialect

    return simple_map, polyphonic_map

def split_pinyin_units(pinyin_line):
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
        raise ValueError("Unfinished #...# blocks appear in the pinyin.")
    return result

def split_with_single_hash(text):
    # Use regular expression matching to identify the continuous reading blocks marked as #...#.
    result = []
    pos = 0
    pattern = re.finditer(r"#([^#]+)#", text)
    for match in pattern:
        start, end = match.span()
        if start > pos:
            result.extend(list(text[pos:start]))  # Non-continuous part, add word by word
        result.append(text[start:end])  # The entire consecutive block is added together
        pos = end
    if pos < len(text):
        result.extend(list(text[pos:]))
    return result


def map_to_dialect(text_unit, pinyin_unit, simple_map, polyphonic_map):
    # Future may require more processing for consecutive pronunciation
    if text_unit.startswith("#"):
        return pinyin_unit
    else:
        return polyphonic_map.get((text_unit, pinyin_unit), simple_map.get(text_unit, pinyin_unit))


def convert_file(txt_path, excel_path, output_path):
    simple_map, polyphonic_map = load_mapping(excel_path)

    with open(txt_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                index, text, pinyin = line.split("\t")
            except ValueError:
                print(f"Invalid format, skipping line: {line}")
                continue
            text_units = split_with_single_hash(text)

            pinyin_units = split_pinyin_units(pinyin)

            assert len(text_units) == len(pinyin_units), f"Alignment failed: {text_units} vs {pinyin_units}"

            dialect_pinyin_units = []
            for t, p in zip(text_units, pinyin_units):
                dp = map_to_dialect(t, p, simple_map, polyphonic_map)
                dialect_pinyin_units.append(dp)

            dialect_line = " ".join(dialect_pinyin_units)
            fout.write(f"{index}\t{text}\t{dialect_line}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="input data.list")
    parser.add_argument("-o", "--output", type=str, help="generate data_pinyin.list")
    parser.add_argument("-d", "--dialect", type=str, help="which dialect")
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    dialect = args.dialect


    excel_path = f"./frontend/dialect/{dialect}/hanzi.xlsx"
    convert_file(input_path, excel_path, output_path)
