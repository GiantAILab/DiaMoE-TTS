import pandas as pd
import argparse

def load_tone_mapping(tone_excel_path):
    df = pd.read_excel(tone_excel_path)
    tone_map = {}
    for _, row in df.iterrows():
        contour = str(row["contour"]).strip()
        mark = str(row["mark"]).strip()
        tone_map[contour] = mark
    return tone_map

def replace_tone_marks_in_ipa(ipa_line, tone_map):
    # Replace tone contour marks one by one
    for contour, mark in tone_map.items():
        ipa_line = ipa_line.replace(contour, mark)
    return ipa_line

def process_ipa_file(input_path, tone_excel_path, output_path):
    tone_map = load_tone_mapping(tone_excel_path)

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                index, text, ipa_line = line.split("\t")
            except ValueError:
                print(f"[Skip] Format error line: {line}")
                continue

            replaced_ipa_line = replace_tone_marks_in_ipa(ipa_line, tone_map)
            fout.write(f"{index}\t{text}\t{replaced_ipa_line}\n")

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
    # input_path = f"./output/{dialect}_ipa1.txt"
    # output_path = f"./output/{dialect}_ipa_format.txt"


    if dialect == 'putonghua':
        tone_excel_path = f"frontend/dialect/{dialect}/base_tone.xlsx"  # Pinyin to IPA mapping table
    else:
        tone_excel_path = f"frontend/dialect/{dialect}/tone.xlsx"



    process_ipa_file(input_path, tone_excel_path, output_path)