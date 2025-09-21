
import re
import argparse

def parse_tone(pinyin):
    match = re.match(r"#?([^\d#]+)(\d+)#?", pinyin)
    if match:
        return match.group(1), int(match.group(2))
    else:
        return pinyin, -1


def tokenize_text(text):
    result = []
    pos = 0
    for match in re.finditer(r"#([^#]+)#", text):
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


# def apply_tone_change(pinyin_units, tone_rule):
#     i = 0
#     result = []
#     while i < len(pinyin_units):
#         matched = False
#         for tone_seq, new_seq in tone_rule.items():
#             n = len(tone_seq)
#             if i + n > len(pinyin_units):
#                 continue
#             segment = pinyin_units[i:i + n]
#             tones = [parse_tone(p)[1] for p in segment]
#             if tones == list(tone_seq):
#                 new_segment = []
#                 for j, p in enumerate(segment):
#                     base, _ = parse_tone(p)
#                     new_tone = new_seq[j]
#                     if p.startswith("#") and p.endswith("#"):
#                         new_segment.append(f"#{base}{new_tone}#")
#                     else:
#                         new_segment.append(f"{base}{new_tone}")
#                 print(f"[Tone sandhi] {segment} → {new_segment}")
#                 result.extend(new_segment)
#                 i += n
#                 matched = True
#                 break
#         if not matched:
#             result.append(pinyin_units[i])
#             i += 1
#     return result
#
#
# def apply_char_specific_tone_change(pinyin_units, text_units, char_tone_rules):
#     assert len(pinyin_units) == len(text_units)
#     result = pinyin_units.copy()
#     for i in range(len(text_units) - 1):
#         char = text_units[i]
#         next_tone = parse_tone(pinyin_units[i + 1])[1]
#         if char in char_tone_rules:
#             rule = char_tone_rules[char]
#             new_tone = None
#             for k, v in rule.items():
#                 if k == "default":
#                     continue
#                 if isinstance(k, tuple) and next_tone in k:
#                     new_tone = v
#                     break
#             if new_tone is None:
#                 new_tone = rule.get("default", None)
#             if new_tone is not None:
#                 base, _ = parse_tone(pinyin_units[i])
#                 if pinyin_units[i].startswith("#") and pinyin_units[i].endswith("#"):
#                     result[i] = f"#{base}{new_tone}#"
#                 else:
#                     result[i] = f"{base}{new_tone}"
#                 print(f"[Specific tone sandhi] {char}({pinyin_units[i]}) → {result[i]}, because next char is tone {next_tone}")
#     return result

def apply_tone_change(pinyin_units, tone_rule):
    """
    Match tone sandhi rules from back to front; after successful matching, 
    retreat index to before the matched segment to avoid cross-matching with 
    already tone-changed parts.
    """

    result = pinyin_units.copy()
    i = len(result) - 1
    while i >= 0:
        matched = False
        for tone_seq, new_seq in tone_rule.items():
            n = len(tone_seq)
            start = i - n + 1
            if start < 0:
                continue

            segment = result[start:i + 1]
            tones = [parse_tone(p)[1] for p in segment]
            if tones == list(tone_seq):
                new_segment = []
                for j, p in enumerate(segment):
                    base, _ = parse_tone(p)
                    new_tone = new_seq[j]
                    if p.startswith("#") and p.endswith("#"):
                        new_segment.append(f"#{base}{new_tone}#")
                    else:
                        new_segment.append(f"{base}{new_tone}")
                print(f"[Tone sandhi] {segment} → {new_segment}")
                result[start:i + 1] = new_segment

                # Tone sandhi successful, refresh checkpoint
                i = start - 1
                matched = True
                break
        if not matched:
            i -= 1
    return result


def apply_char_specific_tone_change(pinyin_units, text_units, char_tone_rules):
    """
    Process specific character tone sandhi from back to front to maintain 
    the same scanning direction as apply_tone_change.
    """
    assert len(pinyin_units) == len(text_units)
    result = pinyin_units.copy()
    for i in range(len(text_units) - 2, -1, -1):  # Reverse scanning
        char = text_units[i]
        next_tone = parse_tone(result[i + 1])[1]  # Use updated result
        if char in char_tone_rules:
            rule = char_tone_rules[char]
            new_tone = None
            for k, v in rule.items():
                if k == "default":
                    continue
                if isinstance(k, tuple) and next_tone in k:
                    new_tone = v
                    break
            if new_tone is None:
                new_tone = rule.get("default", None)
            if new_tone is not None:
                base, _ = parse_tone(result[i])
                if result[i].startswith("#") and result[i].endswith("#"):
                    result[i] = f"#{base}{new_tone}#"
                else:
                    result[i] = f"{base}{new_tone}"
                print(f"[Specific tone sandhi] {char}({pinyin_units[i]}) → {result[i]}, because next char is tone {next_tone}")
    return result

def apply_all_tone_changes_blocked(pinyin_units, text_units, tone_rule, char_tone_rules, dialect, index=None, text=None):
    assert len(pinyin_units) == len(text_units)
    separators = {"，", "。", "！", "？", ",", ".", "!", "?", ";", "；"}

    result = []
    block_start = 0
    i = 0
    while i <= len(text_units):
        is_end = i == len(text_units)
        is_sep = not is_end and any(s in text_units[i] for s in separators)
        if is_end or is_sep:
            block_pinyins = pinyin_units[block_start:i]
            block_texts = text_units[block_start:i]
            block_pinyins = apply_tone_change(block_pinyins, tone_rule)
            block_pinyins = apply_char_specific_tone_change(block_pinyins, block_texts, char_tone_rules)
            if dialect == 'taibei' or dialect == 'gaoxiong':
                block_pinyins = taibei_group_nonfinal_tone_change(block_pinyins, block_texts)
            elif dialect == 'shanghai':
                block_pinyins = shanghai_group_tone_change(block_pinyins, block_texts)
            elif dialect == 'qingdao':
                block_pinyins = qingdao_tone2_change(block_pinyins)
            result.extend(block_pinyins)
            if is_sep:
                result.append(pinyin_units[i])
            block_start = i + 1
        i += 1

    return result


def process_tone_script_with_flexible_rules(input_path, output_path, dialect, TONE_RULES, CHAR_TONE_RULES):
    tone_rule = TONE_RULES.get(dialect, {})
    char_tone_rule = CHAR_TONE_RULES.get(dialect, {})

    print(f"[Info] Processing dialect: {dialect}")
    if not tone_rule:
        print(f"[Warning] No tone sequence sandhi rules, will skip this type of tone sandhi")
    if not char_tone_rule:
        print(f"[Warning] No specific character tone sandhi rules, will skip this type of tone sandhi")


    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                index, text, pinyin_line = line.split("\t")
            except ValueError:
                print(f"Skip format error line: {line}")
                continue

            text_units = tokenize_text(text)
            pinyin_units = tokenize_pinyin(pinyin_line)


            if len(text_units) != len(pinyin_units):
                print(f"[Warning] Alignment failed: {index} text and pinyin length mismatch")
                continue


            pinyin_units = apply_all_tone_changes_blocked(
                    pinyin_units, text_units, tone_rule, char_tone_rule, dialect,
                    index=index, text=text
                )

            fout.write(f"{index}\t{text}\t{' '.join(pinyin_units)}\n")

def qingdao_tone2_change(pinyin_units):
    """
    If a character is tone 2, and the following character is not tone 5 (or doesn't exist), 
    change that tone 2 to tone 4.
    """
    result = pinyin_units.copy()
    for i in range(len(result)):
        base, tone = parse_tone(result[i])
        if tone != 2:
            continue

        # Get tone of next character (record as -1 if doesn't exist)
        next_tone = -1
        if i + 1 < len(result):
            next_tone = parse_tone(result[i + 1])[1]

        # Apply tone sandhi if not tone 5
        if next_tone != 5:
            wrapped = result[i].startswith("#") and result[i].endswith("#")
            new_pinyin = f"#{base}4#" if wrapped else f"{base}4"
            print(f"[Qingdao 2→4 tone sandhi] {result[i]} → {new_pinyin}")
            result[i] = new_pinyin
    return result

def taibei_group_nonfinal_tone_change(pinyin_units, text_units):
    assert len(pinyin_units) == len(text_units)
    separators = {'’', '：', '‘', '“', '；', '。', '|', '』', '…', '─', '！', '『', '、', '「', '」', '”', '？', '，'}


    result = pinyin_units.copy()

    def get_group_indices():
        """Return start and end positions of all character groups"""
        groups = []
        start = 0
        for i, char in enumerate(text_units):
            if any(sep in char for sep in separators):
                if start < i:
                    groups.append((start, i))
                start = i + 1
        if start < len(text_units):
            groups.append((start, len(text_units)))
        return groups

    # Define tone sandhi rules
    base_tone_map = {
        1: 7,
        5: 3,
        2: 1,
        3: 2,
        7: 3,
        8: 3,
    }

    def get_new_tone(base, tone):
        if tone == 4:
            if base.endswith("h"):
                return 2
            else:
                return 8
        return base_tone_map.get(tone, tone)

    for start, end in get_group_indices():
        if end - start <= 1:
            continue  # Skip single character groups
        for i in range(start, end - 1):  # Non-final characters
            orig = result[i]
            base, tone = parse_tone(orig)
            if tone == -1:
                continue
            new_tone = get_new_tone(base, tone)
            if orig.startswith("#") and orig.endswith("#"):
                result[i] = f"#{base}{new_tone}#"
            else:
                result[i] = f"{base}{new_tone}"
            print(f"[Taipei tone sandhi] {text_units[i]}({orig}) → {result[i]}")

    return result

def shanghai_group_tone_change(pinyin_units, text_units):
    assert len(pinyin_units) == len(text_units)
    separators = {'’', '：', '‘', '“', '；', '。', '|', '』', '…', '─', '！', '『', '、', '「', '」', '”', '？', '，'}

    result = pinyin_units.copy()

    def get_group_indices():
        groups = []
        start = 0
        for i, char in enumerate(text_units):
            if any(sep in char for sep in separators):
                if start < i:
                    groups.append((start, i))
                start = i + 1
        if start < len(text_units):
            groups.append((start, len(text_units)))
        return groups

    # ① First tone sandhi rules (regardless of h ending)
    tone_map_nonh = {
        1: [11, 10],
        2: [7, 11, 10],
        3: [8, 11, 10],
        6: [3, 11],
        7: [2, 10],
        8: [3, 10],
        9: [9],
        10: [10],
        11: [1, 10],
    }

    tone_map_h = {
        4: [13, 4, 10],
        5: [12, 5],   # Special case
        12: [12, 5, 10],
    }

    # ② Second tone sandhi rules (apply again under specific conditions after first sandhi)
    second_map_for_h = {11: 4, 7: 13, 8: 12, 3: 5}
    second_map_for_nonh = {4: 11, 13: 7, 12: 8, 5: 3}


    def format_unit(base, tone, wrapped):
        return f"#{base}{tone}#" if wrapped else f"{base}{tone}"

    for start, end in get_group_indices():
        if end - start <= 1:
            continue  # Skip single character groups

        first_base, first_tone = parse_tone(result[start])

        rule_group = None
        fill_tones = None

        if first_tone in tone_map_nonh:
            fill_tones = tone_map_nonh[first_tone]
            rule_group = "nonh"
        elif first_tone in tone_map_h:
            fill_tones = tone_map_h[first_tone]
            rule_group = "h"
        else:
            continue

        fill_len = len(fill_tones)
        for i in range(start, end):
            orig = result[i]
            base, orig_tone = parse_tone(orig)
            wrapped = orig.startswith("#") and orig.endswith("#")
            # Get current tone sandhi value (repeat last one if overflow)
            if first_tone == 5:
                if i<end-1:
                    new_tone = fill_tones[0]
                else:
                    new_tone = fill_tones[1]
            else:
                new_tone = fill_tones[min(i - start, fill_len - 1)]

            # Second tone sandhi (check conditions)
            if rule_group == "nonh" and base.endswith("h"):
                new_tone = second_map_for_h.get(new_tone, new_tone)
            elif rule_group == "h" and not base.endswith("h"):
                new_tone = second_map_for_nonh.get(new_tone, new_tone)

            result[i] = format_unit(base, new_tone, wrapped)
            print(f"[Shanghai tone sandhi] {text_units[i]}({orig}) → {result[i]}")

    return result


# Dialect rules
TONE_RULES = {
    "xian": {
        (1, 1): [2, 1],
    },
    "shijiazhuang": {
        (1, 1): [3, 1],
        (2, 1): [3, 1],
        (2, 4): [3, 4],
    },
    "zhengzhou": {
        (2, 4): [3, 4],
        (3, 3): [2, 3],
        (4, 4): [1, 4],
    },
    "putonghua": {
        (3, 3): [2, 3],
    },
    "nanjing": {
        (1, 1): [4, 1],
        (3, 1): [2, 1],
        (2, 5): [3, 5],
        (3, 3): [2, 3],
        (4, 5): [1, 5],
        (5, 5): [7, 5],
    },
    "qingdao": {
        (1, 1): [3, 1],
        (2, 1): [3, 1],
        (4, 1): [3, 1],
        (1, 2): [3, 1],
        (2, 2): [6, 4],
        (3, 2): [3, 1],
        (4, 2): [6, 4],
        (1, 3): [6, 3],
        (2, 3): [6, 3],
        (3, 3): [6, 3],
        (4, 3): [6, 3],
        (1, 4): [3, 1],
        (2, 4): [6, 4],
        (3, 4): [3, 1],
        (4, 4): [6, 4],
        (3, 5): [1, 5],
        (2, 5): [2, 5],
    },
}
CHAR_TONE_RULES = {
    'shijiazhuang': {
    "一": {
        (4,): 3,
        "default": 1
    },
    "不": {
        (4,): 3,
        "default": 1
    }
    },
    'zhengzhou': {
    "一": {
        (4,): 3,
        (5,): 2,
        "default": 1
    },
    "七": {
        (4,): 3,
        (5,): 2,
        "default": 1
    },
    "八": {
        (4,): 3,
        (5,): 2,
        "default": 1
    },
    "不": {
        (4,): 3,
        (5,): 2,
        "default": 1
    },
    },
'putonghua': {
    "一": {
        (1,): 4,
        (2,): 4,
        (3,): 4,
        (4,): 2,
        "default": 1
    },
    "不": {
        (4,): 2,
        "default": 4
    }
    },
}


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
    # input_path = f"./output/{dialect}_word.txt"
    # output_path = f"./output/{dialect}_liandutone.txt"

    process_tone_script_with_flexible_rules(input_path, output_path, dialect, TONE_RULES, CHAR_TONE_RULES)



