zh_phonemes = [
    "<pad>",
    "<unk>",
    "a",
    "ai",
    "air",
    "an",
    "ang",
    "angr",
    "anr",
    "ao",
    "aor",
    "ar",
    "b",
    "c",
    "ch",
    "d",
    "e",
    "ei",
    "eir",
    "en",
    "eng",
    "engr",
    "enr",
    "er",
    "err",
    "f",
    "g",
    "h",
    "i",
    "ia",
    "ian",
    "iang",
    "iangr",
    "ianr",
    "iao",
    "iaor",
    "iar",
    "ie",
    "ier",
    "ii",
    "iii",
    "iiir",
    "iir",
    "in",
    "ing",
    "ingr",
    "inr",
    "io",
    "iong",
    "iongr",
    "iou",
    "iour",
    "ir",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "or",
    "ong",
    "ongr",
    "ou",
    "our",
    "p",
    "q",
    "r",
    "s",
    "sh",
    "sil",
    "sp",
    "spl",
    "spn",
    "t",
    "u",
    "ua",
    "uar",
    "uai",
    "uair",
    "uan",
    "uanr",
    "uang",
    "uangr",
    "uei",
    "ueir",
    "uen",
    "ueng",
    "uenr",
    "uo",
    "uor",
    "ur",
    "v",
    "vr",
    "van",
    "vanr",
    "ve",
    "ver",
    "vn",
    "vnr",
    "x",
    "z",
    "zh",
    "，",
    "。",
    "？",
    "！",
    # "<eos>",
    "iai",
]

en_phonemes = ["<pad>", "<unk>", "<s>", "</s>"] + [
    "AA0",
    "AA1",
    "AA2",
    "AE0",
    "AE1",
    "AE2",
    "AH0",
    "AH1",
    "AH2",
    "AO0",
    "AO1",
    "AO2",
    "AW0",
    "AW1",
    "AW2",
    "AY0",
    "AY1",
    "AY2",
    "B",
    "CH",
    "D",
    "DH",
    "EH0",
    "EH1",
    "EH2",
    "ER0",
    "ER1",
    "ER2",
    "EY0",
    "EY1",
    "EY2",
    "F",
    "G",
    "HH",
    "IH0",
    "IH1",
    "IH2",
    "IY0",
    "IY1",
    "IY2",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW0",
    "OW1",
    "OW2",
    "OY0",
    "OY1",
    "OY2",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH0",
    "UH1",
    "UH2",
    "UW",
    "UW0",
    "UW1",
    "UW2",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]

en_phonemes_wo_tone = ["<pad>", "<unk>", "<s>", "</s>"] + [
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]

from pathlib import Path
HERE = Path(__file__).resolve().parent

mms = {}
with open(HERE / "modal.txt", "r", encoding="UTF8") as f:
    for idx, cc in enumerate(f.readlines()):
        mm = cc.split(":")[0].strip()
        mms[mm] = f"MM{idx+1}"


def get_new(s2p_file, tones = ["1", "2", "3", "4", "5"]):
    new_pinyin2phone = {}
    new_phone = []
    new_initial = []
    with open(s2p_file, "r") as f:
        for line in f.readlines():
            ppinyin, first_phone, second_phone = line.strip().split("|")
            if first_phone == "" or first_phone == " ":
                phone_str = " " + second_phone
            else:
                phone_str = first_phone + " " + second_phone
            for tone in tones:
                new_pinyin2phone[ppinyin+tone] = phone_str+tone
                
            if first_phone != "" and first_phone != " " and first_phone not in zh_phonemes and first_phone not in new_phone:
                new_phone.append(first_phone)
                new_initial.append(first_phone)
            if second_phone not in zh_phonemes and second_phone not in new_phone:
                new_phone.append(second_phone)
            
    
    return new_pinyin2phone, new_phone, new_initial


import os
lang_s2p = {}
s2p_root_dir = "frontend/dialect"
for subdir, _, files in os.walk(s2p_root_dir):
    if 's2p.txt' in files:
        s2p_file = os.path.join(subdir, 's2p.txt')
        lang_name = os.path.basename(subdir)
        new_pinyin2phone, new_phone, new_initial = get_new(s2p_file)
        lang_s2p[lang_name] = (new_pinyin2phone, new_phone, new_initial)
        
phonemes = (
        ["_", "!", "?", "…", ",", ".", "'", "-",] + zh_phonemes + en_phonemes_wo_tone + ["MM0"] + list(mms.values())
    )


# new_punc = ["、", "："]
# new_num = len(hn_new_phone) + len(zh_new_phone) + len(new_punc)
# if new_num > len(en_phonemes_wo_tone):
#     print(f"error in load phonemes!!! {new_num} more than en_phone: {len(en_phonemes_wo_tone)}, use raw phone!!!")
#     phonemes = (
#         ["_", "!", "?", "…", ",", ".", "'", "-"] + zh_phonemes + en_phonemes_wo_tone + ["MM0"] + list(mms.values())
#     )
# else:
#     # 这里新增的 phone 占用 english 的phone
#     phonemes = (
#         ["_", "!", "?", "…", ",", ".", "'", "-"] + zh_phonemes + zh_new_phone + hn_new_phone + ["、", "："] +  en_phonemes_wo_tone[new_num:] + ["MM0"] + list(mms.values())
#     )
#     print("Load phonemes success!")


num_zh_tones = 7
num_en_tones = 4


language_tone_start_map = {"ZH": 0, "EN": num_zh_tones}
language_id_map = {"ZH": 0, "EN": 1}

num_tones = num_zh_tones + num_en_tones
num_languages = len(language_id_map.keys())



# if __name__ == "__main__":
#     print(phonemes)
