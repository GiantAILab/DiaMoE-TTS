import json
import os
import pdb
import re
from importlib.resources import files
from pathlib import Path
from datasets import Dataset as Dataset_
import random
import soundfile as sf
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm


import json
import pickle
from pathlib import Path

import logging
from collections import Counter
from datetime import datetime

date_str = datetime.now().strftime("%Y-%m-%d")

def check_duplicate_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]

    from collections import Counter
    word_counts = Counter(words)
    duplicates = [word for word, count in word_counts.items() if count > 1]

    if duplicates:
        print(f"found {len(duplicates)} repeats：")
        for word in duplicates:
            print(f"{word}（get {word_counts[word]} times）")
    else:
        print("no repeat in vocab。")



def load_ipa_list(file_path):
    """load IPA list"""
    print(f"load IPA list from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        tem_list = file.read().splitlines()
    result_list = []
    for i in tem_list:
        if '错' not in i and '上' not in i and '改' not in i and i[0] == '[' and i[-1] == ']' :
            result_list.append(i)
    return result_list


def load_punctuation_list(file_path):
    """load punctuation list"""
    print(f"load punctuation list: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().splitlines()


def replace_special_chars(text):
    """process special chars"""
    text = text.strip()
    replacements = {
        "„": "\"",
        "“": "\"",
        "”": "\"",
        "‘": "\'",
        "’": "\'",
        "\"": " \" ",
        "\'": " \' ",
        "}": "}",
        "；": ";",
        "：": ":",
        "（": "(",
        "）": ")",
        "『": "\"",
        "』": "\"",
        "~": " ~ ",
        "|": " | ",
        "...": "…",
        "..": "…",
        "…": " … ",
        "!?": "?",
        "?!": "?",
        "??": "?",
        "!": " ! ",
        "?": " ? ",
        "¿": " ¿ ",
        "¡": " ¡ ",
        ".": " . ",
        ",": " , ",
        ";": " ; ",
        ":": " : ",
        "«": " « ",
        "»": " » ",
        "-": " - ",
        "–": " – ",
        "—": " — ",
        "(": " ( ",
        ")": " ) ",
        "[": " [ ",
        "]": " ] ",
        "{": " { ",
        "€": " € ",
        "§": " § ",
        "。": " 。 ",
        "，": " ， ",
        "、": " 、 ",
        "！": " ！ ",
        "？": " ？ ",
        "·": " · ",
        "《": " 《 ",
        "》": " 》 ",
        "「": " 「 ",
        "」": " 」 ",
        "#": "",
        "﹐": " , ",
    }
    # List of Special Characters Reference
    # [ !?¿¡.,;:«»„""''\'\"\-–—()\[\]{}…€§。，、！？；：·（）《》『』「」~]
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.strip()
    text = re.sub(r' +', ' ', text)
    text = text.strip().split(" ")
    return text


def create_or_load_audio_path_index(base_dir, cache_path="audio_path_map.pkl"):
    """Create or load audio file index"""
    # check if cache exist
    if os.path.exists(f"{base_dir.split('/')[-1]}_audio_path_map.pkl"):
        print(
            f"Loading audio file index from cache: {base_dir.split('/')[-1]}_audio_path_map.pkl")
        with open(f"{base_dir.split('/')[-1]}_audio_path_map.pkl", 'rb') as f:
            return pickle.load(f)

    print("Creating audio file index...")
    audio_path_map = {}

    # Traverse all subfolders
    for root, _, files in tqdm(list(os.walk(base_dir))):
        for file in files:
            if file.endswith('.mp3'):
                audio_path_map[file] = os.path.join(root, file)

    # Save the index to a file
    cache_path = f"{base_dir.split('/')[-1]}_audio_path_map.pkl"
    print(f"Save the index to: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(audio_path_map, f)
    return audio_path_map

# cache for duration
def load_duration_cache(cache_dir, dialect):
    """
    Load the duration cache file for the specified dialect and return a dictionary.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{dialect}_duration_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {}

def query_duration_with_cache(audio_path, duration_cache):
    """
    Given the audio path and the existing cache dict, if the cache exists, return it; otherwise, query and add it to the dict.
    """
    if audio_path in duration_cache:
        return float(duration_cache[audio_path])
    try:
        duration = sf.info(audio_path).duration
    except RuntimeError:
        raise ValueError(f"cant load audio: {audio_path}")
    duration_cache[audio_path] = duration
    return duration

def save_duration_cache(duration_cache, cache_dir, dialect):
    """
    Save the updated "duration_cache" to the specified file.
    """
    cache_path = os.path.join(cache_dir, f"{dialect}_duration_cache.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(duration_cache, f, ensure_ascii=False, indent=2)

def save_ipa_counter(ipa_counter, save_dir, file_prefix="ipa_counts"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{file_prefix}.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        for symbol, count in ipa_counter.most_common():
            f.write(f"{symbol}: {count}\n")

def process_line(mp3_id, ipa_text, ori_text, mp3_base_path, ipalist, audio_path_map, punctuation_list, duration_cache, ipa_counter):
    if mp3_id.endswith(".mp3"):
        mp3_id = mp3_id.split(".")[0]
    target_text = ipa_text.split(" ")

    # process IPA tokens

    ipa_text = []
    for it in target_text:

        symbol = '[' + it + ']'
        if symbol in ipalist:
            ipa_text.append(symbol)
            ipa_counter[symbol] += 1
        elif symbol in punctuation_list:
            ipa_text.append(it)
        else:
            if symbol != '[]' and symbol != '[|]':
                error_counter[symbol] += 1

                print('warnning', symbol)
            continue

    # check audio files
    mp3_path = resolve_audio_path(mp3_id, mp3_base_path, audio_path_map)
    if not mp3_path:
        return None

    duration = query_duration_with_cache(mp3_path, duration_cache)
    return {
        "audio_path": str(mp3_path),
        "ori_text": ori_text,
        "text": " ".join(ipa_text),
        "duration": duration
    }

def resolve_audio_path(mp3_id, mp3_base_path, audio_path_map=None):
    # from cache
    if audio_path_map:
        mapped_path = audio_path_map.get(f"{mp3_id}.mp3")
        if mapped_path and os.path.exists(mapped_path):
            return mapped_path

    candidates = [os.path.join(mp3_base_path, f"{mp3_id}.{suf}") for suf in ['mp3','MP3','WAV','wav']]
    for path in candidates:
        if os.path.exists(path):
            return path

    print(f"[Audio missing] Audio file not found: {mp3_id}")
    return None

def process_dataset(file_path, mp3_base_path, ipalist, punctuation_list, dataset_name,
                    shuffle=False, max_lines=None):

    """process dataset"""
    all_data = []

    audio_path_map = None
    if dataset_name == 'zh':
        audio_path_map = create_or_load_audio_path_index(mp3_base_path)

        print(len(audio_path_map))

    # load duration cache
    d_cache_path = './duration_cache'
    duration_cache = load_duration_cache(d_cache_path, dataset_name)


    # Phoneme statistics
    ipa_counter = Counter()

    # load all lines
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file if line.strip()]

    # shuffle
    if shuffle:
        print(f'add shuffle to {dataset_name}')
        random.shuffle(lines)

    # Use only the first max_lines
    if max_lines is not None:
        print(f'use {dataset_name} only first {max_lines}lines')
        lines = lines[:max_lines]

    for line in tqdm(lines, desc=f"Processing {Path(file_path).name}", total=len(lines)):
        try:
            mp3_id, ori_text, ipa_text = line.split("\t")
        except ValueError:
            continue  # skip wrong type

        # Check if the audio exists in the index
        if audio_path_map and f"{mp3_id}.mp3" not in audio_path_map:
            continue

        result = process_line(mp3_id, ipa_text, ori_text, mp3_base_path,
                              ipalist, audio_path_map, punctuation_list, duration_cache, ipa_counter)

        if result:
            result['dialect'] = dataset_name
            all_data.append(result)

    # save duration cache
    save_duration_cache(duration_cache, d_cache_path, dataset_name)
    # Save phoneme statistics
    save_ipa_counter(ipa_counter, f'{save_dir}/{exp_name}', file_prefix=f"ipa_counts_{exp_name}_{dataset_name}")

    return all_data


def save_dataset(data, duration_list, save_path, file_prefix):
    """save dataset"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save arrow file
    arrow_path = os.path.join(save_path, f"{file_prefix}.arrow")
    with ArrowWriter(path=arrow_path) as writer:
        for line in tqdm(data, desc=f"Writing to {file_prefix}.arrow"):
            writer.write(line)
        num_examples, num_bytes = writer.finalize()
        print(f"success write: {num_examples} samples, File size {num_bytes} Bytes")

    # save duration file
    duration_path = os.path.join(save_path, f"{file_prefix}_duration.json")
    with open(duration_path, "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)


if __name__ == "__main__":
    # configs
    dataset_name = "train1"
    exp_name = "exp1"
    vocab_path = "path/to/diamoettsv1/diamoe_tts/data/vocab.txt"

    save_dir = str(files("f5_tts").joinpath("../../")) + f"/data/{dataset_name}"
    check_duplicate_words(vocab_path)  # Word List Recheck
    ipalist = load_ipa_list(vocab_path)
    punctuation_list = load_punctuation_list("path/to/diamoettsv1/diamoe_tts/data/punctuation.txt")


    print(f"\nPrepare for {dataset_name}, will save to {save_dir}\n")

    if not os.path.exists(f"{save_dir}/{exp_name}"):
        os.makedirs(f"{save_dir}/{exp_name}")

    # Construct the log file name with the date included
    log_filename = f"{save_dir}/{exp_name}/error_log_vocab_{exp_name}.log"
    logging.basicConfig(
        filename=log_filename,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # global error counter
    error_counter = Counter()


    data1 = process_dataset(
         "path/to/ipa_dataset.txt",
         "root/path/of/audios",
         ipalist,
         punctuation_list,
         'dialect_type',
    )


    if error_counter:
        logging.info("========= errors =========")
        for err_type, count in error_counter.items():
            logging.info(f"{err_type}: {count} times")

    
    train_data = data1

    # prepare data
    train_duration_list = [item["duration"] for item in train_data]
    # save dataset
    save_dataset(train_data, train_duration_list, save_dir, exp_name)

    print(f"\nFor exp {exp_name} of {dataset_name}:")
    print(f"Training samples: {len(train_data)}")
    print(f"Total duration: {sum(train_duration_list) / 3600:.2f} hours")