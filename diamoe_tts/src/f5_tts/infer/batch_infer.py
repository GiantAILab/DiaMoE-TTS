import argparse
import codecs
import os
import sys
import pdb
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path
import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    mel_spec_type,
    target_rms,
    cross_fade_duration,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    speed,
    fix_duration,
    device,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)


parser = argparse.ArgumentParser(
    prog="python3 infer-cli.py",
    description="Commandline interface for E2/F5 TTS with Advanced Batch Processing.",
    epilog="Specify options above to override one or more settings from config.",
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default=os.path.join(files("f5_tts").joinpath("infer/example"), "basic.toml"),
    help="The configuration file, default see infer/examples/basic/basic.toml",
)


# Note. Not to provide default value here in order to read default from config file

parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="The model name: F5TTS_v1_Base | F5TTS_Base | E2TTS_Base | etc.",
)
parser.add_argument(
    "-mc",
    "--model_cfg",
    type=str,
    help="The path to F5-TTS model config file .yaml",
)
parser.add_argument(
    "-p",
    "--ckpt_file",
    type=str,
    help="The path to model checkpoint .pt, leave blank to use default",
)
parser.add_argument(
    "-v",
    "--vocab_file",
    type=str,
    help="The path to vocab file .txt, leave blank to use default",
)
parser.add_argument(
    "-r",
    "--ref_audio",
    type=str,
    help="The reference audio file.",
)
parser.add_argument(
    "-s",
    "--ref_text",
    type=str,
    help="The transcript/subtitle for the reference audio",
)
parser.add_argument(
    "-t",
    "--gen_text",
    type=str,
    help="The text to make model synthesize a speech",
)
parser.add_argument(
    "-f",
    "--gen_file",
    type=str,
    help="The file with text to generate, will ignore --gen_text",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="The path to output folder",
)
parser.add_argument(
    "-w",
    "--output_file",
    type=str,
    help="The name of output file",
)
parser.add_argument(
    "--save_chunk",
    action="store_true",
    help="To save each audio chunks during inference",
)
parser.add_argument(
    "--remove_silence",
    action="store_true",
    help="To remove long silence found in ouput",
)
parser.add_argument(
    "--load_vocoder_from_local",
    action="store_true",
    help="To load vocoder from local dir, default to ../checkpoints/vocos-mel-24khz",
)
parser.add_argument(
    "--vocoder_name",
    type=str,
    choices=["vocos", "bigvgan"],
    help=f"Used vocoder name: vocos | bigvgan, default {mel_spec_type}",
)
parser.add_argument(
    "--target_rms",
    type=float,
    help=f"Target output speech loudness normalization value, default {target_rms}",
)
parser.add_argument(
    "--cross_fade_duration",
    type=float,
    help=f"Duration of cross-fade between audio segments in seconds, default {cross_fade_duration}",
)
parser.add_argument(
    "--nfe_step",
    type=int,
    help=f"The number of function evaluation (denoising steps), default {nfe_step}",
)
parser.add_argument(
    "--cfg_strength",
    type=float,
    help=f"Classifier-free guidance strength, default {cfg_strength}",
)
parser.add_argument(
    "--sway_sampling_coef",
    type=float,
    help=f"Sway Sampling coefficient, default {sway_sampling_coef}",
)
parser.add_argument(
    "--speed",
    type=float,
    help=f"The speed of the generated audio, default {speed}",
)
parser.add_argument(
    "--fix_duration",
    type=float,
    help=f"Fix the total duration (ref and gen audios) in seconds, default {fix_duration}",
)
parser.add_argument(
    "--device",
    type=str,
    help="Specify the device to run on",
)
parser.add_argument(
    "--use_ema",
    type=bool,
    help=f"whether use ema for inference",
)
parser.add_argument(
    "--use_moe",
    type=str,
    help=f"whether use moe for inference",
)
parser.add_argument(
    "--num_exps",
    type=int,
    help=f"number of experts for MoE",
)
parser.add_argument(
    "--moe_topK",
    type=int,
    help=f"using topK experts for MoE",
)
parser.add_argument(
    "--expert_type",
    type=str,
    help=f"using expert type of MoE",
)


args = parser.parse_args()


# config file

config = tomli.load(open(args.config, "rb"))


# command-line interface parameters

model = args.model or config.get("model", "F5TTS_v1_Base")
ckpt_file = args.ckpt_file or config.get("ckpt_file", "")
vocab_file = args.vocab_file or config.get("vocab_file", "")
use_ema = args.use_ema

# MoE
if args.use_moe.lower() == 'true':
    use_moe = True
else:
    use_moe = False
num_exps = args.num_exps
moe_topK = args.moe_topK
expert_type = args.expert_type.lower()

ref_audio = args.ref_audio or config.get("ref_audio", "infer/examples/basic/basic_ref_en.wav")
ref_audio = ref_audio.split(",")
ref_text = (
    args.ref_text
    if args.ref_text is not None
    else config.get("ref_text", "Some call me nature, others call me mother nature.")
)
ref_text_files = ref_text.split(",")
ref_text = []
for text_file in ref_text_files:
    with open(text_file, "r", encoding="utf-8") as f:
        ref_text.append(f.readlines()[0].strip())
gen_text = args.gen_text or config.get("gen_text", "Here we generate something just for test.")
gen_file = args.gen_file or config.get("gen_file", "")

output_dir = args.output_dir or config.get("output_dir", "tests")
output_dir = output_dir.split(",")
output_file = args.output_file or config.get(
    "output_file", f"infer_cli_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}.wav"
)

save_chunk = args.save_chunk or config.get("save_chunk", False)
remove_silence = args.remove_silence or config.get("remove_silence", False)
load_vocoder_from_local = args.load_vocoder_from_local or config.get("load_vocoder_from_local", False)

vocoder_name = args.vocoder_name or config.get("vocoder_name", mel_spec_type)
target_rms = args.target_rms or config.get("target_rms", target_rms)
cross_fade_duration = args.cross_fade_duration or config.get("cross_fade_duration", cross_fade_duration)
nfe_step = args.nfe_step or config.get("nfe_step", nfe_step)
cfg_strength = args.cfg_strength or config.get("cfg_strength", cfg_strength)
sway_sampling_coef = args.sway_sampling_coef or config.get("sway_sampling_coef", sway_sampling_coef)
speed = args.speed or config.get("speed", speed)
fix_duration = args.fix_duration or config.get("fix_duration", fix_duration)
device = args.device or config.get("device", device)


# patches for pip pkg user
if "infer/examples/" in ref_audio:
    ref_audio = str(files("f5_tts").joinpath(f"{ref_audio}"))
if "infer/examples/" in gen_file:
    gen_file = str(files("f5_tts").joinpath(f"{gen_file}"))
if "voices" in config:
    for voice in config["voices"]:
        voice_ref_audio = config["voices"][voice]["ref_audio"]
        if "infer/examples/" in voice_ref_audio:
            config["voices"][voice]["ref_audio"] = str(files("f5_tts").joinpath(f"{voice_ref_audio}"))


# ignore gen_text if gen_file provided
if gen_file:
    # 修改读取文件的方式，将文本分割成行
    with codecs.open(gen_file, "r", "utf-8") as f:
        gen_lines = [line.strip() for line in f.readlines() if line.strip()]
    gen_text = "\n".join(gen_lines)  # 保持原有的gen_text用于其他地方的兼容


# # output path

# wave_path = Path(output_dir) / output_file
# # spectrogram_path = Path(output_dir) / "infer_cli_out.png"
if save_chunk:
    output_chunk_dir = os.path.join(output_dir, f"{Path(output_file).stem}_chunks")
    if not os.path.exists(output_chunk_dir):
        os.makedirs(output_chunk_dir)


# load vocoder

if vocoder_name == "vocos":
    vocoder_local_path = "../checkpoints/vocos-mel-24khz"

elif vocoder_name == "bigvgan":
    vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"

vocoder = load_vocoder(
    vocoder_name=vocoder_name, is_local=load_vocoder_from_local, local_path=vocoder_local_path, device=device
)

# load TTS model

model_cfg = OmegaConf.load(
    args.model_cfg or config.get("model_cfg", str(files("f5_tts").joinpath(f"configs/{model}.yaml")))
)
model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
model_arc = model_cfg.model.arch


print(f"Using model loaded from {ckpt_file}...")
print(model_cls, model_arc, ckpt_file)
ema_model = load_model(
    model_cls, model_arc, ckpt_file, mel_spec_type=vocoder_name, vocab_file=vocab_file, device=device,
    use_moe=use_moe, num_exps = num_exps, moe_topK = moe_topK, expert_type=expert_type
)


# inference process


def main():
    for cur_ref_audio, cur_ref_text, cur_output_dir in zip(ref_audio, ref_text, output_dir):

        main_voice = {"ref_audio": cur_ref_audio, "ref_text": cur_ref_text}
        if "voices" not in config:
            voices = {"main": main_voice}
        else:
            voices = config["voices"]
            voices["main"] = main_voice
        for voice in voices:
            print("Voice:", voice)
            print("ref_audio ", voices[voice]["ref_audio"])
            voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
                voices[voice]["ref_audio"], voices[voice]["ref_text"]
            )
            print("ref_audio_", voices[voice]["ref_audio"], "\n\n")

        generated_audio_segments = []
            # 如果是从文件读取的多行文本，则逐行处理
        if gen_file:

            for i, line in enumerate(gen_lines):
                if not line.strip():
                    continue
                    
                print(f"\nProcessing line {i+1}: {line}")
                # 默认使用main voice处理每一行
                ref_audio_ = voices["main"]["ref_audio"]
                ref_text_ = voices["main"]["ref_text"]

                audio_segment, final_sample_rate, spectragram = infer_process(
                    ref_audio_,
                    ref_text_.split("\t")[2],
                    line.strip().split("\t")[2],
                    ema_model,
                    vocoder,
                    mel_spec_type=vocoder_name,
                    target_rms=target_rms,
                    cross_fade_duration=cross_fade_duration,
                    nfe_step=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                    speed=speed,
                    fix_duration=fix_duration,
                    device=device,
                )
                generated_audio_segments.append(audio_segment)

                savefile_name = line.strip().split("\t")[0].replace("|", "").strip()
                write_path = os.path.join(cur_output_dir, f"{i+1}_{savefile_name}.wav")
                if not os.path.exists(cur_output_dir):
                    os.makedirs(cur_output_dir)
                print(write_path)
                sf.write(
                    write_path,
                    audio_segment,
                    final_sample_rate,
                )
        else:
            reg1 = r"(?=\[\w+\])"
            chunks = re.split(reg1, gen_text)
            reg2 = r"\[(\w+)\]"
            for text in chunks:
                if not text.strip():
                    continue
                match = re.match(reg2, text)
                if match:
                    voice = match[1]
                else:
                    print("No voice tag found, using main.")
                    voice = "main"
                if voice not in voices:
                    print(f"Voice {voice} not found, using main.")
                    voice = "main"
                text = re.sub(reg2, "", text)
                ref_audio_ = voices[voice]["ref_audio"]
                ref_text_ = voices[voice]["ref_text"]
                #print("ref text",ref_text_)
                gen_text_ = text.strip()
                print(f"Voice: {voice}")
                audio_segment, final_sample_rate, spectragram = infer_process(
                    ref_audio_,
                    ref_text_,
                    gen_text_,
                    ema_model,
                    vocoder,
                    mel_spec_type=vocoder_name,
                    target_rms=target_rms,
                    cross_fade_duration=cross_fade_duration,
                    nfe_step=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                    speed=speed,
                    fix_duration=fix_duration,
                    device=device,
                )
                generated_audio_segments.append(audio_segment)

                if save_chunk:
                    if len(gen_text_) > 200:
                        gen_text_ = gen_text_[:200] + " ... "
                    sf.write(
                        os.path.join(output_chunk_dir, f"{len(generated_audio_segments) - 1}_{gen_text_}.wav"),
                        audio_segment,
                        final_sample_rate,
                    )

            if generated_audio_segments:
                final_wave = np.concatenate(generated_audio_segments)

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                with open(wave_path, "wb") as f:
                    sf.write(f.name, final_wave, final_sample_rate)
                    # Remove silence
                    if remove_silence:
                        remove_silence_for_generated_wav(f.name)
                    print(f.name)


if __name__ == "__main__":
    main()
