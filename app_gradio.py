#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dialect TTS Zero Shot Inference Gradio Interface
Integrating text frontend processing and TTS model inference
"""

import os
import sys
import tempfile
import shutil
import subprocess
import argparse
import codecs
from pathlib import Path
from datetime import datetime
import numpy as np
import soundfile as sf
import gradio as gr
import torch
from omegaconf import OmegaConf
from hydra.utils import get_class

# 添加路径
sys.path.insert(0, "./dialect_frontend")
sys.path.insert(0, "./diamoe_tts/src")

# 从dialect_frontend导入必要的模块
from tools.mix_wrapper import Preprocessor

# 从TTS模块导入
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

# 模型配置参数 - 在这里直接指定
MODEL_CONFIG = {
    "model_name": "gradio",
    "ckpt_file": "/path/to/your/model.pt",  # 请修改为你的模型路径
    "vocab_file": "./diamoe_tts/data/vocab.txt",
    "use_moe": True,
    "num_exps": 9,
    "moe_topK": 1,
    "expert_type": "mlp"
}
 
class DialectTTSPipeline:
    """方言TTS处理管道"""
    
    def __init__(self, auto_load_model=True):
        self.dialect_list = [
            "putonghua", "chengdu", "gaoxiong", "shanghai",
            "shijiazhuang", "wuhan", "xian", "zhengzhou"
        ]
        # 初始化前端处理器
        self.preprocessor = Preprocessor()
        self.frontend = self.preprocessor.frontend['ZH']
        print("Frontend processor initialized!")
        
        # 加载IPA音素列表和标点符号
        self.load_vocab_and_punctuation()
        
        # 模型相关变量
        self.model = None
        self.vocoder = None
        self.model_loaded = False
        
        # 自动加载模型
        if auto_load_model:
            self.load_tts_model(
                model_name=MODEL_CONFIG["model_name"],
                ckpt_file=MODEL_CONFIG["ckpt_file"],
                vocab_file=MODEL_CONFIG["vocab_file"],
                use_moe=MODEL_CONFIG["use_moe"],
                num_exps=MODEL_CONFIG["num_exps"],
                moe_topK=MODEL_CONFIG["moe_topK"],
                expert_type=MODEL_CONFIG["expert_type"]
            )
          
    def load_vocab_and_punctuation(self):
        """加载IPA音素列表和标点符号"""
        try:
            # 加载词汇表
            with open(MODEL_CONFIG["vocab_file"], 'r', encoding='utf-8') as f:
                vocab_lines = [line.strip() for line in f if line.strip()]
            
            # 过滤出用[]包装的音素
            self.ipa_list = []
            for line in vocab_lines:
                if line.startswith('[') and line.endswith(']'):
                    self.ipa_list.append(line)
            
            # 加载标点符号
            punctuation_path = "./diamoe_tts/data/punctuation.txt"
            if os.path.exists(punctuation_path):
                with open(punctuation_path, 'r', encoding='utf-8') as f:
                    self.punctuation_list = [line.strip() for line in f if line.strip()]
            else:
                self.punctuation_list = []
            
            print(f"Loaded {len(self.ipa_list)} IPA phonemes and {len(self.punctuation_list)} punctuation marks")
            
        except Exception as e:
            print(f"Failed to load vocabulary: {e}")
            self.ipa_list = []
            self.punctuation_list = []
    
    def convert_to_ipa_format(self, text: str) -> str:
        """将前端处理的文本转换为IPA格式"""
        if not text or not text.strip():
            return text
        
        # 参考prepare_ipa.py的实现
        # 用空格分割得到token列表
        target_text = text.split(" ")
        
        ipa_text = []
        for it in target_text:
            # 跳过空token
            if not it.strip():
                continue
                
            it = it.strip()
            symbol = '[' + it + ']'
            
            if symbol in self.ipa_list:
                # 如果是有效的IPA音素，添加[token]格式
                ipa_text.append(symbol)
            elif symbol in self.punctuation_list:
                # 如果是标点符号，添加原始token（不加中括号）
                ipa_text.append(it)
            else:
                # 跳过空符号和|符号
                if symbol != '[]' and symbol != '[|]':
                    print(f'Warning: Unknown symbol {symbol}')
                # 跳过未知符号
                continue
        
        result = ' '.join(ipa_text)
        print(f"IPA format conversion: {text[:50]}... -> {result[:50]}...")
        return result
    
    def replace_english_punctuation_with_chinese(self, text: str) -> str:
        """将英文标点转换为中文标点"""
        en_to_zh_punct = {",": "，", ".": "。", "?": "？", "!": "！", ":": "：", ";": "；",
            "(": "（", ")": "）", "[": "【", "]": "】", "{": "｛", "}": "｝",
            "<": "《", ">": "》", '\"': '"', "'": "'", "-": "－", "_": "＿",
            "&": "＆", "@": "＠", "/": "／", "\\": "、", "|": "｜",
            "`": "｀", "~": "～", "^": "＾"}
        
        
        for en, zh in en_to_zh_punct.items():
            text = text.replace(en, zh)
        return text
    
    def process_text_to_pinyin(self, text: str) -> tuple:
        """处理文本到拼音的转换，直接使用已加载的前端处理器"""
        try:
            # 使用前端处理器转换文本到拼音
            phones_list, word2ph, tones_list, ppinyins, oop, zhongwens = self.frontend.get_splited_phonemes_tones([text])
            
            # 标点符号转换
            zhongwens = self.replace_english_punctuation_with_chinese(zhongwens)
            ppinyins = self.replace_english_punctuation_with_chinese(ppinyins)
            
            print(f"Text to pinyin: {text} -> {ppinyins}")
            return zhongwens, ppinyins, oop
        except Exception as e:
            print(f"Pinyin conversion error: {e}")
            return text, text, []
    
    def run_shell_command(self, command: str, cwd: str = None) -> tuple:
        """运行shell命令"""
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                cwd=cwd,
                encoding='utf-8'
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            return 1, "", str(e)
    
    def create_temp_file(self, content: str, suffix: str = ".txt") -> str:
        """创建临时文件"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False, encoding='utf-8')
        temp_file.write(content)
        temp_file.close()
        return temp_file.name
    
    def process_frontend_pipeline(self, text: str, dialect: str) -> str:
        """完整的前端处理管道"""
        print(f"Starting dialect frontend processing: {dialect}")
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        try:
            # 步骤1: 直接使用已加载的前端处理器生成拼音
            print("Executing pinyin conversion...")
            zhongwens, ppinyins, oop = self.process_text_to_pinyin(text)
            
            # 步骤2: 创建拼音文件供后续处理
            pinyin_file = os.path.join(temp_dir, "pinyin_output.txt")
            with open(pinyin_file, 'w', encoding='utf-8') as f:
                f.write(f"temp_id\t{zhongwens}\t{ppinyins}\n")
             
            # 步骤3: 运行方言前端处理脚本
            if dialect == "putonghua":
                # 普通话直接使用拼音结果
                final_output = pinyin_file
            else:
                # 运行single_frontend.sh脚本，使用绝对路径
                current_dir = os.getcwd()
                dialect_dir = os.path.join(current_dir, "dialect_frontend")
                script_path = os.path.join(dialect_dir, "single_frontend.sh")
                
                # 检查脚本文件是否存在
                if not os.path.exists(script_path):
                    print(f"Script file does not exist: {script_path}")
                    print(f"Current working directory: {current_dir}")
                    print(f"Trying to list dialect_frontend directory contents:")
                    try:
                        files = os.listdir(dialect_dir) if os.path.exists(dialect_dir) else ["Directory does not exist"]
                        print(files)
                    except:
                        print("Cannot list directory contents")
                    final_output = pinyin_file
                else:
                    # 使用相对路径在dialect_frontend目录下执行
                    frontend_command = f'bash single_frontend.sh all {dialect} "{os.path.basename(pinyin_file)}"'
                    print(f"Executing frontend processing: {frontend_command}")
                    print(f"Working directory: {dialect_dir}")
                    
                    # 复制文件到dialect_frontend目录以便脚本处理
                    temp_pinyin_file = os.path.join(dialect_dir, os.path.basename(pinyin_file))
                    try:
                        shutil.copy2(pinyin_file, temp_pinyin_file)
                        print(f"File copied to: {temp_pinyin_file}")
                    except Exception as e:
                        print(f"File copy failed: {e}")
                    
                    # 设置环境变量
                    env = os.environ.copy()
                    env['LANG'] = 'C.UTF-8'
                    env['LC_ALL'] = 'C.UTF-8'
                    
                    try:
                        result = subprocess.run(
                            frontend_command,
                            shell=True,
                            capture_output=True,
                            text=True,
                            cwd=dialect_dir,  # 在dialect_frontend目录下执行
                            env=env,
                            encoding='utf-8'
                        )
                        ret_code, stdout, stderr = result.returncode, result.stdout, result.stderr
                    except Exception as e:
                        ret_code, stdout, stderr = 1, "", str(e)
                
                    if ret_code != 0:
                        print(f"Frontend processing failed: {stderr}")
                        print(f"Standard output: {stdout}")
                        # 使用原始拼音文件作为回退
                        final_output = pinyin_file
                    else:
                        # 查找最终的IPA格式文件（在dialect_frontend目录下）
                        base_name = os.path.splitext(temp_pinyin_file)[0]
                        ipa_file = base_name + "_ipa_format.txt"
                        if os.path.exists(ipa_file):
                            # 将结果文件复制回临时目录
                            final_output = os.path.join(temp_dir, "final_ipa_format.txt")
                            try:
                                shutil.copy2(ipa_file, final_output)
                                print(f"Result file copied to: {final_output}")
                            except Exception as e:
                                print(f"Result file copy failed: {e}")
                                final_output = pinyin_file
                        else:
                            print("IPA format file not found, using pinyin file")
                            final_output = pinyin_file
            
            # 读取最终结果
            if os.path.exists(final_output):
                with open(final_output, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        # 取最后一列作为处理后的音素
                        parts = lines[0].strip().split('\t')
                        if len(parts) >= 3:
                            return parts[-1]  # 最后一列是处理后的音素
                        else:
                            return parts[1] if len(parts) > 1 else text
                    else:
                        return text
            else:
                print(f"Output file does not exist: {final_output}")
                return text
                
        except Exception as e:
            print(f"Frontend processing error: {e}")
            return text
        finally:
            # 清理临时文件
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
    
    def load_tts_model(self, model_name: str = "test", 
                      ckpt_file: str = "", 
                      vocab_file: str = "./diamoe_tts/data/vocab.txt",
                      use_moe: bool = True,
                      num_exps: int = 9,
                      moe_topK: int = 1,
                      expert_type: str = "mlp"):
        """加载TTS模型"""
        try:
            print("Starting to load TTS model...")
            
            # 加载vocoder
            vocoder_name = mel_spec_type
            if vocoder_name == "vocos":
                vocoder_local_path = "./checkpoints/vocos-mel-24khz"
            elif vocoder_name == "bigvgan":
                vocoder_local_path = "./checkpoints/bigvgan_v2_24khz_100band_256x"
            else:
                vocoder_local_path = ""
            
            self.vocoder = load_vocoder(
                vocoder_name=vocoder_name, 
                is_local=False, 
                local_path=vocoder_local_path, 
                device=device
            )
            print("Vocoder loaded successfully!")
            
            # 加载TTS模型
            model_cfg_path = f"./diamoe_tts/src/f5_tts/configs/{model_name}.yaml"
            if not os.path.exists(model_cfg_path):
                model_cfg_path = "./diamoe_tts/src/f5_tts/configs/diamoetts.yaml"
            
            model_cfg = OmegaConf.load(model_cfg_path)
            model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
            model_arc = model_cfg.model.arch
            
            print(f"Using model config: {model_cfg_path}")
            print(f"Model class: {model_cls}, architecture: {model_arc}")
            
            if ckpt_file and os.path.exists(ckpt_file):
                print(f"Loading model from checkpoint: {ckpt_file}")
                self.model = load_model(
                    model_cls, model_arc, ckpt_file, 
                    mel_spec_type=vocoder_name, 
                    vocab_file=vocab_file, 
                    device=device,
                    use_moe=use_moe, 
                    num_exps=num_exps, 
                    moe_topK=moe_topK, 
                    expert_type=expert_type.lower()
                )
                self.model_loaded = True
                print("TTS model loaded successfully!")
            else:
                print("No valid model checkpoint file provided")
                self.model_loaded = False
                
        except Exception as e:
            print(f"Model loading failed: {e}")
            self.model_loaded = False
    
    def synthesize_speech(self, text: str, dialect: str, 
                         ref_audio_path: str, ref_text: str,
                         target_rms: float = 0.1,
                         cross_fade_duration: float = 0.15,
                         nfe_step: int = 32,
                         cfg_strength: float = 2.0,
                         sway_sampling_coef: float = -1.0,
                         speed: float = 1.0) -> tuple:
        """合成语音"""
        if not self.model_loaded:
            return None, "Model not loaded, please load model first"
        
        try:
            print(f"Starting speech synthesis: {dialect}")
            print(f"Input text: {text}")
            
            # 步骤1: 前端处理
            processed_text = self.process_frontend_pipeline(text, dialect)
            print(f"Frontend processing result: {processed_text}")
            
            # 步骤2: 预处理参考音频和文本
            ref_audio, ref_text_processed = preprocess_ref_audio_text(ref_audio_path, ref_text)
            print(f"Reference audio processing completed")
            
            # 步骤3: IPA格式转换
            # 对生成文本进行格式转换
            processed_text_ipa = self.convert_to_ipa_format(processed_text)
            
            # 对参考文本也进行前端处理和格式转换
            if ref_text_processed and ref_text_processed.strip():
                # 首先对参考文本进行前端处理
                ref_processed = self.process_frontend_pipeline(ref_text_processed, dialect)
                # 然后进行IPA格式转换
                ref_text_processed_ipa = self.convert_to_ipa_format(ref_processed)
            else:
                ref_text_processed_ipa = ref_text_processed
            
            print(f"IPA format conversion - Generated text: {processed_text_ipa}")
            print(f"IPA format conversion - Reference text: {ref_text_processed_ipa}")
            
            # 步骤4: 进行TTS推理
            audio_segment, final_sample_rate, spectrogram = infer_process(
                ref_audio,
                ref_text_processed_ipa,
                processed_text_ipa,
                self.model,
                self.vocoder,
                mel_spec_type=mel_spec_type,
                target_rms=target_rms,
                cross_fade_duration=cross_fade_duration,
                nfe_step=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                speed=speed,
                fix_duration=None,
                device=device,
            )
            
            # 保存音频
            output_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(output_file.name, audio_segment, final_sample_rate)
            
            return output_file.name, "Synthesis successful!"
            
        except Exception as e:
            print(f"Speech synthesis failed: {e}")
            return None, f"Synthesis failed: {str(e)}"
    
    def generate_frontend_only(self, text: str, dialect: str) -> tuple:
        """Generate frontend processing only without TTS"""
        try:
            print(f"Starting frontend-only processing: {dialect}")
            print(f"Input text: {text}")
            
            # 直接调用完整的前端处理管道，传入纯净的用户文本
            processed_text = self.process_frontend_pipeline(text, dialect)
            print(f"Frontend processing result: {processed_text}")
            
            # IPA格式转换
            processed_text_ipa = self.convert_to_ipa_format(processed_text)
            print(f"IPA format result: {processed_text_ipa}")
            
            return processed_text, processed_text_ipa, "Frontend processing successful!"
            
        except Exception as e:
            print(f"Frontend processing failed: {e}")
            return text, text, f"Frontend processing failed: {str(e)}"

# Global pipeline instance - preload model
print("Initializing Dialect TTS pipeline and preloading model...")
pipeline = DialectTTSPipeline(auto_load_model=True)

# Sample texts for different languages
SAMPLE_TEXTS = {
    "Chinese": [
        "你好，欢迎使用方言语音合成系统。",
        "今天天气真不错，阳光明媚。",
        "春眠不觉晓，处处闻啼鸟。",
        "山重水复疑无路，柳暗花明又一村。",
        "海内存知己，天涯若比邻。"
    ]
}

# Sample reference audios for each dialect (male and female versions)
SAMPLE_REFERENCE_AUDIOS = {
    "putonghua": {
        "male": "prompts/putonghua_male_prompt.wav",
        "female": "prompts/putonghua_female_prompt.wav"
    },
    "chengdu": {
        "male": "prompts/chengdu_male_prompt.wav",
        "female": "prompts/chengdu_female_prompt.wav"
    },
    "gaoxiong": {
        "male": "prompts/hokkien_male_prompt.wav",
        "female": "prompts/hokkien_female_prompt.wav"
    },
    "nanjing": {
        "male": "prompts/nanjing_male_prompt.wav",
        "female": "prompts/nanjing_female_prompt.wav"
    },
    "shanghai": {
        "male": "prompts/shanghai_male_prompt.wav",
        "female": "prompts/shanghai_female_prompt.wav"
    },
    "shijiazhuang": {
        "male": "prompts/shijiazhuang_male_prompt.wav",
        "female": "prompts/shijiazhuang_female_prompt.wav"
    },
    "tianjin": {
        "male": "prompts/tianjin_male_prompt.wav",
        "female": "prompts/tianjin_female_prompt.wav"
    },
    "xian": {
        "male": "prompts/xian_male_prompt.wav",
        "female": "prompts/xian_female_prompt.wav"
    },
    "zhengzhou": {
        "male": "prompts/zhengzhou_male_prompt.wav",
        "female": "prompts/zhengzhou_female_prompt.wav"
    }
}

def create_gradio_interface():
    """Create Gradio interface"""
    
    def synthesize_interface(text, dialect, ref_audio, ref_text, 
                           target_rms, nfe_step, speed):
        """Interface function for speech synthesis"""
        if not text.strip():
            return None, "Please enter text to synthesize"
        
        if not ref_audio:
            return None, "Please upload reference audio"
        
        if not ref_text.strip():
            return None, "Please enter reference text"
        
        audio_file, message = pipeline.synthesize_speech(
            text=text,
            dialect=dialect,
            ref_audio_path=ref_audio,
            ref_text=ref_text,
            target_rms=target_rms,
            cross_fade_duration=0.15,  # Fixed value
            nfe_step=nfe_step,
            cfg_strength=2.0,  # Fixed value
            sway_sampling_coef=-1.0,  # Fixed value
            speed=speed
        )
        
        return audio_file, message
    
    def frontend_only_interface(text, dialect):
        """Interface function for frontend processing only"""
        if not text.strip():
            return "", "", "Please enter text to process"
        
        processed_text, ipa_text, message = pipeline.generate_frontend_only(text, dialect)
        return processed_text, ipa_text, message
    
    def update_sample_text(sample_text):
        """Update text input with selected sample"""
        return sample_text
    
    def update_sample_audio_male(dialect):
        """Update reference audio and text with male sample for selected dialect"""
        dialect_samples = SAMPLE_REFERENCE_AUDIOS.get(dialect, {})
        audio_path = dialect_samples.get("male", "")
        
        ref_text = ""
        if audio_path and os.path.exists(audio_path):
            # 读取对应的txt文件
            txt_path = audio_path.replace('.wav', '.txt')
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        line = f.readline().strip()
                        # 解析格式: TEXT\t文本内容\tIPA内容
                        parts = line.split('\t')
                        if len(parts) >= 3:
                            ref_text = parts[1]  # 提取中间的文本部分
                        elif len(parts) == 2:
                            ref_text = parts[1]  # 兼容只有两列的情况
                except Exception as e:
                    print(f"Error reading reference text from {txt_path}: {e}")
            
            return audio_path, ref_text
        else:
            return None, ""
    
    def update_sample_audio_female(dialect):
        """Update reference audio and text with female sample for selected dialect"""
        dialect_samples = SAMPLE_REFERENCE_AUDIOS.get(dialect, {})
        audio_path = dialect_samples.get("female", "")
        
        ref_text = ""
        if audio_path and os.path.exists(audio_path):
            # 读取对应的txt文件
            txt_path = audio_path.replace('.wav', '.txt')
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        line = f.readline().strip()
                        # 解析格式: TEXT\t文本内容\tIPA内容
                        parts = line.split('\t')
                        if len(parts) >= 3:
                            ref_text = parts[1]  # 提取中间的文本部分
                        elif len(parts) == 2:
                            ref_text = parts[1]  # 兼容只有两列的情况
                except Exception as e:
                    print(f"Error reading reference text from {txt_path}: {e}")
            
            return audio_path, ref_text
        else:
            return None, ""
    
    # Create interface
    with gr.Blocks(title="DiaMoE-TTS") as demo:
        gr.Markdown("# 🎙️ DiaMoE-TTS")
        gr.Markdown("This is a zero-shot dialect speech synthesis system supporting multiple Chinese dialects")
        
        # Display model status
        with gr.Row():
            model_status_display = gr.Markdown(
                f"**Model Status**: {'✅ Loaded' if pipeline.model_loaded else '❌ Not Loaded'} | "
                f"**Model Config**: {MODEL_CONFIG['model_name']} | "
                f"**MoE**: {'Yes' if MODEL_CONFIG['use_moe'] else 'No'} "
                f"({MODEL_CONFIG['num_exps']} experts, Top{MODEL_CONFIG['moe_topK']})"
            )
        
        with gr.Tab("Frontend Processing"):
            gr.Markdown("## Text Frontend Processing Only")
            gr.Markdown("Generate phonetic representation without TTS synthesis")
            
            with gr.Row():
                with gr.Column():
                    frontend_text_input = gr.Textbox(
                        label="Input Text (Chinese text input supported)", 
                        placeholder="Please enter Chinese text to process...",
                        lines=3
                    )
                    
                    frontend_dialect_choice = gr.Dropdown(
                        label="Select Dialect (Choose dialect type for processing)", 
                        choices=pipeline.dialect_list,
                        value="putonghua"
                    )
                    
                    frontend_process_btn = gr.Button("🔤 Process Frontend", variant="primary")
                
                with gr.Column():
                    processed_output = gr.Textbox(
                        label="Processed Text", 
                        interactive=False,
                        lines=3
                    )
                    
                    ipa_output = gr.Textbox(
                        label="IPA Format", 
                        interactive=False,
                        lines=3
                    )
                    
                    frontend_status = gr.Textbox(
                        label="Processing Status", 
                        interactive=False
                    )
            
            frontend_process_btn.click(
                fn=frontend_only_interface,
                inputs=[frontend_text_input, frontend_dialect_choice],
                outputs=[processed_output, ipa_output, frontend_status]
            )
        
        with gr.Tab("Speech Synthesis"):
            gr.Markdown("## Zero-Shot Speech Synthesis")
            
            with gr.Row():
                with gr.Column():
                    # Sample text selection
                    gr.Markdown("### Sample Texts")
                    sample_text_buttons = []
                    for text in SAMPLE_TEXTS["Chinese"]:
                        btn = gr.Button(text[:20] + "..." if len(text) > 20 else text, size="sm")
                        sample_text_buttons.append((btn, text))
                    
                    text_input = gr.Textbox(
                        label="Input Text (Chinese text input supported)", 
                        placeholder="Please enter Chinese text to synthesize...",
                        lines=3
                    )
                    
                    dialect_choice = gr.Dropdown(
                        label="Select Dialect (Choose dialect type for synthesis)", 
                        choices=pipeline.dialect_list,
                        value="putonghua"
                    )
                    
                    # Sample reference audio buttons
                    gr.Markdown("### Reference Audio")
                    with gr.Row():
                        sample_audio_male_btn = gr.Button("📁 Use Sample Audio (Male)", size="sm")
                        sample_audio_female_btn = gr.Button("📁 Use Sample Audio (Female)", size="sm")
                    
                    with gr.Row():
                        ref_audio = gr.Audio(
                            label="Reference Audio (Upload reference audio file)", 
                            type="filepath"
                        )
                        ref_text = gr.Textbox(
                            label="Reference Text (Text content corresponding to reference audio)", 
                            placeholder="Text corresponding to reference audio...",
                            lines=2
                        )
                
                with gr.Column():
                    gr.Markdown("### Parameters")
                    
                    target_rms = gr.Slider(
                        label="Target Volume (Audio volume normalization value)", 
                        minimum=0.01, 
                        maximum=1.0, 
                        value=0.1,
                        step=0.01
                    )
                    
                    nfe_step = gr.Slider(
                        label="Denoising Steps (Diffusion model denoising steps)", 
                        minimum=1, 
                        maximum=100, 
                        value=32,
                        step=1
                    )
                    
                    speed = gr.Slider(
                        label="Speech Speed (Speech playback speed)", 
                        minimum=0.1, 
                        maximum=3.0, 
                        value=1.0,
                        step=0.1
                    )
            
            synthesize_btn = gr.Button("🎵 Start Synthesis", variant="primary")
            
            with gr.Row():
                output_audio = gr.Audio(label="Synthesis Result", type="filepath")
                synthesis_status = gr.Textbox(label="Synthesis Status", interactive=False)
            
            # Connect sample text buttons
            for btn, sample_text in sample_text_buttons:
                btn.click(
                    fn=update_sample_text,
                    inputs=[gr.State(sample_text)],
                    outputs=[text_input]
                )
            
            # Connect sample audio buttons
            sample_audio_male_btn.click(
                fn=update_sample_audio_male,
                inputs=[dialect_choice],
                outputs=[ref_audio, ref_text]
            )
            
            sample_audio_female_btn.click(
                fn=update_sample_audio_female,
                inputs=[dialect_choice],
                outputs=[ref_audio, ref_text]
            )
            
            synthesize_btn.click(
                fn=synthesize_interface,
                inputs=[
                    text_input, dialect_choice, ref_audio, ref_text,
                    target_rms, nfe_step, speed
                ],
                outputs=[output_audio, synthesis_status]
            )
        
        with gr.Tab("User Guide"):
            gr.Markdown("""
            ## 📖 User Guide
            
            ### 1. Model Information
            - **Model Status**: System automatically preloads model at startup
            - **Configuration**: Model parameters are preset in the script
            - **MoE Architecture**: Supports Mixture of Experts model inference
            
            ### 2. Frontend Processing
            - **Input Text**: Supports Chinese text, automatic frontend processing
            - **Dialect Selection**: Supports multiple Chinese dialects
            - **Output**: Provides processed phonetic representation and IPA format
            
            ### 3. Speech Synthesis
            - **Sample Texts**: Click sample text buttons to auto-fill input
            - **Input Text**: Supports Chinese text, automatic frontend processing
            - **Dialect Selection**: Supports multiple Chinese dialects
            - **Reference Audio**: Upload clear reference audio file (WAV format recommended)
            - **Reference Text**: Enter text content corresponding to reference audio
            - **Sample Audio**: Click "Use Sample Audio" to load dialect-specific reference
            
            ### 4. Supported Dialects
            """)
            
            dialect_info = gr.DataFrame(
                value=[
                    ["putonghua", "Mandarin Chinese"],
                    ["chengdu", "Chengdu Dialect"],
                    ["gaoxiong", "Kaohsiung Dialect"], 
                    ["shanghai", "Shanghai Dialect"],
                    ["shijiazhuang", "Shijiazhuang Dialect"],
                    ["wuhan", "Wuhan Dialect"],
                    ["xian", "Xi'an Dialect"],
                    ["zhengzhou", "Zhengzhou Dialect"]
                ],
                headers=["Dialect Code", "Dialect Name"],
                interactive=False
            )
            
            gr.Markdown("""
            ### 5. Notes
            - Ensure reference audio has good quality with minimal noise
            - Reference text must exactly match audio content
            - Model loading is required for first-time use
            - Synthesis time depends on text length and hardware performance
            - Use sample texts and reference audios for quick testing
            """)
    
    return demo

if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description="Dialect TTS Gradio Interface")
    parser.add_argument("--port", type=int, default=7860, help="Service port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Service host")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()
    
    # Create and launch interface
    demo = create_gradio_interface()
    print("Starting Dialect TTS Gradio Interface...")
    print(f"Service address: http://{args.host}:{args.port}")
    
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True
    )
