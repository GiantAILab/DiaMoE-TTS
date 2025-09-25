#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
方言TTS Zero Shot推理的Gradio界面
整合文本前端处理和TTS模型推理
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
    "ckpt_file": "/user-fs/chenzihao/wangzixin/f5_dialect/ckpts/loras/10ep_mlpEXP_model_state_dict.pt",  # 请修改为你的模型路径
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
            "putonghua", "chengdu", "gaoxiong", "jingjujingbai", 
            "jingjuyunbai", "nanjing", "qingdao", "shanghai", 
            "shijiazhuang", "wuhan", "xian", "zhengzhou"
        ]
        
        # 初始化前端处理器
        self.preprocessor = Preprocessor()
        self.frontend = self.preprocessor.frontend['ZH']
        print("前端处理器初始化完成!")
        
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
            
            print(f"加载了{len(self.ipa_list)}个IPA音素和{len(self.punctuation_list)}个标点符号")
            
        except Exception as e:
            print(f"加载词汇表失败: {e}")
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
                    print(f'警告: 未知符号 {symbol}')
                # 跳过未知符号
                continue
        
        result = ' '.join(ipa_text)
        print(f"IPA格式转换: {text[:50]}... -> {result[:50]}...")
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
            
            print(f"文本转拼音: {text} -> {ppinyins}")
            return zhongwens, ppinyins, oop
        except Exception as e:
            print(f"拼音转换错误: {e}")
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
        print(f"开始处理方言前端: {dialect}")
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        try:
            # 步骤1: 直接使用已加载的前端处理器生成拼音
            print("执行拼音转换...")
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
                # 运行single_frontend.sh脚本
                frontend_command = f"bash single_frontend.sh all {dialect} {pinyin_file}"
                print(f"执行前端处理: {frontend_command}")
                
                ret_code, stdout, stderr = self.run_shell_command(frontend_command, cwd="./dialect_frontend")
                
                if ret_code != 0:
                    print(f"前端处理失败: {stderr}")
                    print(f"标准输出: {stdout}")
                    # 使用原始拼音文件作为回退
                    final_output = pinyin_file
                else:
                    # 查找最终的IPA格式文件
                    base_name = os.path.splitext(pinyin_file)[0]
                    ipa_file = base_name + "_ipa_format.txt"
                    if os.path.exists(ipa_file):
                        final_output = ipa_file
                    else:
                        print("未找到IPA格式文件，使用拼音文件")
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
                print(f"输出文件不存在: {final_output}")
                return text
                
        except Exception as e:
            print(f"前端处理出错: {e}")
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
            print("开始加载TTS模型...")
            
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
            print("Vocoder加载成功!")
            
            # 加载TTS模型
            model_cfg_path = f"./diamoe_tts/src/f5_tts/configs/{model_name}.yaml"
            if not os.path.exists(model_cfg_path):
                model_cfg_path = "./diamoe_tts/src/f5_tts/configs/diamoetts.yaml"
            
            model_cfg = OmegaConf.load(model_cfg_path)
            model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
            model_arc = model_cfg.model.arch
            
            print(f"使用模型配置: {model_cfg_path}")
            print(f"模型类: {model_cls}, 架构: {model_arc}")
            
            if ckpt_file and os.path.exists(ckpt_file):
                print(f"从检查点加载模型: {ckpt_file}")
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
                print("TTS模型加载成功!")
            else:
                print("未提供有效的模型检查点文件")
                self.model_loaded = False
                
        except Exception as e:
            print(f"模型加载失败: {e}")
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
            return None, "模型未加载，请先加载模型"
        
        try:
            print(f"开始合成语音: {dialect}")
            print(f"输入文本: {text}")
            
            # 步骤1: 前端处理
            processed_text = self.process_frontend_pipeline(text, dialect)
            print(f"前端处理结果: {processed_text}")
            
            # 步骤2: 预处理参考音频和文本
            ref_audio, ref_text_processed = preprocess_ref_audio_text(ref_audio_path, ref_text)
            print(f"参考音频处理完成")
            
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
            
            print(f"IPA格式转换 - 生成文本: {processed_text_ipa}")
            print(f"IPA格式转换 - 参考文本: {ref_text_processed_ipa}")
            
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
            
            return output_file.name, "合成成功!"
            
        except Exception as e:
            print(f"语音合成失败: {e}")
            return None, f"合成失败: {str(e)}"

# 全局管道实例 - 预加载模型
print("正在初始化方言TTS管道并预加载模型...")
pipeline = DialectTTSPipeline(auto_load_model=True)

# Sample texts for quick input
SAMPLE_TEXTS = [
    ["Hello, this is a test.", "Hello, this is a test."],
    ["How are you today?", "How are you today?"],
    ["Welcome to DiaMoE-TTS!", "Welcome to DiaMoE-TTS!"],
    ["The weather is nice today.", "The weather is nice today."],
    ["Thank you for using our system.", "Thank you for using our system."]
]

# Default reference audio and text for each dialect
DEFAULT_REFS = {
    "putonghua": {
        "audio_path": "/path/to/putonghua_ref.wav",
        "text": "This is a Putonghua reference."
    },
    "chengdu": {
        "audio_path": "/path/to/chengdu_ref.wav", 
        "text": "This is a Chengdu dialect reference."
    },
    "shanghai": {
        "audio_path": "/path/to/shanghai_ref.wav",
        "text": "This is a Shanghai dialect reference."
    },
    "jingjuyunbai": {
        "audio_path": "/path/to/jingjuyunbai_ref.wav",
        "text": "This is a Jingju Yunbai reference."
    },
    # Add more dialects as needed
}

def create_gradio_interface():
    """Create Gradio Interface"""
    
    def convert_text_to_ipa_interface(text, dialect):
        """Convert text to IPA format interface function"""
        if not text.strip():
            return "Please input text to convert", ""
        
        try:
            # Process frontend pipeline
            processed_text = pipeline.process_frontend_pipeline(text, dialect)
            # Convert to IPA format
            ipa_result = pipeline.convert_to_ipa_format(processed_text)
            return processed_text, ipa_result
        except Exception as e:
            return f"Error: {str(e)}", ""
    
    def fill_sample_text(evt: gr.SelectData):
        """Fill sample text when clicked"""
        return SAMPLE_TEXTS[evt.index][1]
    
    def fill_default_ref(dialect):
        """Fill default reference audio and text for selected dialect"""
        if dialect in DEFAULT_REFS:
            ref_info = DEFAULT_REFS[dialect]
            return ref_info["audio_path"], ref_info["text"]
        return None, ""
    
    def synthesize_interface(text, dialect, ref_audio, ref_text, 
                           target_rms, cross_fade_duration, nfe_step, 
                           cfg_strength, sway_sampling_coef, speed):
        """Speech synthesis interface function"""
        if not text.strip():
            return None, "Please input text to synthesize"
        
        if not ref_audio:
            return None, "Please upload reference audio"
        
        if not ref_text.strip():
            return None, "Please input reference text"
        
        audio_file, message = pipeline.synthesize_speech(
            text=text,
            dialect=dialect,
            ref_audio_path=ref_audio,
            ref_text=ref_text,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed
        )
        
        return audio_file, message
    
    # Create interface with blue-purple theme
    with gr.Blocks(
        title="DiaMoE-TTS: Dialectal Mixture of Experts Text-to-Speech",
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.purple,
            secondary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate
        ).set(
            body_background_fill_dark="*neutral_950",
            block_background_fill="*neutral_50",
            block_border_width="1px",
            block_title_text_weight="600"
        )
    ) as demo:
        gr.Markdown(
            "# 🎙️ DiaMoE-TTS: Dialectal Mixture of Experts Text-to-Speech",
            elem_classes="center-text"
        )
        gr.Markdown(
            "A zero-shot dialectal speech synthesis system supporting multiple Chinese dialects with MoE architecture.",
            elem_classes="center-text"
        )
        
        # Display model status
        with gr.Row():
            model_status_display = gr.Markdown(
                f"**Model Status**: {'✅ Loaded' if pipeline.model_loaded else '❌ Not Loaded'} | "
                f"**Model Config**: {MODEL_CONFIG['model_name']} | "
                f"**MoE**: {'Yes' if MODEL_CONFIG['use_moe'] else 'No'} "
                f"({MODEL_CONFIG['num_exps']} experts, Top{MODEL_CONFIG['moe_topK']})"
            )
        
        # Text to IPA Conversion Module
        with gr.Tab("📝 Text to IPA Conversion"):
            gr.Markdown("## Convert Text to IPA Phonemes")
            
            with gr.Row():
                with gr.Column():
                    ipa_input_text = gr.Textbox(
                        label="Input Text",
                        placeholder="Enter text to convert to IPA...",
                        lines=3
                    )
                    
                    ipa_dialect_choice = gr.Dropdown(
                        label="Select Dialect",
                        choices=pipeline.dialect_list,
                        value="putonghua"
                    )
                    
                    convert_btn = gr.Button("🔄 Convert to IPA", variant="primary")
                
                with gr.Column():
                    ipa_frontend_output = gr.Textbox(
                        label="Frontend Processing Result",
                        lines=3,
                        interactive=False
                    )
                    
                    ipa_final_output = gr.Textbox(
                        label="Final IPA Format", 
                        lines=3,
                        interactive=False
                    )
            
            convert_btn.click(
                fn=convert_text_to_ipa_interface,
                inputs=[ipa_input_text, ipa_dialect_choice],
                outputs=[ipa_frontend_output, ipa_final_output]
            )
        
        # Speech Synthesis Module 
        with gr.Tab("🎵 Speech Synthesis"):
            gr.Markdown("## Zero-Shot Dialectal Speech Synthesis")
            
            with gr.Row():
                # Left Column - Input Section
                with gr.Column():
                    # Sample Texts Table
                    gr.Markdown("### 📋 Quick Input Examples")
                    sample_table = gr.DataFrame(
                        value=SAMPLE_TEXTS,
                        headers=["ID", "Sample Text"],
                        interactive=False,
                        wrap=True
                    )
                    
                    text_input = gr.Textbox(
                        label="Input Text", 
                        placeholder="Enter text to synthesize...",
                        lines=3
                    )
                    
                    dialect_choice = gr.Dropdown(
                        label="Select Dialect", 
                        choices=pipeline.dialect_list,
                        value="putonghua"
                    )
                    
                    # Default Reference Selection
                    gr.Markdown("### 🎯 Quick Reference Selection")
                    use_default_ref_btn = gr.Button("📁 Use Default Reference for Selected Dialect", variant="secondary")
                    
                    with gr.Row():
                        ref_audio = gr.Audio(
                            label="Reference Audio", 
                            type="filepath"
                        )
                        ref_text = gr.Textbox(
                            label="Reference Text", 
                            placeholder="Reference audio transcription...",
                            lines=2
                        )
                
                # Right Column - Advanced Parameters
                with gr.Column():
                    gr.Markdown("### ⚙️ Advanced Parameters")
                    
                    target_rms = gr.Slider(
                        label="Target RMS", 
                        info="Output audio volume normalization value",
                        minimum=0.01, 
                        maximum=1.0, 
                        value=0.1,
                        step=0.01
                    )
                    
                    cross_fade_duration = gr.Slider(
                        label="Cross-fade Duration", 
                        info="Cross-fade duration between audio segments",
                        minimum=0.0, 
                        maximum=1.0, 
                        value=0.15,
                        step=0.01
                    )
                    
                    nfe_step = gr.Slider(
                        label="NFE Steps", 
                        info="Number of function evaluation steps",
                        minimum=1, 
                        maximum=100, 
                        value=32,
                        step=1
                    )
                    
                    cfg_strength = gr.Slider(
                        label="CFG Strength", 
                        info="Classifier-free guidance strength",
                        minimum=0.0, 
                        maximum=5.0, 
                        value=2.0,
                        step=0.1
                    )
                    
                    sway_sampling_coef = gr.Slider(
                        label="Sway Sampling Coefficient", 
                        info="Sway sampling coefficient",
                        minimum=-2.0, 
                        maximum=2.0, 
                        value=-1.0,
                        step=0.1
                    )
                    
                    speed = gr.Slider(
                        label="Speed", 
                        info="Speech playback speed",
                        minimum=0.1, 
                        maximum=3.0, 
                        value=1.0,
                        step=0.1
                    )
            
            synthesize_btn = gr.Button("🎵 Start Synthesis", variant="primary", size="lg")
            
            with gr.Row():
                output_audio = gr.Audio(label="Synthesis Result", type="filepath")
                synthesis_status = gr.Textbox(label="Synthesis Status", interactive=False)
            
            # Event handlers
            sample_table.select(
                fn=fill_sample_text,
                inputs=None,
                outputs=text_input
            )
            
            use_default_ref_btn.click(
                fn=fill_default_ref,
                inputs=dialect_choice,
                outputs=[ref_audio, ref_text]
            )
            
            synthesize_btn.click(
                fn=synthesize_interface,
                inputs=[
                    text_input, dialect_choice, ref_audio, ref_text,
                    target_rms, cross_fade_duration, nfe_step, 
                    cfg_strength, sway_sampling_coef, speed
                ],
                outputs=[output_audio, synthesis_status]
            )
        
        with gr.Tab("📖 Documentation"):
            gr.Markdown("""
            ## 📖 Usage Guide
            
            ### 1. Model Information
            - **Model Status**: Model is automatically pre-loaded at system startup
            - **Configuration**: Model parameters are preset in the script
            - **MoE Architecture**: Supports multi-expert mixture model inference
            
            ### 2. Text to IPA Conversion
            - **Input Text**: Supports Chinese text with automatic frontend processing
            - **Dialect Selection**: Choose the target dialect for conversion
            - **Frontend Processing**: Shows intermediate processing results
            - **Final IPA**: Displays the final IPA phoneme sequence
            
            ### 3. Speech Synthesis
            - **Quick Examples**: Click on sample texts to quickly fill input
            - **Input Text**: Supports Chinese text with automatic frontend processing
            - **Dialect Selection**: Supports multiple Chinese dialects
            - **Reference Audio**: Upload clear reference audio files (WAV format recommended)
            - **Reference Text**: Input the transcription of reference audio
            - **Default References**: Use predefined reference audio for each dialect
            
            ### 4. Supported Dialects
            """)
            
            dialect_info = gr.DataFrame(
                value=[
                    ["putonghua", "Mandarin Chinese"],
                    ["chengdu", "Chengdu Dialect"],
                    ["gaoxiong", "Kaohsiung Dialect"], 
                    ["jingjujingbai", "Beijing Opera Jingbai"],
                    ["jingjuyunbai", "Beijing Opera Yunbai"],
                    ["nanjing", "Nanjing Dialect"],
                    ["qingdao", "Qingdao Dialect"],
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
            ### 5. Important Notes
            - Ensure reference audio quality is good with minimal noise
            - Reference text must match the audio content exactly
            - The model is pre-loaded at system startup
            - Synthesis time depends on text length and hardware performance
            - Use the quick examples and default references for faster setup
            
            ### 6. Features
            - **Zero-shot synthesis**: Generate speech in any supported dialect
            - **MoE Architecture**: Efficient mixture of experts model
            - **Frontend Processing**: Automatic text normalization and IPA conversion
            - **Multiple Dialects**: Support for 12+ Chinese dialects
            - **Quick Setup**: Sample texts and default references for easy testing
            """)
    
    return demo

if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description="DiaMoE-TTS Gradio Interface")
    parser.add_argument("--port", type=int, default=30769, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()
    
    # Create and launch interface
    demo = create_gradio_interface()
    print("Starting DiaMoE-TTS Gradio Interface...")
    print(f"Server URL: http://{args.host}:{args.port}")
    
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        show_error=True
    )
