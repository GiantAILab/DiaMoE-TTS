# DiaMoE-TTS

<p align="center">
  <img src="../pics/diamoe_tts.PNG" width="720" alt="backbone">
</p>

## Configs
### `src/f5_tts/configs/diamoetts.yaml`
This document is used to define the specific structure of the model, the hyperparameters used in the training process, and the fine-tuning strategies employed.

**Key Configuration Items**:
*   `MoE`:
    *   `use_moe`: bool. `true` for using Mixture of Experts (MoE) .
    *   `num_experts`: int. Num of experts in MoE.
*   `model`:
    *   `backbone`: str.  `DiT` using in stage 1/2, `DiT_peft` for peft in stage 3。

## Training
### Preprocessing
  ```bash
  python ./src/f5_tts/train/datasets/prepare_ipa.py
  ```
  save arrow dataset for training
### Start training
```bash
accelerate launch --config_file default_config.yaml \
  src/f5_tts/train/train.py \
  --config-name diamoetts.yaml
```

## Infer
```bash
bash ./src/f5_tts/infer/batch_infer.sh
```
### ⚠️ Note
Please be aware of the following:
- Before doing inference, the IPA frontend in testset and reference should be: [IPA1] [IPA2], 
rather than IPA1 | IPA2
- e.g. 
'[t͡ʂ] [ˈuᴴᴹ] [ɻ] [ˈəᴴᴸ] [n] [n] [ˈiᴴᴹ] [h] [ˈɑᴴᴹ] [ʊ̯] ，' is correct, while 
't͡ʂ ˈuᴴᴹ |ɻ ˈəᴴᴸ n |n ˈiᴴᴹ |h ˈɑᴴᴹ ʊ̯ |，' will lead to the failure of inference