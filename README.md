# DIAMOE-TTS: A Unified IPA-based Dialect TTS Framework with Mixture-of-Experts and Parameter-Efficient Zero-Shot Adaptation

---

## TODO
- [x] release code for train/infer
- [x] release code for IPA frontend
- [ ] release training dataset
- [ ] release our checkpoints

---

## Installation
```bash
# clone code
git clone https://github.com/GiantAILab/DiaMoE-TTS.git
cd DiaMoE-TTS

# conda environment
conda create -n diamoetts python=3.10
conda activate diamoetts
cd diamoe_tts
pip install -e .
```

---

## Train/Infer
See [diamoe_tts](./diamoe_tts/README.md) for more details.

## IPA frontend
See [ipa_frontend](./dialect_frontend/README.md) for more details.

## Acknowledgements
- Thanks to all contributors and community members who helped improve this project.
- This work builds upon [F5-TTS](https://github.com/SWivid/F5-TTS) and related research.

---

## License
Our code is released under MIT License. 
