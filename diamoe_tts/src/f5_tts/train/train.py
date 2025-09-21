# training script.

import os
import pdb
import sys
from importlib.resources import files

import hydra
from omegaconf import OmegaConf

from f5_tts.model import CFM
from f5_tts.model import Trainer
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer




os.chdir(str(files("f5_tts").joinpath("../..")))  # change working directory to root of project (local editable)
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"


@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs")), config_name=None)
def main(model_cfg):
    model_cls = hydra.utils.get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    tokenizer = model_cfg.model.tokenizer
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type

    exp_name = f"{model_cfg.model.name}_{mel_spec_type}_{model_cfg.model.tokenizer}_{model_cfg.datasets.name}"
    wandb_resume_id = None

    # set text tokenizer
    if tokenizer != "custom":
        tokenizer_path = model_cfg.datasets.name
        vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer, model_cfg.datasets.vocab)
    else:
        tokenizer_path = model_cfg.model.tokenizer_path
        vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    # MoE
    if model_cfg.MoE.use_moe:
        num_exps = model_cfg.MoE.num_exps
        moe_topK = model_cfg.MoE.moe_topK
        use_dialect_clf = model_cfg.MoE.use_dialect_clf
        dialect_clf_lambda = model_cfg.MoE.dialect_clf_lambda
        dialect_kinds = model_cfg.MoE.dialect_kinds
        expert_type = model_cfg.MoE.expert_type.lower()
        from f5_tts.model.moe import SimpleGate, MoeLayer, EXPERT_DICT

        assert expert_type in EXPERT_DICT, f'当前专家种类仅支持{list(EXPERT_DICT.keys())}'
        expert = EXPERT_DICT[expert_type]
        experts = [expert() for _ in range(num_exps)]  # text_embed:(batch,length, 512)
        gate = SimpleGate(num_exps)
        moe = MoeLayer(experts=experts,
                       gate=gate,
                       num_experts=num_exps,
                       num_experts_per_tok=moe_topK,
                       use_residual=True,
                       use_dialect_clf=use_dialect_clf,
                       dialect_clf_lambda=dialect_clf_lambda,
                       dialect_kinds=dialect_kinds)
    else:
        moe = None


    # set model
    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=model_cfg.model.mel_spec.n_mel_channels, moe=moe),
        mel_spec_kwargs=model_cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
        use_moe = model_cfg.MoE.use_moe
    )

    # init trainer

    trainer = Trainer(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=str(files("f5_tts").joinpath(f"../../{model_cfg.ckpts.save_dir}")),
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_project="CFM-TTS",
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
        basic_ckpt_path=model_cfg.ckpts.basic_ckpt_path,
        moe_args = model_cfg.MoE,
        use_old_scheduler = model_cfg.optim.use_old_scheduler

    )


    train_dataset = load_dataset(model_cfg.datasets.name, tokenizer, mel_spec_kwargs=model_cfg.model.mel_spec, arrow_name=model_cfg.datasets.arrow_name)
    print(len(train_dataset))


    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
