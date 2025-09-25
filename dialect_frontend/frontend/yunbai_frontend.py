# coding=utf-8
from frontend.dialect_frontend import DialectFrontend


class YunbaiFrontend(DialectFrontend):
    def __init__(self, g2p_model="g2pW", vocab_phones=None, vocab_tones=None, use_rhy=False, ):
        super().__init__(dialect_type="yunbai", g2p_model=g2p_model, vocab_phones=vocab_phones, vocab_tones=vocab_tones, use_rhy=use_rhy,)
        
    