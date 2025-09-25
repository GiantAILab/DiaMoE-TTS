# coding=utf-8
from frontend.dialect_frontend import DialectFrontend


class HenanFrontend(DialectFrontend):
    def __init__(self, g2p_model="g2pW", vocab_phones=None, vocab_tones=None, use_rhy=False, ):
        super().__init__(dialect_type="henan", g2p_model=g2p_model, vocab_phones=vocab_phones, vocab_tones=vocab_tones, use_rhy=use_rhy,)
        self.dialect2pt_tone = {"0": "0", "3": "1", "1": "2", "4": "3", "2": "4", "5": "5"}   # 河南话音调转普通话音调
        
    
    # 连续变调：一七八不 这四个字，平常读第一声，在第四声前读第二声。读第四声的字，在第四声前面读第一声。
    def apply_tone_sandhi(self, word_pinyin, special_words="一七八不"):
        def split_pinyin(pinyin):
            for i in range(len(pinyin)):
                if pinyin[i].isdigit():
                    return pinyin[:i], pinyin[i]
            return pinyin, ''  # default case, although should not happen

        def adjust_pinyin(word, pinyin_list):
            adjusted_list = []
            for i in range(len(pinyin_list)):
                current_pinyin, current_tone = split_pinyin(pinyin_list[i])
                
                if current_tone == '1' and word[i] in special_words:
                    # 检查当前拼音是否应该变调
                    if i < len(pinyin_list) - 1:
                        next_pinyin, next_tone = split_pinyin(pinyin_list[i + 1])
                        if next_tone == '4':
                            current_tone = '2'  # 将当前拼音变为第二声
                            
                if current_tone == '4':
                    # 遇到第四声时，检查后一个字是否是第四声
                    if i < len(pinyin_list) - 1:
                        next_pinyin, next_tone = split_pinyin(pinyin_list[i + 1])
                        if next_tone == '4':
                            current_tone = '1'  # 将当前拼音变为第一声
                
                adjusted_list.append(current_pinyin + current_tone)
            return adjusted_list

        # 新的字典，包含变调后的结果
        new_pinyin_dict = {}
        for word, pinyin_str in word_pinyin.items():
            # if len(pinyin_str) == 1:  # 标点情况
            if word in self.punc:  # 标点情况
                new_pinyin_dict[word] = pinyin_str
            else:
                pinyin_list = pinyin_str.split()
                adjusted_pinyin_list = adjust_pinyin(word, pinyin_list)
                new_pinyin_dict[word] = ' '.join(adjusted_pinyin_list)
        
        return new_pinyin_dict
    
    
    # 普通话读音到河南话读音的规则，1000句的那个规则
    def pt_to_dialect(self, seg, seg_cut, pinyins):
        # seg 是一句话， pinyins 是这句话对应的拼音，按道理每个字都有一个对应的拼音。
        # print(seg, pinyins)
        assert len(seg) == len(pinyins), print("There was an error when getting the pinyin")  
        for i in range(len(seg)):   # 过规则之前，首先把所有的 一 变成 yi1, 不 变成 bu1
            if seg[i] == "一":
                pinyins[i] = "yi1"
            elif seg[i] == "不":
                pinyins[i] = "bu1"
                
        word_pinyin = self.pt_to_dialect_base(seg, seg_cut, pinyins)
        word_pinyin = self.apply_tone_sandhi(word_pinyin)   # 河南话连续变调
        word_pinyin = self.change_to_pttone(word_pinyin)
        return word_pinyin

