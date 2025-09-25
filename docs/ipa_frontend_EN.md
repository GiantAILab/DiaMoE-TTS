# Abstract
The International Phonetic Alphabet (IPA) is the most widely used phonetic annotation system in the investigation and study of Chinese dialects. The vast majority of Chinese dialect corpora, including homophone tables, dictionaries and texts, utilize the IPA for phonetic transcription. The phonetic annotation system for this project is based on the IPA. It constructs a highly scalable phoneme inventory (currently containing 442 units) from a base of 100+ IPA phoneme symbols. This system is designed to support the phonetic annotation of all known Chinese dialects and is also extensible to European languages. (It currently supports 11 dialects and Mandarin; its validity has also been verified for English, French, German and the Bildts dialect of Dutch).

# Phonetic Annotation
The phonetic annotation system for this project is based on the IPA.
Phonetic Annotation Example:
(Each line follows the format: <audio_id>\t<text>\t<phonetic_annotation>. Punctuation marks from the text must be retained in the phonetic annotation.)
```
TEXT    好的开始，是成功的一半。    χ ˈɑᴹᴸ ʊ̯ | t ə | kʰ ˈaᴴᴴ ɪ̯ | ʂ ˈɻ̍ᴹᴸ | ， | ʂ ˈɻ̍ᴴᴸ | t͡ʂʰ ˈɤᴹᴴ ŋ | k ˈʊᴴᴴ ŋ | t ə | j ˈiᴹᴴ | p ˈaᴴᴸ n | 。
```
Phonetic Annotation Rules:
1. General Rules
Phonemes are separated by spaces " ", and syllables are separated by " | ".
2. Phonemes Annotation
Phonemes are categorized into five types: Model Consonants, Model Vowels, Syllabic Consonants, Non-Syllabic Vowels, and Toned Vowels.

① Model Consonants: Phonemes listed under "Consonants" in the main IPA chart, along with complex articulations marked using diacritics such as the ligature tie " ͜ " or " ͡ ", nasalization " ˜", palatalization "ʲ", velarization " ̴ " (avoid using "ˠ"), labialization "ʷ", laminalization " ̻ ", etc. Examples: "m", "b", "p", "t͜s", "k͡p", "ɹ̻ʷ".
② Model Vowels: Phonemes listed under "Vowels" in the main IPA chart, including their lengthened forms marked by diacritics like the long mark "ː", half-long mark "ˑ", or extra-short mark " ˘". Examples: "a", "i", "u", "aː", "iˑ", "ŭ". When transcribing a nasalized vowel, the nasalization diacritic "˜" must be written as a separate symbol. For example, "ã" should be written as "˜ a" (indicating nasalization throughout the entire vowel) or "a ˜" (indicating nasalization primarily in the latter part of the vowel).
③ Syllabic Consonants: Model Consonants marked with a syllabic diacritic (above or below). Examples: "ɹ̩", "ŋ̍", "ʋ̩".
④ Non-Syllabic Vowels: Model Vowels (without length diacritics) marked with a non-syllabic diacritic below. Examples: "ɪ̯", "ʊ̯", "ə̯".
⑤ Toned Vowels: Model Vowels preceded by the stress mark "ˈ" and followed by a tone mark. Examples: "ˈɔᴴᴸ", "ˈoᴴᴴ", "ˈɚᴴᴹ".
When selecting appropriate phoneme symbols, adhere to the following principles:
  1. Do not omit the aforementioned articulation diacritics, syllabic diacritics, non-syllabic diacritics, or other symbols constituting syllabic consonants, non-syllabic vowels or toned vowels. For example, write "t͜s", not "t s"; write "k͡p", not "k p"; write "m̩", not "m"; write "ŋ̍", not "ŋ"; write "a ɪ̯", not "a ɪ".
  2. Avoid using certain extended symbols:
Avoid using the following symbols not defined in standard IPA:
    1. The open central unrounded vowel "ᴀ": Rewrite based on actual phonetic value and phonemic contrasts, typically as "a" or "ɑ". For instance, Mandarin "ᴀ" is recommended to be written as "ɑ".
    2. The mid front unrounded vowel "ᴇ": Rewrite based on actual phonetic value and phonemic contrasts, typically as "ɛ" or "e". For instance, Mandarin "ᴇ" is recommended to be written as "ɛ".
    3. The apical vowels "ɿ", "ʮ", "ʅ", "ʯ": Rewrite as the corresponding syllabic consonants with the same place of articulation, without or with labialization: "ɹ̩", "ɹ̩ʷ", "ɻ̍", "ɻ̍ʷ" respectively.
  3. Avoid over-phonemicization:
Avoid transcribing a phoneme that can be represented by the IPA (especially without diacritics) into a different phoneme or phoneme cluster that does not conflict with it topologically. For example, write "æ", not "a i"; write "ɚ", not "e r"; write "ɻ", not "ʐ".
  4. Avoid other specific diacritics:
Avoid using the following IPA diacritics:
    1. Rhoticity diacritic "˞": Rewrite as the sequence of the rhotic mid central vowel with the non-syllabic diacritic "ɚ̯". For example, "a˞" is recommended to be written as "a ɚ̯".
    2. Release diacritics like "ˡ", "ʴ", "ˢ", etc.: Rewrite as a lateral flap or an affricate. For example, "dˡ" is recommended to be written as "ɺ"; "tˢ" as "t͜s".
    3. More rounded " ̹" and less rounded " ̜" diacritics: Vowels with these diacritics should be rewritten based on actual phonetic value and phonemic contrasts as another vowel with the same rounding but a slightly different tongue position (half-step difference vertically or horizontally). For example, "ə̹" is recommended to be written as "ɵ".
3. Tones Annotation
  Chinese linguistics traditionally uses the vertical bar tone marking system or its "transcription" – the five-level tone numeral system, which divides the pitch range into five levels. This project uses the High-Medium-Low (HML) tone marking system, dividing the pitch range into three levels. Generally, the traditional five levels "5", "3", "1" correspond to this project's "ᴴ", "ᴹ", "ᴸ" respectively. The traditional level "4" should be merged into "ᴴ" or "ᴹ" based on context, and level "2" should be merged into "ᴹ" or "ᴸ" based on context.
  This project uses only two tone categories, totaling 12 marks:
  ① Checked Tones: ᴴ, ᴹ, ᴸ
  ② Non-checked Tones: ᴴᴴ, ᴴᴹ, ᴴᴸ, ᴹᴴ, ᴹᴹ, ᴹᴸ, ᴸᴴ, ᴸᴹ, ᴸᴸ
  For falling-rising and rising-falling tones, in principle, only the starting point (onset) or ending point (offset) is taken. For example, Mandarin's falling-rising tone ("214") is generally considered to have its distinctive feature manifested mainly in the onset, so it is treated as "21" and marked as "ᴹᴸ".
  Examples:
  Mandarin Four Tones: High Level "55" → "ᴴᴴ", Rising "35" → "ᴹᴴ", Falling-Rising "214" → (taking onset "21") "ᴹᴸ", Falling "51" → "ᴴᴸ"
  The tone mark is placed immediately after the phoneme acting as the syllable peak, without a space. The peak phoneme must be preceded by the stress mark "ˈ". Neutral tones are not marked, and the peak phoneme in a neutral-tone syllable is not preceded by the stress mark.
The multi-dialect Chinese phonetic annotation schemes developed for this project according to the above rules are provided in: [speech_annotation_scheme](./speech_annotation_scheme.xlsx)

