from typing import Dict, List, Tuple

import spacy
from pymorphy3.analyzer import Parse
from pymorphy3.tagset import OpencorporaTag
from pymorphy3 import MorphAnalyzer


POS_DICT: Dict[str, str] = {
    'NOUN': 'NOUN',
    'ADJ': 'ADJF',
    'PRON': 'NPRO',
    'DET': 'NPRO',
    'PROPN': 'NOUN',
    'VERB': 'VERB',
    'NUM': 'NUMR',
    'AUX': 'PRCL',
    'PART': 'PRCL',
    'CCONJ': 'CCONJ',
    'ADV': 'ADVB',
    'INTJ': 'INTJ',
    'ADP': 'PREP',
    'SCONJ': 'CCONJ'
}

CASE_DICT: Dict[str, str] = {
    'Acc': 'accs',
    'Dat': 'datv',
    'Gen': 'gent',
    'Ins': 'ablt',
    'Loc': 'loct',
    'Nom': 'nomn',
    'Par': 'gen2',
    'Voc': 'voct'
}

NUMBER_DICT: Dict[str, str] = {
    'Sing': 'sing',
    'Plur': 'plur'
}


def initialize_nlp() -> Tuple[spacy.Language, MorphAnalyzer]:
    return spacy.load('ru_core_news_lg'), MorphAnalyzer()


def check_grammeme(grammeme: str, tag: OpencorporaTag) -> int:
    try:
        res = 1 if (grammeme in tag) else 0
    except:
        res = 0
    return res


def find_best_parsing(variants_of_parsing: List[Parse], pos: str, case: str, number: str) -> Parse:
    best_variant = variants_of_parsing[0]
    best_sim = check_grammeme(pos, best_variant.tag)
    best_sim += check_grammeme(case, best_variant.tag)
    best_sim += check_grammeme(number, best_variant.tag)
    for cur_variant in variants_of_parsing[1:]:
        cur_sim = check_grammeme(pos, cur_variant.tag)
        cur_sim += check_grammeme(case, cur_variant.tag)
        cur_sim += check_grammeme(number, cur_variant.tag)
        if cur_sim > best_sim:
            best_sim = cur_sim
            best_variant = cur_variant
    return best_variant


def parse_text(text: str, nlp: spacy.Language, morph: MorphAnalyzer) -> List[Tuple[int, int, Parse]]:
    doc = nlp(text)
    parsed = []
    for token in doc:
        pos = POS_DICT.get(str(token.pos_), str(token.pos_))
        case = token.morph.get('Case')
        if len(case) > 0:
            if len(case[0]) > 0:
                case = CASE_DICT[str(case[0])]
            else:
                case = ''
        else:
            case = ''
        number = token.morph.get('Number')
        if len(number) > 0:
            if len(number[0]) > 0:
                number = NUMBER_DICT[str(number[0])]
            else:
                number = ''
        else:
            number = ''
        parsed.append(
            (
                token.idx,
                token.idx + len(token.text),
                find_best_parsing(morph.parse(token.text), pos, case, number)
            )
        )
    del doc
    return parsed


def find_main_token(phrase: str, nlp: spacy.Language) -> Tuple[int, bool]:
    doc = nlp(phrase)
    if len(doc) < 2:
        return 0, (doc[0].pos_ in {'NOUN', 'NPRO'})
    found_idx = -1
    is_noun = False
    for idx, val in enumerate(doc):
        if val.dep_ == 'ROOT':
            found_idx = idx
            if val.pos_ == 'NOUN':
                is_noun = True
            break
    return found_idx, is_noun


def get_case_and_number(full_text: str, tokens: List[Tuple[int, int, Parse]], subphrase_start: int, subphrase_end: int,
                        nlp: spacy.Language) -> Tuple[str, str]:
    source_subphrase = full_text[tokens[subphrase_start][0]:tokens[subphrase_start][1]]
    for token_index in range(subphrase_start + 1, subphrase_end):
        n_spaces = tokens[token_index][0] - tokens[token_index - 1][1]
        while n_spaces > 0:
            source_subphrase += ' '
            n_spaces -= 1
        source_subphrase += full_text[tokens[token_index][0]:tokens[token_index][1]]
    main_token_index, is_noun = find_main_token(source_subphrase, nlp)
    source_case = tokens[main_token_index + subphrase_start][2].tag.case
    source_number = tokens[main_token_index + subphrase_start][2].tag.number
    if source_number is None:
        source_number = ''
    if source_case is None:
        source_case = ''
    return source_case, source_number


def inflect_word(word: Parse, target_case: str, target_number: str) -> str:
    if (len(target_number) > 0) and (len(target_case) > 0):
        try:
            inflected = word.inflect({target_case, target_number})
        except:
            raise RuntimeError(f'{word} cannot be inflected with case = {target_case} and number = {target_number}.')
    elif len(target_number) > 0:
        try:
            inflected = word.inflect({target_number})
        except:
            raise RuntimeError(f'{word} cannot be inflected with case = {target_case} and number = {target_number}.')
    elif len(target_case) > 0:
        try:
            inflected = word.inflect({target_case})
        except:
            raise RuntimeError(f'{word} cannot be inflected with case = {target_case} and number = {target_number}.')
    else:
        inflected = None
    if inflected is None:
        return word.word
    return inflected.word


def find_token_by_character_index(tokens: List[Tuple[int, int, Parse]], char_idx: int) -> int:
    found_idx = -1
    for idx, (start_pos, end_pos, _) in enumerate(tokens):
        if (char_idx >= start_pos) and (char_idx < end_pos):
            found_idx = idx
            break
    return found_idx


def inflect_subphrase(full_text: str, tokens: List[Tuple[int, int, Parse]], subphrase_start: int, subphrase_end: int,
                      nlp: spacy.Language, target_case: str, target_number: str) -> Tuple[str, bool]:
    source_subphrase = full_text[tokens[subphrase_start][0]:tokens[subphrase_start][1]]
    for token_index in range(subphrase_start + 1, subphrase_end):
        n_spaces = tokens[token_index][0] - tokens[token_index - 1][1]
        while n_spaces > 0:
            source_subphrase += ' '
            n_spaces -= 1
        source_subphrase += full_text[tokens[token_index][0]:tokens[token_index][1]]
    main_token_index, is_noun = find_main_token(source_subphrase, nlp)
    if not is_noun:
        return source_subphrase, False
    inflected_subphrase = inflect_word(tokens[subphrase_start][2], target_case, target_number)
    if full_text[tokens[subphrase_start][0]:tokens[subphrase_start][1]].isupper():
        inflected_subphrase = inflected_subphrase.upper()
    elif full_text[tokens[subphrase_start][0]:tokens[subphrase_start][1]].istitle():
        inflected_subphrase = inflected_subphrase.title()
    for token_index in range(subphrase_start + 1, subphrase_end):
        n_spaces = tokens[token_index][0] - tokens[token_index - 1][1]
        while n_spaces > 0:
            inflected_subphrase += ' '
            n_spaces -= 1
        if (token_index - subphrase_start) <= main_token_index:
            new_word = inflect_word(tokens[token_index][2], target_case, target_number)
            old_word = full_text[tokens[token_index][0]:tokens[token_index][1]]
            if old_word.isupper():
                new_word = new_word.upper()
            elif old_word.istitle():
                new_word = new_word.title()
        else:
            new_word = full_text[tokens[token_index][0]:tokens[token_index][1]]
        inflected_subphrase += new_word
    return inflected_subphrase, True
