import codecs
import json
import os
from typing import List, Tuple
import warnings


def find_entity(bounds_of_entities: List[Tuple[int, int]], char_idx: int) -> int:
    found_idx = -1
    for idx, (start_pos, end_pos) in enumerate(bounds_of_entities):
        if (char_idx >= start_pos) and (char_idx < end_pos):
            found_idx = idx
            break
    return found_idx


def load_rucoco(dataset_dir: str) -> List[Tuple[str, List[List[Tuple[int, int]]]]]:
    if not os.path.isdir(dataset_dir):
        raise IOError(f'The directory "{dataset_dir}" does not exist!')
    data_files = list(map(
        lambda it2: os.path.join(dataset_dir, it2),
        filter(
            lambda it1: it1.lower().endswith('.json'),
            os.listdir(dataset_dir)
        )
    ))
    if len(data_files) == 0:
        raise IOError(f'The directory "{dataset_dir}" is empty!')
    data_samples = []
    for cur_fname in data_files:
        with codecs.open(cur_fname, mode='r', encoding='utf-8') as fp:
            sample = json.load(fp)
        err_msg = f'The file "{cur_fname}" contains a wrong data!'
        if not isinstance(sample, dict):
            raise IOError(err_msg + f' Expected {type({"a": "b"})}, got {type(sample)}.')
        if 'text' not in sample:
            raise IOError(err_msg + f' The "text" field is not found.')
        if 'entities' not in sample:
            raise IOError(err_msg + f' The "entities" field is not found.')
        full_text = sample['text']
        if not isinstance(full_text, str):
            err_msg += f' The "text" field is wrong. Expected {type("1")}, got {type(full_text)}.'
            raise IOError(err_msg)
        coreference_chains = sample['entities']
        if not isinstance(coreference_chains, list):
            err_msg += f' The "entities" field is wrong. Expected {type(["1", "2"])}, got {type(coreference_chains)}.'
            raise IOError(err_msg)
        prepared_coreference_chains = []
        for cur_chain in coreference_chains:
            if not isinstance(cur_chain, list):
                err_msg += f' The "entities" field is wrong.'
                raise IOError(err_msg)
            new_chain = []
            for it in cur_chain:
                if not isinstance(it, list):
                    err_msg += f' The "entities" field is wrong.'
                    raise IOError(err_msg)
                if len(it) != 2:
                    err_msg += f' The "entities" field is wrong.'
                    raise IOError(err_msg)
                if (not isinstance(it[0], int)) or (not isinstance(it[1], int)):
                    err_msg += f' The "entities" field is wrong.'
                    raise IOError(err_msg)
                if (it[0] < 0) or (it[1] >= len(full_text)) or (it[0] >= it[1]):
                    err_msg += f' The "entities" field is wrong.'
                    raise IOError(err_msg)
                exists = False
                for char_idx in range(it[0], it[1]):
                    if find_entity(new_chain, char_idx) >= 0:
                        exists = True
                        break
                if not exists:
                    new_chain.append((it[0], it[1]))
            if len(new_chain) == 0:
                err_msg += f' The "entities" field is wrong.'
                raise IOError(err_msg)
            prepared_coreference_chains.append(sorted(new_chain))
            del new_chain
        if len(prepared_coreference_chains) == 0:
            err_msg += ' The "entities" field is empty.'
            raise IOError(err_msg)
        filled_chars = [0 for _ in range(len(full_text))]
        bad_entity = ''
        for cur_chain in prepared_coreference_chains:
            for start_char_pos, end_char_pos in cur_chain:
                for char_idx in range(start_char_pos, end_char_pos):
                    if filled_chars[char_idx] != 0:
                        bad_entity = f'{(start_char_pos, end_char_pos)}'
                        break
                    filled_chars[char_idx] = 1
                if len(bad_entity) > 0:
                    break
            if len(bad_entity) > 0:
                break
        if len(bad_entity) > 0:
            warn_msg = err_msg +  (f' The "entities" field is wrong. Entity {bad_entity} is overlapped. '
                                   f'{prepared_coreference_chains}')
            warnings.warn(warn_msg)
        else:
            data_samples.append((full_text, prepared_coreference_chains))
        del filled_chars
    return data_samples
