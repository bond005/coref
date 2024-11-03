from argparse import ArgumentParser
import codecs
import copy
import csv
import os
import warnings

from tqdm import tqdm

from io_utils.io_utils import load_rucoco
from linguistic_utils.linguistic_utils import initialize_nlp
from linguistic_utils.linguistic_utils import find_token_by_character_index
from linguistic_utils.linguistic_utils import inflect_subphrase, parse_text, find_main_token
from linguistic_utils.linguistic_utils import get_case_and_number


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The path to the input RuCoCo.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The path to the output HF-formatted dataset.')
    args = parser.parse_args()

    input_dataset_path = os.path.normpath(args.input_name)
    if not os.path.isdir(input_dataset_path):
        raise IOError(f'The directory "{input_dataset_path}" does not exist!')

    output_dataset_path = os.path.normpath(args.output_name)
    if not os.path.isdir(output_dataset_path):
        base_dir = os.path.dirname(output_dataset_path)
        if len(base_dir) > 0:
            if not os.path.isdir(base_dir):
                raise IOError(f'The directory "{base_dir}" does not exist!')
        os.mkdir(output_dataset_path)

    nlp, morph = initialize_nlp()
    print('The NLP subsystem is initialized.')

    source_data = load_rucoco(input_dataset_path)
    print(f'There are {len(source_data)} samples are loaded from {input_dataset_path}.')

    n_rows = 0
    with codecs.open(os.path.join(output_dataset_path, 'train_data.csv'), mode='w', encoding='utf-8') as fp:
        data_writer = csv.writer(fp, delimiter=',', quotechar='"')
        data_writer.writerow(['source_text', 'text_without_coreference'])
        for text, coreference_chains in tqdm(source_data):
            tokens = parse_text(text, nlp, morph)
            substitutions = dict()
            is_valid_sample = True
            for cur_chain in coreference_chains:
                main_item = copy.copy(cur_chain[0])
                for it in cur_chain[1:]:
                    if (it[1] - it[0]) > (main_item[1] - main_item[0]):
                        main_item = copy.copy(it)
                main_item = (
                    find_token_by_character_index(tokens, main_item[0]),
                    find_token_by_character_index(tokens, main_item[1] - 1) + 1
                )
                err_msg = f'The coreference chain {cur_chain} does not correspond to the text {text}'
                if (main_item[0] < 0) or (main_item[1] <= 0):
                    raise ValueError(err_msg)
                prepared_chain = []
                for it in cur_chain:
                    prepared_chain.append((
                        find_token_by_character_index(tokens, it[0]),
                        find_token_by_character_index(tokens, it[1] - 1) + 1
                    ))
                    if (prepared_chain[-1][0] < 0) or (prepared_chain[-1][1] <= 0):
                        raise ValueError(err_msg)
                source_case, source_number = get_case_and_number(text, tokens, main_item[0], main_item[1], nlp)
                if (len(source_number) == 0) or (len(source_case) == 0):
                    is_valid_sample = False
                    warn_msg = f'The phrase has not a root. {tokens[main_item[0]:main_item[1]]}'
                    warnings.warn(warn_msg)
                    break
                substitutions_for_cur_chain = []
                for it in prepared_chain:
                    if it == main_item:
                        continue
                    target_case, target_number = get_case_and_number(text, tokens, it[0], it[1], nlp)
                    if (len(target_number) == 0) and (len(target_case) == 0):
                        is_valid_sample = False
                        warn_msg = f'The phrase has not a root. {tokens[it[0]:it[1]]}'
                        warnings.warn(warn_msg)
                        break
                    inflected_phrase, ok = inflect_subphrase(text, tokens, main_item[0], main_item[1], nlp,
                                                             target_case, target_number)
                    if not ok:
                        is_valid_sample = False
                        break
                    substitutions_for_cur_chain.append((it[0], it[1], inflected_phrase))
                if not is_valid_sample:
                    break
                for token_start, token_end, inflected_phrase in substitutions_for_cur_chain:
                    substitutions[token_start] = (token_end, inflected_phrase)
                del substitutions_for_cur_chain
            if is_valid_sample:
                decoded_text = ''
                prev_token_end = 0
                token_idx = 0
                while token_idx < len(tokens):
                    n_spaces = tokens[token_idx][0] - prev_token_end
                    while n_spaces > 0:
                        decoded_text += ' '
                        n_spaces -= 1
                    if token_idx in substitutions:
                        decoded_text += substitutions[token_idx][1]
                        token_idx = substitutions[token_idx][0]
                    else:
                        decoded_text += text[tokens[token_idx][0]:tokens[token_idx][1]]
                        token_idx += 1
                    prev_token_end = tokens[token_idx - 1][1]
                data_writer.writerow([text, decoded_text])
                n_rows += 1
    print(f'There are {n_rows} are written into the "{os.path.join(output_dataset_path, "train_data.csv")}".')


if __name__ == '__main__':
    main()
