from argparse import ArgumentParser
import codecs
import csv
import os
from typing import List, Tuple
import warnings

from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedTokenizer, GenerationMixin

from io_utils.io_utils import load_rucoco


SYSTEM_PROMPT: str = ('Представь себя опытным филологом, знатоком русского языка, и исправь, пожалуйста, '
                      'грамматические ошибки текста, связанные с несогласованностью падежей. '
                      'Я приведу тебе несколько примеров исходных текстов с ошибками и соответствующих им '
                      'исправленных текстов. Когда будешь писать свой ответ, руководствуйся этими примерами. '
                      'В качестве ответа просто напиши исправленный текст, без пояснений и других комментариев.')
EXAMPLES: List[Tuple[str, str]] = [
    (
        'Недалеко от Биробиджана обнаружена таежная плантация конопли\n\nВ Еврейском автономном округе, недалеко от '
        'Биробиджана, обнаружена таежная плантация конопли. База для изготовления всевозможных продуктов из этого '
        'растения и лагерь для жилья работников разместились в труднодоступном и хорошо замаскированном '
        'таежном месте. труднодоступном и хорошо замаскированном таежном месте же был обнаружен и семенной фонд '
        'будущих посевов.\n\nКак сообщил дальневосточный корреспондент НТВ, на таежная плантация конопли "можно '
        'было заблудиться" - площадь таежная плантация конопли составляет несколько сот квадратных метров. '
        'Высота травы - два метра и более. таежная плантация конопли работала по замкнутому циклу: '
        'посев, выращивание, сушка, получение готовой продукции, заготовка семян и снова посев.\n\n'
        'Самих "плантаторов" в лагерь для жилья работников не оказалось, и кто Самих "плантаторов" пока не '
        'известно. Задержаны только два сторожа. два сторожа оказались хорошо вооружены, но применить оружие '
        'не успели. В момент задержания два сторожа были заняты изготовлением гашишного масла на кустарной установке, '
        'напоминающей самогонный аппарат.\n\nПо словам начальника пресс-центра Дальневосточного РУБОП '
        'Вячеслава Суздальцева, подобные операции проводятся ежегодно, так как Еврейском автономном округе, '
        'наряду с Амурской областью и Хабаровским краем, "является регионом естественного распространения конопли". '
        '"Однако то, что в данном случае было обнаружено, поразило даже видавших виды оперативников", '
        'сказал начальника пресс-центра Дальневосточного РУБОП Вячеслава Суздальцева в интервью НТВ.\n\n'
        'Самих "плантаторов" успели снять урожай лишь с небольшой части Самих "плантаторов"таежная плантация '
        'конопли. Оперативники приняли решение уничтожить и готовую продукцию, и таежная плантация конопли. '
        'База для изготовления всевозможных продуктов из этого растения и лагерь для жилья работников сожгли, '
        'а таежная плантация конопли погибла под гусеницами вездехода.',  # input
        'Недалеко от Биробиджана обнаружена таежная плантация конопли\n\nВ Еврейском автономном округе, недалеко от '
        'Биробиджана, обнаружена таежная плантация конопли. База для изготовления всевозможных продуктов из этого '
        'растения и лагерь для жилья работников разместились в труднодоступном и хорошо замаскированном '
        'таежном месте. В труднодоступном и хорошо замаскированном таежном месте же был обнаружен и семенной фонд '
        'будущих посевов.\n\nКак сообщил дальневосточный корреспондент НТВ, на таежной плантации конопли '
        '"можно было заблудиться" - площадь таежной плантации конопли составляет несколько сот квадратных метров. '
        'Высота травы - два метра и более. Таежная плантация конопли работала по замкнутому циклу: '
        'посев, выращивание, сушка, получение готовой продукции, заготовка семян и снова посев.\n\n'
        'Самих "плантаторов" в лагерь для жилья работников не оказалось, и кто сами "плантаторы" пока не известно. '
        'Задержаны только два сторожа. Два сторожа оказались хорошо вооружены, но применить оружие не успели. '
        'В момент задержания два сторожа были заняты изготовлением гашишного масла на кустарной установке, '
        'напоминающей самогонный аппарат.\n\nПо словам начальника пресс-центра Дальневосточного РУБОП '
        'Вячеслава Суздальцева, подобные операции проводятся ежегодно, так как Еврейский автономный округ, '
        'наряду с Амурской областью и Хабаровским краем, "является регионом естественного распространения конопли". '
        '"Однако то, что в данном случае было обнаружено, поразило даже видавших виды оперативников", '
        'сказал начальник пресс-центра Дальневосточного РУБОП Вячеслав Суздальцев в интервью НТВ.\n\n'
        'Сами "плантаторы" успели снять урожай лишь с небольшой части таежной плантации конопли самих "плантаторов". '
        'Оперативники приняли решение уничтожить и готовую продукцию, и таежную плантацию конопли. '
        'База для изготовления всевозможных продуктов из этого растения и лагерь для жилья работников сожгли, '
        'а таежная плантация конопли погибла под гусеницами вездехода.'  # target
    ),
    (
        'Российский боксер-супертяжеловес Олег Маскаев проиграл в Нью-Йорке бой канадцу Кирку Джонсону\n\n'
        'В четвертом раунде канадцу Кирку Джонсону нанес Российский боксер-супертяжеловес Олег Маскаев '
        'сокрушительный удар, после сокрушительный удар Российский боксер-супертяжеловес Олег Маскаев не смог '
        'подняться на ноги.\n\nДо этого поединка Российский боксер-супертяжеловес Олег Маскаев ни разу не проигрывал '
        'на профессиональном ринге и в случае победы мог бросить вызов чемпиону мира по версии WBA - американцу '
        'Эвандеру Холифилду.\n\nОднако с самого начала боя преимуществом владел канадцу Кирку Джонсону. '
        'канадцу Кирку Джонсону трижды посылал Российский боксер-супертяжеловес Олег Маскаев в нокдаун, '
        'причем в третий раз Российский боксер-супертяжеловес Олег Маскаев вылетел за пределы ринга. '
        'Для канадцу Кирку Джонсону нынешняя победа стала 31-й в карьере, причем 23 из них канадцу Кирку Джонсону '
        'одержал нокаутом.',  # input
        'Российский боксер-супертяжеловес Олег Маскаев проиграл в Нью-Йорке бой канадцу Кирку Джонсону\n\n'
        'В четвертом раунде канадец Кирк Джонсон нанес российскому боксеру-супертяжеловесу Олегу Маскаеву '
        'сокрушительный удар, после сокрушительного удара российский боксер-супертяжеловес Олег Маскаев не смог '
        'подняться на ноги.\n\nДо этого поединка российский боксер-супертяжеловес Олег Маскаев ни разу не проигрывал '
        'на профессиональном ринге и в случае победы мог бросить вызов чемпиону мира по версии WBA - американцу '
        'Эвандеру Холифилду.\n\nОднако с самого начала боя преимуществом владел канадец Кирк Джонсон. '
        'Канадец Кирк Джонсон трижды посылал российского боксера-супертяжеловеса Олега Маскаева в нокдаун, '
        'причем в третий раз российский боксер-супертяжеловес Олег Маскаев вылетел за пределы ринга. '
        'Для канадца Кирка Джонсона нынешняя победа стала 31-й в карьере, причем 23 из них канадец Кирк Джонсон '
        'одержал нокаутом.',  # target
    ),
    (
        'Федеральные войска проводят в Грозном масштабную спецоперацию\n\nКак сообщает "Интерфакс", в Грозном '
        'Федеральные войска проводят масштабную спецоперацию. Жизнь в Грозном практически парализована действиями '
        'Федеральные войска. Жители опасаются, что командование группировки в Чечне может отдать приказ о проведении '
        'зачистки "по примеру Серноводска и Ассиновской". Сейчас Грозном блокирован по секторам. О целях '
        'этой операции командование группировки в Чечне не сообщает местным властям, отметили собеседники "Интерфакс" '
        'в мэрии Грозном.',  # input
        'Федеральные войска проводят в Грозном масштабную спецоперацию\n\nКак сообщает "Интерфакс", в Грозном '
        'федеральные войска проводят масштабную спецоперацию. Жизнь в Грозном практически парализована действиями '
        'федеральных войск. Жители опасаются, что командование группировки в Чечне может отдать приказ о проведении '
        'зачистки "по примеру Серноводска и Ассиновской". Сейчас Грозный блокирован по секторам. О целях '
        'этой операции командование группировки в Чечне не сообщает местным властям, отметили собеседники "Интерфакса" '
        'в мэрии Грозного.'  # target
    ),
    (
        'Для Россией открывается уникальная возможность развития рынка программного обеспечения\n\n'
        'Газета Financial Times опубликовала большую статью, посвященную перспективам развития '
        'рынка программного обеспечения в Россией. Объем этого глобального рынка должен составить '
        'один триллион долларов к 2008 году по сравнению с 327 миллиардами в 1997 году.\n\n'
        'В настоящий момент лидером в развитии оффшорных зон, в оффшорных зон создаются компании, '
        'производящие программы для компьютерной отрасли является Индией, сообщает BBC. ИндиейЕе рынок '
        'оценивается в 6,2 миллиарда долларов, и на Ее рынок создание понадобилось 20 лет правительственной '
        'поддержки. По сравнению с Индией, объем этого сектора рынка в Россией составляет пока что менее '
        '150 миллионов долларов.\n\nСейчас для Россией открывается уникальная историческая возможность. '
        'Во всем мире растет неудовлетворенный спрос на программистов - только в США, по одной из оценок, '
        'имеется 800 тысяч вакансий. В Россией в год готовится до 100 тысяч дипломированных программистов.\n\n'
        'В качестве примера использования таких возможностей Газета Financial Times приводит Петербург, '
        'Петербург четыре университета готовят полторы тысячи программистов в год. Как пишет Газета Financial Times, '
        'эти специалисты обладают высочайшим уровнем профессиональной и теоретической подготовки, хотя '
        'о коммерческой стороне дела знают меньше.\n\nРоссийский специалист, работающий в местной фирме, '
        'выполняющей международные заказы, получает около 5-7 тысяч долларов в год по сравнению с 35-60 тысячами, '
        '35-60 тысячами Российский специалист платили бы в Европе или в США. Поэтому разработка программного '
        'обеспечения в Россией обходится на 70% дешевле, чем на Западе.\n\nОсновными условиями успеха российской '
        'промышленности программного обеспечения является, по мнению Газета Financial Times, принятие '
        'правительством Россией осмысленной коммерческой стратегии. Пока что такие центры этой отрасли, '
        'как Петербург, развивались стихийно, без правительственной поддержки. Кроме того, по мнению '
        'Газета Financial Times, необходимо создание законодательной базы, и, прежде всего, соблюдение законов '
        'по защите интеллектуальной собственности.\n\nЕсли эта историческая возможность не будет в течение '
        'нескольких ближайших лет использована Россией, эта историческая возможность воспользуются другие, '
        'заключает Газета Financial Times. В Польше и Венгрии усиленными темпами развиваются программы '
        'подготовки специалистов, а на очереди - Китай.',  # input
        'Для России открывается уникальная возможность развития рынка программного обеспечения\n\n'
        'Газета Financial Times опубликовала большую статью, посвященную перспективам развития рынка программного '
        'обеспечения в России. Объем этого глобального рынка должен составить один триллион долларов к 2008 году '
        'по сравнению с 327 миллиардами в 1997 году.\n\nВ настоящий момент лидером в развитии оффшорных зон, '
        'в оффшорных зонах создаются компании, производящие программы для компьютерной отрасли, является Индия, '
        'сообщает BBC. Рынок Индии оценивается в 6,2 миллиарда долларов, и на создание рынка Индии понадобилось '
        '20 лет правительственной поддержки. По сравнению с Индией, объем этого сектора рынка в России составляет '
        'пока что менее 150 миллионов долларов.\n\nСейчас для России открывается уникальная историческая возможность. '
        'Во всем мире растет неудовлетворенный спрос на программистов - только в США, по одной из оценок, имеется '
        '800 тысяч вакансий. В России в год готовится до 100 тысяч дипломированных программистов.\n\n'
        'В качестве примера использования таких возможностей газета Financial Times приводит Петербург, '
        'в Петербурге четыре университета готовят полторы тысячи программистов в год. Как пишет газета '
        'Financial Times, эти специалисты обладают высочайшим уровнем профессиональной и теоретической подготовки, '
        'хотя о коммерческой стороне дела знают меньше.\n\nРоссийский специалист, работающий в местной фирме, '
        'выполняющей международные заказы, получает около 5-7 тысяч долларов в год по сравнению с 35-60 тысячами, '
        '35-60 тысяч российскому специалисту платили бы в Европе или в США. Поэтому разработка программного '
        'обеспечения в России обходится на 70% дешевле, чем на Западе.\n\nОсновными условиями успеха российской '
        'промышленности программного обеспечения является, по мнению газеты Financial Times, принятие '
        'правительством России осмысленной коммерческой стратегии. Пока что такие центры этой отрасли, как Петербург, '
        'развивались стихийно, без правительственной поддержки. Кроме того, по мнению газеты Financial Times, '
        'необходимо создание законодательной базы, и, прежде всего, соблюдение законов по защите интеллектуальной '
        'собственности.\n\nЕсли эта историческая возможность не будет в течение нескольких ближайших лет '
        'использована Россией, этой исторической возможностью воспользуются другие, заключает газета Financial Times. '
        'В Польше и Венгрии усиленными темпами развиваются программы подготовки специалистов, '
        'а на очереди - Китай.'  # target
    )
]


def correct_text(source_text: str, tokenizer: PreTrainedTokenizer, model: GenerationMixin, device: str) -> str:
    messages = [
        {
            'role': 'system', 'content': SYSTEM_PROMPT
        }
    ]
    for input_text, true_output in EXAMPLES:
        messages += [
            {
                'role': 'user', 'content': input_text
            },
            {
                'role': 'assistant', 'content': true_output
            }
        ]
    messages.append({'role': 'user', 'content': source_text})
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors='pt').to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max(10, 2 * len(tokenizer.tokenize(source_text)))
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The path to the input RuCoCo.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The path to the output HF-formatted dataset.')
    parser.add_argument('-m', '--model', dest='large_language_model', type=str, required=True,
                        help='The large language model for text correction.')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    device = 'cuda:0'

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

    tokenizer = AutoTokenizer.from_pretrained(args.large_language_model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.large_language_model, torch_dtype=torch.bfloat16).to(device)
    except:
        model = AutoModelForCausalLM.from_pretrained(args.large_language_model, torch_dtype=torch.float16).to(device)
    model.eval()
    print(f'LLM is loaded from {args.large_language_model}.')

    source_data = load_rucoco(input_dataset_path)
    print(f'There are {len(source_data)} samples are loaded from {input_dataset_path}.')

    n_rows = 0
    output_fname = os.path.join(output_dataset_path, 'train_data.csv')
    with codecs.open(output_fname, mode='w', encoding='utf-8', buffering=0) as fp:
        data_writer = csv.writer(fp, delimiter=',', quotechar='"')
        data_writer.writerow(['source_text', 'text_without_coreference'])
        for sample_idx, (text, coreference_chains) in enumerate(tqdm(source_data)):
            found_idx = text.find('Источник: ')
            if found_idx >= 0:
                prepared_text = text[:found_idx].rstrip()
            else:
                prepared_text = text
            is_valid = True
            for cur_chain in coreference_chains:
                for entity_start, entity_end in cur_chain:
                    if entity_end > len(prepared_text):
                        is_valid = False
                        break
                if not is_valid:
                    break
            if is_valid:
                substitutions = []
                for cur_chain in coreference_chains:
                    entities = []
                    for entity_start, entity_end in cur_chain:
                        entities.append(prepared_text[entity_start:entity_end])
                    if len(entities) < 2:
                        is_valid = False
                    else:
                        entities.sort(key=lambda it: (-len(it), it))
                        main_entity = ''
                        for cur_entity in entities:
                            if (len(cur_entity) > 1) and cur_entity.isupper():
                                main_entity = cur_entity
                                break
                        if len(main_entity) == 0:
                            main_entity = entities[0]
                        if len(main_entity) < 2:
                            warnings.warn(f'Main entity in the sample {sample_idx} is not found.')
                            is_valid = False
                        else:
                            for entity_start, entity_end in cur_chain:
                                substitutions.append((entity_start, entity_end, main_entity))
                    del entities
                    if not is_valid:
                        break
                if is_valid:
                    if len(substitutions) > 0:
                        substitutions.sort(key=lambda it: (it[0], it[1], len(it[2])))
                        filtered_substitutions = []
                        filled = [0 for _ in range(len(prepared_text))]
                        for entity_start, entity_end, entity_text in substitutions:
                            ok = True
                            for char_idx in range(entity_start, entity_end):
                                if filled[char_idx] != 0:
                                    ok = False
                                    break
                            if ok:
                                for char_idx in range(entity_start, entity_end):
                                    filled[char_idx] = 1
                                filtered_substitutions.append((entity_start, entity_end, entity_text))
                        if len(filtered_substitutions) < 2:
                            is_valid = False
                        if is_valid:
                            n_rows += 1
                            new_text = prepared_text[0:substitutions[0][0]]
                            new_text += substitutions[0][2]
                            prev_entity_end = substitutions[0][1]
                            for entity_idx in range(1, len(substitutions)):
                                cur_entity_start = substitutions[entity_idx][0]
                                cur_entity_end = substitutions[entity_idx][1]
                                new_text += prepared_text[prev_entity_end:cur_entity_start]
                                new_text += substitutions[entity_idx][2]
                                prev_entity_end = cur_entity_end
                            new_text += prepared_text[substitutions[-1][1]:]
                            corrected_text = correct_text(new_text.strip(), tokenizer, model, device)
                            data_writer.writerow([prepared_text.strip(), corrected_text.strip()])
                        else:
                            warnings.warn(f'Some entities in the sample {sample_idx} are overlapped.')
                del substitutions
            else:
                warnings.warn(f'Some entities in the sample {sample_idx} have a wrong bounds.')
    print(f'There are {n_rows} are written into the "{output_fname}".')


if __name__ == '__main__':
    main()
