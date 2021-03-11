# encoding: utf8

import os
import json
import copy
from tqdm import tqdm
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                          do_lower_case=True)
rel2id = json.load(open('dataset/rel2id.json', 'r'))
fact_in_train = set()
span_wrong_dict = set()


def convert_feature(file_name, output_file, max_seq_length=512, is_training=True, is_test=False, debug=False):
    i_line = 0
    max_len_for_doc = max_seq_length - 2  # [CLS] [SEP]

    pos_samples = 0
    neg_samples = 0

    print('convert features...')
    with open(output_file, 'w') as w:
        with open(file_name, 'r') as f:
            data_samples = json.load(f)
            for sample in tqdm(data_samples):
                if not is_test:
                    # 训练集、开发集
                    labels = sample['labels']

                # 外面先wordpiece分词,映射每句的word index
                sents = []
                sent_map = []
                for sent in sample['sents']:
                    new_sent = []
                    new_map = {}    # 索引为句子token index
                    for i_t, token in enumerate(sent):
                        tokens_wordpiece = tokenizer.tokenize(token)
                        new_map[i_t] = len(new_sent)
                        new_sent.extend(tokens_wordpiece)
                    new_map[i_t + 1] = len(new_sent)
                    sent_map.append(new_map)
                    sents.append(new_sent)

                entitys = sample['vertexSet']

                # 先存储有relation的实体关系
                train_triple = {}
                if not is_test:
                    # 训练集、开发集

                    for label in labels:
                        evidence = label['evidence']
                        r = int(rel2id[label['r']])

                        # 由于同一组实体可能存在多个关系，这里要用list存！
                        if (label['h'], label['t']) not in train_triple:
                            # 添加关系三元组
                            train_triple[(label['h'], label['t'])] = [{'relation': r, 'evidence': evidence}]

                        else:  # 不过要确保他们的关系是不同的
                            in_triple = False
                            for tmp_r in train_triple[(label['h'], label['t'])]:
                                if tmp_r['relation'] == r:
                                    in_triple = True
                                    break
                            if not in_triple:
                                train_triple[(label['h'], label['t'])].append({'relation': r, 'evidence': evidence})

                        intrain = False
                        # 登记哪些实体关系在train中出现过了
                        for e1i in entitys[label['h']]:         # 头实体
                            for e2i in entitys[label['t']]:     # 尾实体
                                if is_training:
                                    # 训练集
                                    fact_in_train.add((e1i['name'], e2i['name'], r))
                                elif not is_test:
                                    # 验证集
                                    if (e1i['name'], e2i['name'], r) in fact_in_train:
                                        for train_tmp in train_triple[(label['h'], label['t'])]:
                                            train_tmp['intrain'] = True
                                        intrain = True

                        if not intrain:
                            for train_tmp in train_triple[(label['h'], label['t'])]:
                                train_tmp['intrain'] = False

                # 遍历所有实体构建关系，没有关系的打上NA
                for e1, entity1 in enumerate(entitys):
                    for e2, entity2 in enumerate(entitys):
                        if e1 != e2:
                            # 在所有实体1前后加上[unused0]和[unused1]用来给实体定位,在所有实体2前后加上[unused2]和[unused3]用来给实体定位
                            # [unused0] Hirabai Badodekar [unused1] , Gangubai Hangal , Mogubai Kurdikar ) ,
                            # made the [unused2] Indian [unused3] classical music so much greater .

                            entity1_ = copy.deepcopy(entity1)
                            entity2_ = copy.deepcopy(entity2)
                            for e in entity1_:
                                e['first'] = True  # 是entity1
                            for e in entity2_:
                                e['first'] = False  # 是entity2
                            new_sents = copy.deepcopy(sents)

                            # TODO 添加在 new_sents 里面加入 [unused0] [unused1] [unused2] [unused3] 的代码

                            doc_tokens = []
                            for sent in new_sents:
                                doc_tokens.extend(sent)

                            if len(doc_tokens) > max_len_for_doc:
                                continue

                            tokens = ['[CLS]'] + doc_tokens + ['[SEP]']
                            input_ids = tokenizer.convert_tokens_to_ids(tokens)
                            segment_ids = [0] * len(input_ids)
                            input_mask = [1] * len(input_ids)

                            intrain = None
                            relation_label = None
                            evidence = []
                            if not is_test:
                                # 训练集、验证集
                                if (e1, e2) not in train_triple:
                                    relation_label = [0] * len(rel2id)
                                    relation_label[0] = 1
                                    evidence = []
                                    intrain = False
                                    neg_samples += 1
                                else:
                                    relation_label = [0] * len(rel2id)
                                    # 一个实体可能存在多个关系
                                    for train_tmp in train_triple[(e1, e2)]:
                                        relation_label[train_tmp['relation']] = 1
                                        evidence.append(train_tmp['evidence'])
                                    intrain = train_triple[(e1, e2)][0]['intrain']
                                    pos_samples += 1

                            # Zero-pad up to the sequence length.
                            while len(input_ids) < max_seq_length:
                                input_ids.append(0)
                                input_mask.append(0)
                                segment_ids.append(0)

                            assert len(input_ids) == max_seq_length
                            assert len(input_mask) == max_seq_length
                            assert len(segment_ids) == max_seq_length

                            if debug and i_line == 1:
                                print('#' * 100)
                                print('E1:', [e['name'] for e in entity1])
                                print('E2:', [e['name'] for e in entity2])
                                print('intrain:', intrain)
                                print('Evidence:', evidence)
                                print('tokens:', tokens)
                                print('segment ids:', segment_ids)
                                print('input ids:', input_ids)
                                print('input mask', input_mask)
                                print('relation_label:', relation_label)

                            i_line += 1

                            feature = {'input_ids': input_ids,
                                       'input_mask': input_mask,
                                       'segment_ids': segment_ids,
                                       'labels': relation_label,
                                       'evidences': evidence,
                                       'intrain': intrain}

                            w.write(json.dumps(feature, ensure_ascii=False) + '\n')

    print(output_file, 'final samples', i_line)
    print('pos samples:', pos_samples)
    print('neg samples:', neg_samples)
