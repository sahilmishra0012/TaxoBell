import os
import numpy as np
import pickle as pkl
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import json

# Dataset classes and loaders.


class Data_TRAIN_Multiparent(Dataset):

    def __init__(self, args, tokenizer):

        super(Data_TRAIN_Multiparent, self).__init__()

        self.args = args
        self.dataset = args.dataset
        print("Dataset: {}".format(self.dataset))
        self.data = self.__load_data__(self.dataset)
        self.tokenizer = tokenizer
        # self.txid_concept_json = json.load(
        #     open(f"../data/{self.dataset}/key_value.json"))

        self.concept_set = self.data["concept_set"]
        self.concept_id = self.data["concept2id"]
        self.id_concept = self.data["id2concept"]
        self.id_context = self.data["id2context"]
        self.train_concept_set = self.data["train_concept_set"]
        self.train_parent_list = self.data["train_parent_list"]
        self.train_child_list = self.data["train_child_list"]
        self.train_negative_parent_dict = self.data["train_negative_parent_dict"]
        self.child_parent_pair = self.data["child_parent_pair"]
        self.child_neg_parent_pair = self.data["child_neg_parent_pair"]
        self.path2root = self.data["path2root"]

        self.train_child_parent_negative_parent_triple = self.data[
            "train_child_parent_negative_parent_triple"]
        print("Training samples: {}".format(
            len(self.train_child_parent_negative_parent_triple)))

        self.encode_all = self.generate_all_token_ids(self.tokenizer)
        # self.encode_all_paths = self.generate_all_token_ids_paths(self.tokenizer)

    def __load_data__(self, dataset):

        with open(os.path.join("../data/", dataset, "processed", "taxonomy_data_"+str(self.args.expID)+str(self.args.negsamples)+"_.pkl"), "rb") as f:
            data = pkl.load(f)

        return data

    # def join_paths_from_indices(self, path, sep=" is child of "):
    #     return sep.join([self.id_concept[idx] for idx in path])

    # def generate_all_token_ids_paths(self,tokenizer):
    #     all_nodes_paths = [self.path2root[cid] for cid in self.concept_set]
    #     all_nodes_context = [self.join_paths_from_indices(path) for path in all_nodes_paths]

    #     encode_all = tokenizer(all_nodes_context, padding='max_length', max_length=self.args.padmaxlen, return_tensors='pt')

    #     if self.args.cuda:
    #         a_input_ids = encode_all['input_ids'].cuda()
    #         a_token_type_ids = encode_all['token_type_ids'].cuda()
    #         a_attention_mask = encode_all['attention_mask'].cuda()

    #         encode_all = {'input_ids' : a_input_ids,
    #                     'token_type_ids' : a_token_type_ids,
    #                     'attention_mask' : a_attention_mask}
    #     return encode_all

    def generate_all_token_ids(self, tokenizer):

        if self.args.model == 'e5':
            all_nodes_context = ["query: "+self.id_context[cid]
                                 for cid in self.concept_set]
        else:
            all_nodes_context = [self.id_context[cid]
                                 for cid in self.concept_set]

        encode_all = tokenizer(all_nodes_context, padding='max_length',
                               max_length=self.args.padmaxlen, return_tensors='pt', truncation=True)

        if self.args.cuda and self.args.model == 'bert':
            a_input_ids = encode_all['input_ids'].cuda()
            a_token_type_ids = encode_all['token_type_ids'].cuda()
            a_attention_mask = encode_all['attention_mask'].cuda()

            encode_all = {'input_ids': a_input_ids,
                          'token_type_ids': a_token_type_ids,
                          'attention_mask': a_attention_mask}
        elif self.args.cuda and self.args.model == 'snowflake':
            a_input_ids = encode_all['input_ids'].cuda()
            a_attention_mask = encode_all['attention_mask'].cuda()

            encode_all = {
                'input_ids': a_input_ids,
                'attention_mask': a_attention_mask
            }
        elif self.args.cuda and self.args.model == 'e5':
            a_input_ids = encode_all['input_ids'].cuda()
            a_attention_mask = encode_all['attention_mask'].cuda()

            encode_all = {
                'input_ids': a_input_ids,
                'attention_mask': a_attention_mask
            }

        if self.args.cuda:
            for key, value in encode_all.items():
                encode_all[key] = value.cuda()

        return encode_all

    def index_token_ids(self, encode_dic, index):

        if self.args.model == 'bert':
            input_ids, token_type_ids, attention_mask = encode_dic[
                "input_ids"], encode_dic["token_type_ids"], encode_dic["attention_mask"]

            res_dic = {'input_ids': input_ids[index],
                       'token_type_ids': token_type_ids[index],
                       'attention_mask': attention_mask[index]}
        elif self.args.model == 'e5':
            input_ids, attention_mask = encode_dic['input_ids'], encode_dic['attention_mask']
            res_dic = {
                'input_ids': input_ids[index],
                'attention_mask': attention_mask[index]
            }
        else:
            input_ids, attention_mask = encode_dic[
                "input_ids"], encode_dic["attention_mask"]

            res_dic = {'input_ids': input_ids[index],
                       'attention_mask': attention_mask[index]}

        return res_dic

    def generate_parent_child_token_ids(self, index):

        child_id, parent_id, negative_parent_id = self.train_child_parent_negative_parent_triple[
            index]
        encode_child = self.index_token_ids(self.encode_all, child_id)
        encode_parent = self.index_token_ids(self.encode_all, parent_id)
        encode_negative_parents = self.index_token_ids(
            self.encode_all, negative_parent_id)

        # encode_parent_path = self.index_token_ids(self.encode_all_paths,parent_id)
        # encode_negative_parents_path = self.index_token_ids(self.encode_all_paths,negative_parent_id)
        return encode_parent, encode_child, encode_negative_parents

    def __getitem__(self, index):
        encode_parent, encode_child, encode_negative_parents = self.generate_parent_child_token_ids(
            index)

        return encode_parent, encode_child, encode_negative_parents

    def __len__(self):

        return len(self.train_child_parent_negative_parent_triple)


class Data_TEST_Multiparent(Dataset):

    def __init__(self, args, tokenizer):

        super(Data_TEST_Multiparent, self).__init__()

        self.args = args
        self.dataset = args.dataset
        print("Dataset: {}".format(self.dataset))
        self.data = self.__load_data__(self.dataset)
        self.tokenizer = tokenizer

        self.concept_set = self.data["concept_set"]
        self.concept_id = self.data["concept2id"]
        self.id_concept = self.data["id2concept"]
        # self.txid_concept_json = json.load(
        #     open(f"../data/{self.dataset}/key_value.json"))
        self.id_context = self.data["id2context"]

        self.train_concept_set = list(self.data["train_concept_set"])
        self.path2root = self.data["path2root"]
        self.test_concepts_id = self.data["test_concepts_id"]
        self.test_gt_id = self.data["test_gt_id"]
        self.true_concept_set = list(
            self.concept_set-set(self.test_concepts_id))

        self.encode_all = self.generate_all_token_ids(self.tokenizer)
        # self.encode_all_paths = self.generate_all_token_ids_paths(self.tokenizer)

        self.encode_query = self.generate_test_token_ids(
            self.tokenizer, self.test_concepts_id)

    def __load_data__(self, dataset):

        with open(os.path.join("../data/", dataset, "processed", "taxonomy_data_"+str(self.args.expID)+str(self.args.negsamples)+"_.pkl"), "rb") as f:
            data = pkl.load(f)

        return data

    def generate_all_token_ids(self, tokenizer):

        if self.args.model == 'e5':
            all_nodes_context = ["query: "+self.id_context[cid]
                                 for cid in self.concept_set]
        else:
            all_nodes_context = [self.id_context[cid]
                                 for cid in self.concept_set]

        encode_all = tokenizer(all_nodes_context, padding='max_length',
                               max_length=self.args.padmaxlen, return_tensors='pt', truncation=True)

        if self.args.cuda and self.args.model == 'bert':
            a_input_ids = encode_all['input_ids'].cuda()
            a_token_type_ids = encode_all['token_type_ids'].cuda()
            a_attention_mask = encode_all['attention_mask'].cuda()

            encode_all = {'input_ids': a_input_ids,
                          'token_type_ids': a_token_type_ids,
                          'attention_mask': a_attention_mask}
        elif self.args.cuda and self.args.model == 'snowflake':
            a_input_ids = encode_all['input_ids'].cuda()
            a_attention_mask = encode_all['attention_mask'].cuda()

            encode_all = {
                'input_ids': a_input_ids,
                'attention_mask': a_attention_mask
            }
        elif self.args.cuda and self.args.model == 'e5':
            a_input_ids = encode_all['input_ids'].cuda()
            a_attention_mask = encode_all['attention_mask'].cuda()

            encode_all = {
                'input_ids': a_input_ids,
                'attention_mask': a_attention_mask
            }

        if self.args.cuda:
            for key, value in encode_all.items():
                encode_all[key] = value.cuda()

        return encode_all

    # def join_paths_from_indices(self, path, sep=" is child of "):
    #     return sep.join([self.id_concept[idx] for idx in path])

    # def generate_all_token_ids_paths(self,tokenizer):
    #     all_nodes_paths = [self.path2root[cid] for cid in self.concept_set]
    #     all_nodes_context = [self.join_paths_from_indices(path) for path in all_nodes_paths]

    #     encode_all = tokenizer(all_nodes_context, padding='max_length', max_length=self.args.padmaxlen, return_tensors='pt')

    #     if self.args.cuda:
    #         a_input_ids = encode_all['input_ids'].cuda()
    #         a_token_type_ids = encode_all['token_type_ids'].cuda()
    #         a_attention_mask = encode_all['attention_mask'].cuda()

    #         encode_all = {'input_ids' : a_input_ids,
    #                     'token_type_ids' : a_token_type_ids,
    #                     'attention_mask' : a_attention_mask}
    #     return encode_all

    def generate_test_token_ids(self, tokenizer, test_concepts_id):

        if self.args.model == 'e5':
            test_nodes_context = ["query: "+self.id_context[cid]
                                  for cid in test_concepts_id]
        else:
            test_nodes_context = [self.id_context[cid]
                                  for cid in test_concepts_id]

        encode_all = tokenizer(test_nodes_context, padding='max_length',
                               max_length=self.args.padmaxlen, return_tensors='pt', truncation=True)
        encode_test = encode_all
        if self.args.cuda and self.args.model == 'bert':
            a_input_ids = encode_all['input_ids'].cuda()
            a_token_type_ids = encode_all['token_type_ids'].cuda()
            a_attention_mask = encode_all['attention_mask'].cuda()

            encode_test = {'input_ids': a_input_ids,
                           'token_type_ids': a_token_type_ids,
                           'attention_mask': a_attention_mask}
        elif self.args.cuda and self.args.model == 'snowflake':
            a_input_ids = encode_all['input_ids'].cuda()
            a_attention_mask = encode_all['attention_mask'].cuda()

            encode_test = {
                'input_ids': a_input_ids,
                'attention_mask': a_attention_mask
            }
        elif self.args.cuda and self.args.model == 'e5':
            a_input_ids = encode_all['input_ids'].cuda()
            a_attention_mask = encode_all['attention_mask'].cuda()

            encode_test = {
                'input_ids': a_input_ids,
                'attention_mask': a_attention_mask
            }

        return encode_test

    def index_token_ids(self, encode_dic, index):

        if self.args.model == 'bert':
            input_ids, token_type_ids, attention_mask = encode_dic[
                "input_ids"], encode_dic["token_type_ids"], encode_dic["attention_mask"]

            res_dic = {'input_ids': input_ids[index],
                       'token_type_ids': token_type_ids[index],
                       'attention_mask': attention_mask[index]}
        else:
            input_ids, attention_mask = encode_dic[
                "input_ids"], encode_dic["attention_mask"]

            res_dic = {'input_ids': input_ids[index],
                       'attention_mask': attention_mask[index]}

        return res_dic

    def __getitem__(self, index):

        candidate_ids = self.true_concept_set[index]
        encode_candidate = self.index_token_ids(self.encode_all, candidate_ids)
        # encode_candidate_path = self.index_token_ids(self.encode_all_paths,candidate_ids)
        return encode_candidate

    def __len__(self):
        return len(self.true_concept_set)


class Data_TRAIN(Dataset):

    def __init__(self, args, tokenizer):

        super(Data_TRAIN, self).__init__()

        self.args = args
        self.dataset = args.dataset
        print("Dataset: {}".format(self.dataset))
        self.data = self.__load_data__(self.dataset)
        self.tokenizer = tokenizer

        self.concept_set = self.data["concept_set"]
        self.concept_id = self.data["concept2id"]
        self.id_concept = self.data["id2concept"]
        self.id_context = self.data["id2context"]
        self.train_concept_set = self.data["train_concept_set"]
        self.train_parent_list = self.data["train_parent_list"]
        self.train_child_list = self.data["train_child_list"]
        self.train_negative_parent_dict = self.data["train_negative_parent_dict"]
        self.train_sibling_dict = self.data["train_sibling_dict"]
        self.child_parent_pair = self.data["child_parent_pair"]
        self.child_neg_parent_pair = self.data["child_neg_parent_pair"]
        self.child_sibling_pair = self.data["child_sibling_pair"]
        self.path2root = self.data["path2root"]

        self.train_child_parent_negative_parent_triple = self.data[
            "train_child_parent_negative_parent_triple"]
        print("Training samples: {}".format(
            len(self.train_child_parent_negative_parent_triple)))

        self.encode_all = self.generate_all_token_ids(self.tokenizer)
        # self.encode_all_paths = self.generate_all_token_ids_paths(self.tokenizer)

    def __load_data__(self, dataset):

        with open(os.path.join("../data/", dataset, "processed", "taxonomy_data_"+str(self.args.expID)+str(self.args.negsamples)+"_.pkl"), "rb") as f:
            data = pkl.load(f)

        return data

    # def join_paths_from_indices(self, path, sep=" is child of "):
    #     return sep.join([self.id_concept[idx] for idx in path])

    # def generate_all_token_ids_paths(self,tokenizer):
    #     all_nodes_paths = [self.path2root[cid] for cid in self.concept_set]
    #     all_nodes_context = [self.join_paths_from_indices(path) for path in all_nodes_paths]

    #     encode_all = tokenizer(all_nodes_context, padding='max_length', max_length=self.args.padmaxlen, return_tensors='pt')

    #     if self.args.cuda:
    #         a_input_ids = encode_all['input_ids'].cuda()
    #         a_token_type_ids = encode_all['token_type_ids'].cuda()
    #         a_attention_mask = encode_all['attention_mask'].cuda()

    #         encode_all = {'input_ids' : a_input_ids,
    #                     'token_type_ids' : a_token_type_ids,
    #                     'attention_mask' : a_attention_mask}
    #     return encode_all

    def generate_all_token_ids(self, tokenizer):

        all_nodes_context = [self.id_context[cid] for cid in self.concept_set]

        encode_all = tokenizer(all_nodes_context, padding='max_length',
                               max_length=self.args.padmaxlen, return_tensors='pt', truncation=True)

        if self.args.cuda and self.args.model != 'snowflake':
            a_input_ids = encode_all['input_ids'].cuda()
            a_token_type_ids = encode_all['token_type_ids'].cuda()
            a_attention_mask = encode_all['attention_mask'].cuda()

            encode_all = {'input_ids': a_input_ids,
                          'token_type_ids': a_token_type_ids,
                          'attention_mask': a_attention_mask}
        elif self.args.cuda and self.args.model == 'snowflake':
            a_input_ids = encode_all['input_ids'].cuda()
            a_attention_mask = encode_all['attention_mask'].cuda()

            encode_all = {
                'input_ids': a_input_ids,
                'attention_mask': a_attention_mask
            }
        return encode_all

    def index_token_ids(self, encode_dic, index):

        if self.args.model != 'snowflake':
            input_ids, token_type_ids, attention_mask = encode_dic[
                "input_ids"], encode_dic["token_type_ids"], encode_dic["attention_mask"]

            res_dic = {'input_ids': input_ids[index],
                       'token_type_ids': token_type_ids[index],
                       'attention_mask': attention_mask[index]}
        else:
            input_ids, attention_mask = encode_dic[
                "input_ids"], encode_dic["attention_mask"]

            res_dic = {'input_ids': input_ids[index],
                       'attention_mask': attention_mask[index]}

        return res_dic

    def generate_parent_child_token_ids(self, index):

        child_id, parent_id, negative_parent_id = self.train_child_parent_negative_parent_triple[
            index]
        encode_child = self.index_token_ids(self.encode_all, child_id)
        encode_parent = self.index_token_ids(self.encode_all, parent_id)
        encode_negative_parents = self.index_token_ids(
            self.encode_all, negative_parent_id)

        # encode_parent_path = self.index_token_ids(self.encode_all_paths,parent_id)
        # encode_negative_parents_path = self.index_token_ids(self.encode_all_paths,negative_parent_id)
        return encode_parent, encode_child, encode_negative_parents

    def __getitem__(self, index):
        encode_parent, encode_child, encode_negative_parents = self.generate_parent_child_token_ids(
            index)

        return encode_parent, encode_child, encode_negative_parents

    def __len__(self):

        return len(self.train_child_parent_negative_parent_triple)


class Data_TEST(Dataset):

    def __init__(self, args, tokenizer):

        super(Data_TEST, self).__init__()

        self.args = args
        self.dataset = args.dataset
        print("Dataset: {}".format(self.dataset))
        self.data = self.__load_data__(self.dataset)
        self.tokenizer = tokenizer

        self.concept_set = self.data["concept_set"]
        self.concept_id = self.data["concept2id"]
        self.id_concept = self.data["id2concept"]
        self.id_context = self.data["id2context"]

        self.train_concept_set = list(self.data["train_concept_set"])
        self.path2root = self.data["path2root"]
        self.test_concepts_id = self.data["test_concepts_id"]
        self.test_gt_id = self.data["test_gt_id"]

        self.encode_all = self.generate_all_token_ids(self.tokenizer)
        # self.encode_all_paths = self.generate_all_token_ids_paths(self.tokenizer)

        self.encode_query = self.generate_test_token_ids(
            self.tokenizer, self.test_concepts_id)

    def __load_data__(self, dataset):

        with open(os.path.join("../data/", dataset, "processed", "taxonomy_data_"+str(self.args.expID)+str(self.args.negsamples)+"_.pkl"), "rb") as f:
            data = pkl.load(f)

        return data

    def generate_all_token_ids(self, tokenizer):

        all_nodes_context = [self.id_context[cid] for cid in self.concept_set]

        encode_all = tokenizer(all_nodes_context, padding='max_length',
                               max_length=self.args.padmaxlen, return_tensors='pt', truncation=True)

        if self.args.cuda and self.args.model != 'snowflake':
            a_input_ids = encode_all['input_ids'].cuda()
            a_token_type_ids = encode_all['token_type_ids'].cuda()
            a_attention_mask = encode_all['attention_mask'].cuda()

            encode_all = {'input_ids': a_input_ids,
                          'token_type_ids': a_token_type_ids,
                          'attention_mask': a_attention_mask}
        elif self.args.cuda and self.args.model == 'snowflake':
            a_input_ids = encode_all['input_ids'].cuda()
            a_attention_mask = encode_all['attention_mask'].cuda()

            encode_all = {
                'input_ids': a_input_ids,
                'attention_mask': a_attention_mask
            }

        if self.args.cuda:
            for key, value in encode_all.items():
                encode_all[key] = value.cuda()

        return encode_all

    # def join_paths_from_indices(self, path, sep=" is child of "):
    #     return sep.join([self.id_concept[idx] for idx in path])

    # def generate_all_token_ids_paths(self,tokenizer):
    #     all_nodes_paths = [self.path2root[cid] for cid in self.concept_set]
    #     all_nodes_context = [self.join_paths_from_indices(path) for path in all_nodes_paths]

    #     encode_all = tokenizer(all_nodes_context, padding='max_length', max_length=self.args.padmaxlen, return_tensors='pt')

    #     if self.args.cuda:
    #         a_input_ids = encode_all['input_ids'].cuda()
    #         a_token_type_ids = encode_all['token_type_ids'].cuda()
    #         a_attention_mask = encode_all['attention_mask'].cuda()

    #         encode_all = {'input_ids' : a_input_ids,
    #                     'token_type_ids' : a_token_type_ids,
    #                     'attention_mask' : a_attention_mask}
    #     return encode_all

    def generate_test_token_ids(self, tokenizer, test_concepts_id):

        test_nodes_context = [self.id_context[cid] for cid in test_concepts_id]

        encode_all = tokenizer(test_nodes_context, padding='max_length',
                               max_length=self.args.padmaxlen, return_tensors='pt', truncation=True)
        encode_test = encode_all
        if self.args.cuda and self.args.model != 'snowflake':
            a_input_ids = encode_all['input_ids'].cuda()
            a_token_type_ids = encode_all['token_type_ids'].cuda()
            a_attention_mask = encode_all['attention_mask'].cuda()

            encode_test = {'input_ids': a_input_ids,
                           'token_type_ids': a_token_type_ids,
                           'attention_mask': a_attention_mask}
        elif self.args.cuda and self.args.model == 'snowflake':
            a_input_ids = encode_all['input_ids'].cuda()
            a_attention_mask = encode_all['attention_mask'].cuda()

            encode_test = {
                'input_ids': a_input_ids,
                'attention_mask': a_attention_mask
            }

        return encode_test

    def index_token_ids(self, encode_dic, index):

        if self.args.model != 'snowflake':
            input_ids, token_type_ids, attention_mask = encode_dic[
                "input_ids"], encode_dic["token_type_ids"], encode_dic["attention_mask"]

            res_dic = {'input_ids': input_ids[index],
                       'token_type_ids': token_type_ids[index],
                       'attention_mask': attention_mask[index]}
        else:
            input_ids, attention_mask = encode_dic[
                "input_ids"], encode_dic["attention_mask"]

            res_dic = {'input_ids': input_ids[index],
                       'attention_mask': attention_mask[index]}

        return res_dic

    def __getitem__(self, index):

        candidate_ids = self.train_concept_set[index]
        encode_candidate = self.index_token_ids(self.encode_all, candidate_ids)
        # encode_candidate_path = self.index_token_ids(self.encode_all_paths,candidate_ids)
        return encode_candidate

    def __len__(self):
        return len(self.train_concept_set)


def load_data(args, tokenizer, flag):

    if flag in set(['test', 'val']):
        shuffle_flag = False
        drop_last = False
        batch_size = 1  # 1 by default
        # data_set = Data_TEST(args, tokenizer)
        if args.dataset in ['computer_science', 'psychology', 'mesh', 'wordnet_verb', 'semeval_food']:
            data_set = Data_TEST_Multiparent(args, tokenizer)
        else:
            data_set = Data_TEST(args, tokenizer)
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        # data_set = Data_TRAIN(args, tokenizer)

        if args.dataset in ['computer_science', 'psychology', 'mesh', 'wordnet_verb', 'semeval_food']:
            data_set = Data_TRAIN_Multiparent(args, tokenizer)
        else:
            data_set = Data_TRAIN(args, tokenizer)

    # torch.manual_seed(42) # For Reproducibility
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last,
    )
    print(f"Created DataLoader for {flag} with {len(data_loader)} batches.")

    return data_loader, data_set
