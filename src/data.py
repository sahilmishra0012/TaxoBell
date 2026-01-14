import os
import numpy as np
import pickle as pkl
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import json


class Data_TRAIN_Multiparent(Dataset):
    """
    Dataset class for training models on taxonomy datasets where concepts may have 
    multiple parents (DAG structure), such as Computer Science, Psychology, or MeSH.

    It handles loading processed taxonomy data, tokenizing concepts using a 
    pretrained tokenizer, and generating (Child, Parent, Negative Parent) triplets 
    for training.
    """

    def __init__(self, args, tokenizer):
        """
        Initializes the training dataset.

        Args:
            args (Namespace): Configuration arguments containing dataset paths, 
                              batch size, model type, etc.
            tokenizer (PreTrainedTokenizer): HuggingFace tokenizer for encoding text.
        """
        super(Data_TRAIN_Multiparent, self).__init__()

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
        self.child_parent_pair = self.data["child_parent_pair"]
        self.child_neg_parent_pair = self.data["child_neg_parent_pair"]
        self.path2root = self.data["path2root"]

        self.train_child_parent_negative_parent_triple = self.data[
            "train_child_parent_negative_parent_triple"]
        print("Training samples: {}".format(
            len(self.train_child_parent_negative_parent_triple)))

        self.encode_all = self.generate_all_token_ids(self.tokenizer)

    def __load_data__(self, dataset):
        """
        Loads the preprocessed taxonomy pickle file.

        Args:
            dataset (str): Name of the dataset directory.

        Returns:
            dict: Dictionary containing taxonomy relationships and mappings.
        """
        with open(os.path.join("../data/", dataset, "processed", "taxonomy_data_"+str(self.args.expID)+str(self.args.negsamples)+"_.pkl"), "rb") as f:
            data = pkl.load(f)

        return data

    def generate_all_token_ids(self, tokenizer):
        """
        Tokenizes all concepts in the dataset at once.
        Handles specific formatting for models like 'e5' (adding "query: ").

        Args:
            tokenizer: The tokenizer object.

        Returns:
            dict: A dictionary of tensor batches (input_ids, attention_mask, etc.), 
                  optionally moved to CUDA.
        """
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
        """
        Retrieves the tokenized tensors for a specific concept index.

        Args:
            encode_dic (dict): The dictionary of all pre-tokenized concepts.
            index (int): The index of the concept to retrieve.

        Returns:
            dict: Dictionary containing 'input_ids', 'attention_mask' (and 'token_type_ids') for the specific index.
        """
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
        """
        Generates the input features for a training triplet.

        Args:
            index (int): The index in the list of training triples.

        Returns:
            tuple: (encode_parent, encode_child, encode_negative_parents) dictionaries.
        """
        child_id, parent_id, negative_parent_id = self.train_child_parent_negative_parent_triple[
            index]
        encode_child = self.index_token_ids(self.encode_all, child_id)
        encode_parent = self.index_token_ids(self.encode_all, parent_id)
        encode_negative_parents = self.index_token_ids(
            self.encode_all, negative_parent_id)

        return encode_parent, encode_child, encode_negative_parents

    def __getitem__(self, index):
        """
        Returns a single training sample.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: Tokenized inputs for Parent, Child, and Negative Parent.
        """
        encode_parent, encode_child, encode_negative_parents = self.generate_parent_child_token_ids(
            index)

        return encode_parent, encode_child, encode_negative_parents

    def __len__(self):
        """Returns the total number of training triples."""
        return len(self.train_child_parent_negative_parent_triple)


class Data_TEST_Multiparent(Dataset):
    """
    Dataset class for testing/evaluating models on multi-parent taxonomy datasets.

    This class is responsible for loading test queries (child concepts) and 
    candidate parents to evaluate the model's ability to predict correct placements.
    """

    def __init__(self, args, tokenizer):
        """
        Initializes the test dataset.

        Args:
            args (Namespace): Configuration arguments.
            tokenizer (PreTrainedTokenizer): Tokenizer object.
        """
        super(Data_TEST_Multiparent, self).__init__()

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

        self.true_concept_set = list(
            self.concept_set-set(self.test_concepts_id))

        self.encode_all = self.generate_all_token_ids(self.tokenizer)

        self.encode_query = self.generate_test_token_ids(
            self.tokenizer, self.test_concepts_id)

    def __load_data__(self, dataset):
        """Loads the processed data pickle."""
        with open(os.path.join("../data/", dataset, "processed", "taxonomy_data_"+str(self.args.expID)+str(self.args.negsamples)+"_.pkl"), "rb") as f:
            data = pkl.load(f)
        return data

    def generate_all_token_ids(self, tokenizer):
        """Tokenizes all concepts in the taxonomy (potential parents)."""
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

    def generate_test_token_ids(self, tokenizer, test_concepts_id):
        """Tokenizes specifically the test query concepts."""
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
        """Retrieves token tensors for a specific index."""
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
        """
        Returns a candidate parent for evaluation.

        Args:
            index (int): Index into the 'true_concept_set' (candidates).

        Returns:
            dict: Tokenized input for the candidate concept.
        """
        candidate_ids = self.true_concept_set[index]
        encode_candidate = self.index_token_ids(self.encode_all, candidate_ids)
        return encode_candidate

    def __len__(self):
        """Returns the number of candidate concepts for validation."""
        return len(self.true_concept_set)


class Data_TRAIN(Dataset):
    """
    Dataset class for training on standard single-parent (tree) taxonomies.
    Similar to Data_TRAIN_Multiparent but optimized for datasets where 
    siblings and strict tree structures are defined.
    """

    def __init__(self, args, tokenizer):
        """
        Initializes the standard training dataset.
        """
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

    def __load_data__(self, dataset):
        """Loads data from pickle file."""
        with open(os.path.join("../data/", dataset, "processed", "taxonomy_data_"+str(self.args.expID)+str(self.args.negsamples)+"_.pkl"), "rb") as f:
            data = pkl.load(f)
        return data

    def generate_all_token_ids(self, tokenizer):
        """Tokenizes all concepts."""
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
        """Indexes into the tokenized batch."""
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
        """Generates (Parent, Child, Negative Parent) triplets."""
        child_id, parent_id, negative_parent_id = self.train_child_parent_negative_parent_triple[
            index]
        encode_child = self.index_token_ids(self.encode_all, child_id)
        encode_parent = self.index_token_ids(self.encode_all, parent_id)
        encode_negative_parents = self.index_token_ids(
            self.encode_all, negative_parent_id)

        return encode_parent, encode_child, encode_negative_parents

    def __getitem__(self, index):
        """Returns training sample."""
        encode_parent, encode_child, encode_negative_parents = self.generate_parent_child_token_ids(
            index)
        return encode_parent, encode_child, encode_negative_parents

    def __len__(self):
        """Returns training set size."""
        return len(self.train_child_parent_negative_parent_triple)


class Data_TEST(Dataset):
    """
    Dataset class for testing on standard single-parent taxonomies.
    """

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
        self.encode_query = self.generate_test_token_ids(
            self.tokenizer, self.test_concepts_id)

    def __load_data__(self, dataset):
        with open(os.path.join("../data/", dataset, "processed", "taxonomy_data_"+str(self.args.expID)+str(self.args.negsamples)+"_.pkl"), "rb") as f:
            data = pkl.load(f)
        return data

    def generate_all_token_ids(self, tokenizer):
        """Tokenizes all concepts (potential parents)."""
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

    def generate_test_token_ids(self, tokenizer, test_concepts_id):
        """Tokenizes test query concepts."""
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
        """Returns a candidate parent from the training concept set."""
        candidate_ids = self.train_concept_set[index]
        encode_candidate = self.index_token_ids(self.encode_all, candidate_ids)
        return encode_candidate

    def __len__(self):
        """Returns size of candidate set (train concepts)."""
        return len(self.train_concept_set)


def load_data(args, tokenizer, flag):
    """
    Factory function to initialize and return a DataLoader for the specific dataset and mode.

    Args:
        args (Namespace): Configuration arguments (dataset name, batch size, etc.).
        tokenizer: HuggingFace tokenizer.
        flag (str): 'train', 'test', or 'val'. Determines which Dataset class to instantiate.

    Returns:
        tuple: (DataLoader, Dataset)
    """

    if flag in set(['test', 'val']):
        shuffle_flag = False
        drop_last = False
        batch_size = 1

        if args.dataset in ['computer_science', 'psychology', 'mesh', 'wordnet_verb', 'semeval_food']:
            data_set = Data_TEST_Multiparent(args, tokenizer)
        else:
            data_set = Data_TEST(args, tokenizer)
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

        if args.dataset in ['computer_science', 'psychology', 'mesh', 'wordnet_verb', 'semeval_food']:
            data_set = Data_TRAIN_Multiparent(args, tokenizer)
        else:
            data_set = Data_TRAIN(args, tokenizer)

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last,
    )
    print(f"Created DataLoader for {flag} with {len(data_loader)} batches.")

    return data_loader, data_set
