
import os
import json
import pickle as pkl
import numpy as np
import collections
import itertools
import time
import sys
import random
from collections import deque, defaultdict
from typing import Dict, List
from tqdm import tqdm
import csv
import collections


def create_child_to_parents_map(taxo_file_path: str, dataset: str) -> dict:
    """
    Reads a raw taxonomy file and creates a JSON mapping of Child ID -> List of Parent IDs.

    This is useful for multi-parent datasets (DAGs) to quickly look up all valid parents 
    for a specific concept.

    Args:
        taxo_file_path (str): Path to the raw taxonomy text file (tab-separated).
        dataset (str): Name of the dataset (used for output directory).

    Returns:
        None: Writes the result to `../data/{dataset}/test_taxo.json`.
    """
    with open(f"../data/{dataset}/key_value.json", 'r') as f:
        id_term_map = json.load(f)

    child_to_parents = defaultdict(list)

    print(f"Reading taxonomy from: {taxo_file_path}")
    with open(taxo_file_path, 'r', encoding='utf-8') as f:
        for line in f:

            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')

            if len(parts) == 2:
                parent_id, child_id = parts
                child_to_parents[id_term_map[child_id]].append(
                    id_term_map[parent_id])
            else:
                print(f"Warning: Skipping malformed line: '{line}'")

    with open(f"../data/{dataset}/test_taxo.json", "w") as f:
        json.dump(child_to_parents, f, indent=4)


def terms_to_json(file, dataset):
    """
    Converts a tab-separated term definition file into a JSON dictionary.

    Args:
        file (str): Path to the input file.
        dataset (str): Dataset name.
    """
    def_dict = {}

    with open(file, 'r') as file:
        for line in file:
            contents = line.strip().split("\t")
            def_dict[contents[0]] = contents[1]

    with open(f"../data/{dataset}/defs.json", "w") as f:
        json.dump(def_dict, f, indent=4)


def csv_to_json(csv_file):
    """
    Parses a CSV file containing (id, label, term, definition) and saves a 
    Term->Definition JSON mapping.
    """
    def_dict = {}
    with open(csv_file, mode='r') as file:
        csv_reader = csv.reader(file)

        for line in csv_reader:

            i, lab, term, defi = line
            def_dict[term] = defi

    with open("../data/psychology/defs.json", 'w') as f:
        json.dump(def_dict, f, indent=4)


def analyze_parent_child_relationships(filepath: str):
    """
    Analyzes a taxonomy file to determine if it is a Tree (single parent) or a DAG (multi-parent).

    Args:
        filepath (str): Path to the taxonomy file.

    Returns:
        tuple: 
            - child_to_parent_dict (dict): Map of Child->Parent if unique.
            - multi_parent_children (dict): Map of Child->[Parents] if overlaps exist.
    """
    child_to_parents_map = collections.defaultdict(list)

    print(f"--- Analyzing file: {filepath} ---")

    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) != 2:
                    print(f"Warning: Skipping malformed line #{i+1}: '{line}'")
                    continue

                parent_id, child_id = parts[0], parts[1]

                child_to_parents_map[child_id].append(parent_id)

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None, None

    print("\n--- Verifying Unique Parent Constraint ---")

    multi_parent_children = {}
    for child, parents in child_to_parents_map.items():
        if len(parents) > 1:
            multi_parent_children[child] = parents

    if not multi_parent_children:
        print("Every child has a unique parent.")

        child_to_parent_dict = {
            child: parents[0] for child, parents in child_to_parents_map.items()
        }

        print("Generated Dictionary (first 5 items):")
        for i, (child, parent) in enumerate(child_to_parent_dict.items()):
            if i >= 5:
                break
            print(f"  '{child}': '{parent}'")
        print(f"Total items in dictionary: {len(child_to_parent_dict)}")

        return child_to_parent_dict, {}

    else:
        print(
            f"Check failed: Found {len(multi_parent_children)} children with multiple parents.")
        for child, parents in multi_parent_children.items():
            print(
                f"  - Child '{child}' is linked to {len(parents)} parents: {parents}")

        print("\nCannot generate a unique child:parent dictionary due to the errors above.")
        return None, multi_parent_children


def id_to_json(filepath: str, dataset: str):
    """
    Reads a file mapping IDs to Terms and saves it as a JSON key-value store.
    """
    try:
        with open(filepath, 'r') as f:
            contents = f.readlines()
        key_value = {id_no: term for line in contents for id_no,
                     term in [line.strip().split("\t")]}
        with open(f"../data/{dataset}/key_value.json", "w") as f:
            json.dump(key_value, f, indent=4)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")


def pre_process_multiparent(args, outID=True):
    """
    Preprocesses datasets that have a Directed Acyclic Graph (DAG) structure,
    where a concept can have multiple parents (e.g., Computer Science, MeSH).

    Pipeline:
    1. Loads raw taxonomy and term files.
    2. Maps string concepts to Integer IDs.
    3. Constructs Adjacency Lists (Forward and Reverse).
    4. Loads Context/Definitions for embedding initialization.
    5. Generates Training Triplets (Child, Parent, Negative Parent).
       - **Hard Negative Sampling**: Prioritizes siblings (nodes sharing a parent) 
         and grandparents as negative samples to make training robust.
       - **Fallback**: Uses random sampling if hard negatives are insufficient.

    Args:
        args: Argument parser object containing dataset paths and hyperparameters.
        outID (bool): If True, processes using Integer IDs.

    Returns:
        tuple: A collection of dictionaries, lists, and sets required to initialize 
               the `Data_TRAIN_Multiparent` and `Data_TEST_Multiparent` classes.
    """
    print("Processing Multiparent datasets...")
    dataset = args.dataset
    negsamples = args.negsamples

    def get_n_hop_neighbors(start_node, taxo_forward, taxo_reverse, max_hops=2):
        """
        Finds unique N-hop neighbors using Breadth-First Search (BFS).
        This traverses both up (to parents) and down (to children).
        """
        neighbors = set()
        queue = deque([(start_node, 0)])
        visited = {start_node}

        while queue:
            current_node, current_hop = queue.popleft()

            if current_hop >= max_hops:
                continue

            parents = taxo_reverse.get(current_node, [])
            for p in parents:
                if p not in visited:
                    visited.add(p)
                    neighbors.add(p)
                    queue.append((p, current_hop + 1))

            children = taxo_forward.get(current_node, [])
            for c in children:
                if c not in visited:
                    visited.add(c)
                    neighbors.add(c)
                    queue.append((c, current_hop + 1))

        return neighbors

    def load_file(filepath: str) -> list[str]:
        try:
            with open(filepath, 'r') as f:
                return f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")

    id_to_json(f"../data/{dataset}/{dataset}.terms", f"{dataset}")
    print("Saved ID-term map as json.")

    with open(f"../data/{dataset}/key_value.json", 'r') as f:
        id_term_map = json.load(f)

    def process_pair(pair):
        ids = pair.strip().split("\t")
        return (id_term_map[ids[0]], id_term_map[ids[1]])

    taxonomy_file = os.path.join(f"../data/{dataset}/{dataset}.taxo")
    full_taxonomy_pairs = load_file(taxonomy_file)

    all_concept_set_str = set([])
    all_taxo_dict_str = collections.defaultdict(list)
    for pair in full_taxonomy_pairs:
        parent, child = process_pair(pair)
        all_concept_set_str.add(child)
        all_concept_set_str.add(parent)
        all_taxo_dict_str[parent].append(child)

    concepts = sorted(all_concept_set_str)
    concept_id = {concept: idx for idx, concept in enumerate(concepts)}
    id_concept = {idx: concept for idx, concept in enumerate(concepts)}

    all_taxo_dict = collections.defaultdict(list)
    all_taxo_dict_reverse = collections.defaultdict(list)

    if outID:
        concept_set = set(concept_id.values())
        for parent_str, children_str in all_taxo_dict_str.items():
            parent_id = concept_id[parent_str]
            children_ids = [concept_id[c] for c in children_str]
            all_taxo_dict[parent_id].extend(children_ids)
            for child_id in children_ids:
                all_taxo_dict_reverse[child_id].append(parent_id)
    else:
        concept_set = all_concept_set_str
        all_taxo_dict = all_taxo_dict_str

    print(f"Loaded {len(concept_set)} total concepts.")

    train_taxnomy_file = os.path.join(
        f"../data/{dataset}/{dataset}_train.taxo")
    train_taxonomy_pairs = load_file(train_taxnomy_file)

    parent_list, child_list = [], []
    train_concept_set = set()
    chd2par_dict = collections.defaultdict(set)
    taxo_dict = collections.defaultdict(list)
    taxo_edges = []

    print("Processing Training Data...")
    for pair in train_taxonomy_pairs:
        parent, child = process_pair(pair)

        if outID:
            parent, child = concept_id[parent], concept_id[child]
        parent_list.append(parent)
        child_list.append(child)
        train_concept_set.add(parent)
        train_concept_set.add(child)

        chd2par_dict[child].add(parent)
        taxo_dict[parent].append(child)
        taxo_edges.append((parent, child))

    all_children = set(child_list)
    roots = train_concept_set-all_children
    print(f"Found {len(taxo_edges)} training edges....")

    if args.dataset == 'computer_science':
        supernode = concept_id['computer science']
    elif args.dataset == 'psychology':
        supernode = concept_id['psychology']

    dic_file = os.path.join(f"../data/{args.dataset}/defs.json")
    def_dic = json.load(open(dic_file))

    id_context = {}
    definitions_not_found_count = 0

    for cid, concept in id_concept.items():
        concept_lower = concept.lower()
        if concept_lower in def_dic:
            id_context[cid] = f"{concept_lower}: {def_dic[concept_lower]}"
        else:
            id_context[cid] = f"{concept_lower}: {concept_lower}"
            definitions_not_found_count += 1

    test_terms_file = os.path.join(
        f"../data/{dataset}/{dataset}_test.terms")

    test_term_lines = load_file(test_terms_file)
    with open(f"../data/{args.dataset}/test_taxo.json") as f:
        test_map = json.load(f)

    print("Processing validation and test sets...")

    def get_eval_data(lines):
        concept_ids, gts_ids = [], []
        q = 0
        for line in lines:
            _, child_term = line.strip().split("\t")
            parent_ids = list()
            if child_term in list(concept_id.keys()):
                child_id = concept_id[child_term]
                parent_terms = test_map[child_term]

                for p in parent_terms:
                    parent_ids.append(concept_id[p])

                concept_ids.append(child_id)
                gts_ids.append(parent_ids)
            else:
                print(
                    f"Found invalid term {child_term}..removing from test set.")

        return concept_ids, gts_ids

    val_concepts_ids, val_gts_ids = [], []
    test_concepts_ids, test_gts_ids = get_eval_data(test_term_lines)

    sampled_negative_parent_dict = {}
    negative_parent_list = []

    child_parent_pair = [[child, parent]
                         for child, parent in zip(child_list, parent_list)]

    count_hard_neg_samples = 0
    count_fallback_samples = 0

    training_triplets = list()
    count_skipped = 0
    print(
        f"Definitions for {definitions_not_found_count} concepts not found in the wiki dictionary.")
    print(f"There are {len(child_parent_pair)} in the training set ")

    for child_id, parent_id in tqdm(child_parent_pair, desc="Generating (c, p, n) Triplets"):

        found_negatives = []
        hard_candidate_pool = set()

        siblings = set(all_taxo_dict.get(parent_id, []))
        siblings.discard(child_id)
        hard_candidate_pool.update(siblings)

        grandparents = set(all_taxo_dict_reverse.get(parent_id, []))
        hard_candidate_pool.update(grandparents)

        filtered_hard_candidates = [
            node for node in hard_candidate_pool
            if node in train_concept_set and node != 'root'
        ]

        num_hard_to_sample = min(negsamples, len(filtered_hard_candidates))

        if num_hard_to_sample > 0:
            hard_samples = np.random.choice(
                filtered_hard_candidates,
                size=num_hard_to_sample,
                replace=False
            ).tolist()
            found_negatives.extend(hard_samples)
            count_hard_neg_samples += len(hard_samples)

        while len(found_negatives) < negsamples:
            random_negative = np.random.choice(list(train_concept_set))

            if (random_negative != child_id and
                random_negative != parent_id and
                random_negative != 'root' and
                    random_negative not in found_negatives):

                found_negatives.append(random_negative)
                count_fallback_samples += 1

        for neg_id in found_negatives:
            training_triplets.append((child_id, parent_id, neg_id))

    print("Negative sampling done.")
    print(
        f"Skipped for {count_fallback_samples} triplets since no hard negative was found. Using fallback option.")

    child_neg_parent_pair = []
    child_parent_negative_parent_triple = training_triplets

    print(
        f"There are {len(child_parent_negative_parent_triple)} samples for training")

    print(
        f"There are {len(test_concepts_ids)} test concepts with {len(test_gts_ids)} grount truth mappings")

    path2root = collections.defaultdict(list)
    print("Preprocessing complete.")

    return (
        concept_set, concept_id, id_concept, id_context, train_concept_set, taxo_dict,
        sampled_negative_parent_dict, child_parent_negative_parent_triple, parent_list, child_list,
        negative_parent_list, all_taxo_dict, path2root, child_parent_pair,
        child_neg_parent_pair, val_concepts_ids, val_gts_ids, test_concepts_ids, test_gts_ids
    )


def preprocess(args, outID=True):
    """
    Preprocesses standard taxonomy data (Tree structures like WordNet or single-parent datasets).

    Differences from `pre_process_multiparent`:
    - optimized for finding siblings, cousins, and specific relative structures.
    - handles the "wordnet" supernode specifically.
    - assumes a stricter hierarchy logic.

    Args:
        args: Command-line arguments.
        outID (bool): If True, outputs IDs for concepts; otherwise, outputs names.

    Returns:
        Tuple containing processed data structures for single-parent taxonomy evaluation.
    """
    dataset = args.dataset

    def load_file(filepath: str) -> list[str]:
        """Helper function to load a file and return lines."""
        try:
            with open(filepath, 'r') as f:
                return f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")

    def process_pair(pair: str, dataset: str) -> tuple[str, str]:
        """Helper function to split and process a taxonomy pair."""
        text = pair.strip().split("\t")
        if (dataset == "wordnet" or "wordnet" in dataset[:8]):
            return (text[-1], text[-2])
        return (text[-2], text[-1]) if len(text) >= 3 else (text[0], text[1])

    taxonomy_file = os.path.join(f"../data/{dataset}/{dataset}_raw_en.taxo")
    taxonomy = load_file(taxonomy_file)

    concept_set = set([])
    all_taxo_dict = collections.defaultdict(list)

    for pair in taxonomy:
        child, parent = process_pair(pair, dataset)
        concept_set.add(child)
        concept_set.add(parent)

    concepts = sorted(concept_set)
    concept_id = {concept: idx for idx, concept in enumerate(concepts)}
    id_concept = {idx: concept for concept, idx in concept_id.items()}

    if outID:
        concept_set = set([concept_id[con] for con in list(concept_set)])
        for pair in taxonomy:
            child, parent = process_pair(pair, dataset)
            all_taxo_dict[concept_id[parent]].append(concept_id[child])

    train_taxonomy_file = os.path.join(
        f"../data/{dataset}/{dataset}_train.taxo")
    train_taxonomy = load_file(train_taxonomy_file)

    parent_list, child_list = [], []
    train_concept_set = set([])
    chd2par_dict = collections.defaultdict(set)
    taxo_dict = collections.defaultdict(list)

    taxo_edges = []
    for pair in train_taxonomy:
        parent, child = process_pair(pair, dataset)
        if outID:
            parent, child = concept_id[parent], concept_id[child]
        parent_list.append(parent)
        child_list.append(child)
        train_concept_set.add(parent)
        train_concept_set.add(child)
        chd2par_dict[child].add(parent)
        taxo_dict[parent].append(child)
        taxo_edges.append((parent, child))

    all_children = set(child_list)
    roots = train_concept_set - all_children

    if dataset == "wordnet" or "wordnet" in dataset[:8]:
        supernode = len(concepts)
        concept_id[dataset] = supernode
        id_concept[supernode] = dataset

        for root in roots:
            taxo_dict[supernode].append(root)
            chd2par_dict[root].add(supernode)
    else:
        if outID:
            supernode = concept_id[dataset]

    sibling_dict = collections.defaultdict(set)
    for parent, children in taxo_dict.items():
        for child in children:
            sibling_dict[child].update(set(children) - {child})

    if dataset == "wordnet" or "wordnet" in dataset[:8]:
        observe_nodes = train_concept_set - \
            {supernode} - set(taxo_dict[supernode])
    else:
        observe_nodes = train_concept_set

    sib_pair = [[k, l] for k, children in sibling_dict.items()
                for l in children]

    cousin_dict = collections.defaultdict(set)
    for node in observe_nodes:
        pars = chd2par_dict[node]
        for par in pars:
            cousins = sibling_dict[par] - pars
            cousin_dict[node].update(cousins)
            for uncle in cousins:
                cousin_dict[node].update(taxo_dict[uncle])
            cousin_dict[node] -= sibling_dict[node]

    relative_triple = [[node, s, c]
                       for node in observe_nodes for s in sibling_dict[node] for c in cousin_dict[node]]

    negative_parent_dict = {
        cid: sibling_dict[cid] | cousin_dict[cid] for cid in id_concept}

    negative_parent_list = []
    sampled_negative_parent_dict = {}

    for cid in child_list:
        negative_parents = list(negative_parent_dict[cid])
        if len(negative_parents) > args.negsamples:
            negative_parents = list(np.random.choice(
                negative_parents, args.negsamples, replace=False))
        sampled_negative_parent_dict[cid] = negative_parents
        negative_parent_list.extend(negative_parents)

    child_parent_negative_parent_triple = [
        [child_list[i], parent_list[i], neg]
        for i, cid in enumerate(child_list)
        for neg in sampled_negative_parent_dict[cid]
    ]

    child_parent_pair = [[child, parent]
                         for child, parent in zip(child_list, parent_list)]

    child_neg_parent_pair = [
        [cid, neg]
        for cid in child_list
        for neg in sampled_negative_parent_dict[cid]
    ]

    child_sibling_pair = [
        [cid, sib]
        for cid in child_list
        for sib in sibling_dict[cid]
    ]

    dic_file = os.path.join(f"../data/{dataset}/dic.json")
    def_dic = json.load(open(dic_file))
    def_dic = {key.lower(): value for key, value in def_dic.items()}
    if dataset == "wordnet" or "wordnet" in dataset[:8]:
        if dataset not in def_dic:
            def_dic[dataset] = ["Supernode"]

    id_context = {
        cid: f"{concept.lower()}: {def_dic[concept.lower()][0]}"
        for cid, concept in id_concept.items()
    }

    test_terms_file = os.path.join(f"../data/{dataset}/{dataset}_eval.terms")
    test_gt_file = os.path.join(f"../data/{dataset}/{dataset}_eval.gt")
    test_terms = load_file(test_terms_file)
    test_gt = load_file(test_gt_file)

    test_concepts_id = [concept_id[term.strip()] for term in test_terms]
    test_gt_id = [concept_id[term.strip()] for term in test_gt]

    shuffled_data = list(zip(test_concepts_id, test_gt_id))
    np.random.shuffle(shuffled_data)
    split_idx = len(shuffled_data) // 2
    val_concept, val_gt = zip(*shuffled_data[:split_idx])
    test_concept, test_gt = zip(*shuffled_data[split_idx:])

    path2root = collections.defaultdict(list)
    for node in train_concept_set:
        current = node
        while current != supernode:
            path2root[node].append(current)
            current = list(chd2par_dict[current])[0]
        path2root[node].append(supernode)

    return (
        concept_set, concept_id, id_concept, id_context, train_concept_set, taxo_dict,
        negative_parent_dict, child_parent_negative_parent_triple, parent_list, child_list,
        negative_parent_list, sibling_dict, cousin_dict, relative_triple, test_concepts_id,
        test_gt_id, all_taxo_dict, path2root, sib_pair, child_parent_pair, child_neg_parent_pair,
        child_sibling_pair, val_concept, val_gt, test_concept, test_gt
    )


def create_multiparent_data(args):
    """
    Wrapper function that runs the multi-parent preprocessing pipeline and saves the 
    result as a pickle file for efficient loading during training.

    Args:
        args: Command-line arguments containing dataset configuration.
    """
    print("Waiting for preprocess data....")

    concept_set, concept_id, id_concept, id_context, train_concept_set, taxo_dict, negative_parent_dict, child_parent_negative_parent_triple, parent_list, child_list, negative_parent_list, all_taxo_dict, path2root, child_parent_pair, child_neg_parent_pair, val_concept, val_gt, test_concepts_id, test_gt = pre_process_multiparent(
        args)
    save_data = {
        "concept_set": concept_set,
        "concept2id": concept_id,
        "id2concept": id_concept,
        "id2context": id_context,
        "train_concept_set": train_concept_set,
        "train_taxo_dict": taxo_dict,
        "all_taxo_dict": all_taxo_dict,
        "train_negative_parent_dict": negative_parent_dict,
        "train_child_parent_negative_parent_triple": child_parent_negative_parent_triple,
        "train_parent_list": parent_list,
        "train_child_list": child_list,
        "train_negative_parent_list": negative_parent_list,
        "test_concepts_id": test_concepts_id,
        "test_gt_id": test_gt,
        "path2root": path2root,
        "child_parent_pair": child_parent_pair,
        "child_neg_parent_pair": child_neg_parent_pair,
        "val_concept": val_concept,
        "val_gt": val_gt,
        "test_concept": test_concepts_id,
        "test_gt": test_gt
    }

    with open("../data/"+str(args.dataset)+"/processed/taxonomy_data_"+str(args.expID)+str(args.negsamples)+"_.pkl", "wb") as f:
        pkl.dump(save_data, f)

    print("Waiting for saving processed data....")
    print("Done!")
    print(
        f"From processed data, there are :{len(child_parent_negative_parent_triple)} training instances")
    print(f"From processed data, there are :{len(test_gt)} test instances")


def create_data(args, maxlimit=None):
    """
    Wrapper function that runs the single-parent (tree) preprocessing pipeline and saves the 
    result as a pickle file.

    Args:
        args: Command-line arguments.
        maxlimit: Unused argument kept for API consistency.
    """
    concept_set, concept_id, id_concept, id_context, train_concept_set, train_taxo_dict, negative_parent_dict, train_child_parent_negative_parent_triple, train_parent_list, \
        train_child_list, train_negative_parent_list, train_sibling_dict, train_cousin_dict, train_relative_triple, test_concepts_id, test_gt_id, \
        all_taxo_dict, path2root, sib_pair, child_parent_pair, child_neg_parent_pair, child_sibling_pair, val_concept, val_gt, test_concept, test_gt = preprocess(
            args)

    print("Waiting for preprocess data....")
    time.sleep(3)
    print("Done!")
    save_data = {
        "concept_set": concept_set,
        "concept2id": concept_id,
        "id2concept": id_concept,
        "id2context": id_context,
        "all_taxo_dict": all_taxo_dict,
        "train_concept_set": train_concept_set,
        "train_taxo_dict": train_taxo_dict,
        "train_negative_parent_dict": negative_parent_dict,
        "train_child_parent_negative_parent_triple": train_child_parent_negative_parent_triple,
        "train_parent_list": train_parent_list,
        "train_child_list": train_child_list,
        "train_negative_parent_list": train_negative_parent_list,
        "train_sibling_dict": train_sibling_dict,
        "train_cousin_dict": train_cousin_dict,
        "train_relative_triple": train_relative_triple,
        "test_concepts_id": test_concepts_id,
        "test_gt_id": test_gt_id,
        "path2root": path2root,
        "sib_pair": sib_pair,
        "child_parent_pair": child_parent_pair,
        "child_neg_parent_pair": child_neg_parent_pair,
        "child_sibling_pair": child_sibling_pair,
        "val_concept": val_concept,
        "val_gt": val_gt,
        "test_concept": test_concept,
        "test_gt": test_gt}

    with open("../data/"+str(args.dataset)+"/processed/taxonomy_data_"+str(args.expID)+str(args.negsamples)+"_.pkl", "wb") as f:
        pkl.dump(save_data, f)

    print("Waiting for saving processed data....")
    time.sleep(3)
    print("Done!")
    print(
        f"From processed data, there are :{len(train_child_parent_negative_parent_triple)} training instances")
    print(f"From processed data, there are :{len(test_gt_id)} test instances")
