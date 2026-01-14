import os
import pandas as pd
import random
import networkx as nx
import helpers
import pdb
from networkx.algorithms import descendants, ancestors
from itertools import product, chain, combinations
import datetime
from collections import defaultdict
from tqdm import tqdm
import copy
from collections import deque
import sys
MAX_TEST_SIZE = 1000
MAX_VALIDATION_SIZE = 1000
date_time = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

# Prepares the Taxonomy dataset for multi parent data. This needs to be run first to create .pkl files for multi parent datasets. A run example for a given
# is written at the end of this file.


class Taxon(object):
    def __init__(self, tx_id, rank=-1, norm_name="none", display_name="None", main_type="", level="-100", p_count=0, c_count=0, create_date="None"):
        self.tx_id = tx_id
        self.rank = int(rank)
        self.norm_name = norm_name
        self.display_name = display_name
        self.main_type = main_type
        self.level = int(level)
        self.p_count = int(p_count)
        self.c_count = int(c_count)
        self.create_date = create_date

    def __str__(self):
        return "Taxon {} (name: {}, level: {})".format(self.tx_id, self.norm_name, self.level)

    def __lt__(self, another_taxon):
        if self.level < another_taxon.level:
            return True
        else:
            return self.rank < another_taxon.rank


class MultiParentDataset(object):
    def __init__(self, name, path, existing_partition=True, partition_pattern='leaf', shortest_path=False):
        self.name = name
        self.existing_partition = existing_partition
        self.partition_pattern = partition_pattern
        self.train_nods_ids = []
        self.validation_node_ids = []
        self.test_node_ids = []
        self.shortest_path = shortest_path

        self._load_dataset_raw(path)

    def _load_dataset_raw(self, dir_path):
        print("Loading nodes and edges files")
        node_file_name = os.path.join(dir_path, f"{self.name}.terms")
        edge_file_name = os.path.join(dir_path, f"{self.name}.taxo")

        if self.existing_partition:
            train_node_file_name = os.path.join(
                dir_path, f"{self.name}_train.terms")
            validation_node_file_name = os.path.join(
                dir_path, f"{self.name}_val.terms")
            test_file_name = os.path.join(dir_path, f"{self.name}_test.terms")

        tx_id2taxon = {}

        self.taxonomy = nx.DiGraph()

        print("Add node to networkx graph")
        with open(node_file_name, "r") as fin:
            for line in tqdm(fin, desc="Loading Terms"):
                line = line.strip()

                if line:
                    segs = line.split("\t")
                    assert len(
                        segs) == 2, f"Wrong number of segmentations {line}"
                    taxon = Taxon(
                        tx_id=segs[0], norm_name=segs[1], display_name=segs[1])
                    tx_id2taxon[segs[0]] = taxon
                    self.taxonomy.add_node(taxon)

        print("add edges to networkx graph")
        with open(edge_file_name, "r") as fin:
            for line in tqdm(fin, desc="Loading relations"):
                line = line.strip()

                if line:
                    segs = line.split("\t")
                    assert len(
                        segs) == 2, f"Wrong number of segmentations {line}"
                    parent_taxon = tx_id2taxon[segs[0]]
                    child_taxon = tx_id2taxon[segs[1]]
                    self.taxonomy.add_edge(parent_taxon, child_taxon)

        print(
            f"Taxonomy has {len(list(self.taxonomy.nodes()))} nodes and {len(list(self.taxonomy.edges()))} edges")
        if self.name == 'psychology' or self.name == 'computer_science':
            isolated = [n for n in self.taxonomy.nodes()
                        if self.taxonomy.in_degree(n) == 0 and self.taxonomy.out_degree(n) == 0]
            print(f"Isolated nodes : {len(isolated)}")
            self.taxonomy.remove_nodes_from(isolated)

        print("Adding pseudo rot....")
        self.root = Taxon(
            tx_id='root', norm_name='root', display_name='root'
        )
        roots = [node for node in self.taxonomy.nodes(
        ) if self.taxonomy.in_degree(node) == 0]
        self.taxonomy.add_node(self.root)
        for node in roots:
            if node != self.root:
                self.taxonomy.add_edge(self.root, node)

        try:
            cycles = nx.find_cycle(self.full_graph, orientation="original")
            for tupl in cycles:
                self.full_graph.add_edge(self.root, tupl[0])
        except:
            print("no cycles found")
        self.taxonomy.remove_edges_from(nx.selfloop_edges(self.taxonomy))

        roots = [n for n, d in self.taxonomy.in_degree() if d == 0]
        leaves = [n for n, d in self.taxonomy.out_degree() if d == 0]
        print(
            f"There are now {len(roots)} roots (should be 1) and {len(leaves)} leaves in updated taxonomy")
        print(
            f"There are {self.taxonomy.number_of_nodes()} nodes in the updated taxonomy")

        if self.existing_partition:
            print("Loading existing train/validation/test partitions")
            raw_train_node_list = self._load_node_list(train_node_file_name)
            raw_validation_node_list = self._load_node_list(
                validation_node_file_name)
            raw_test_node_list = self._load_node_list(test_file_name)

        if self.existing_partition:
            self.train_node_ids = [tx_id2taxon[tx_id]
                                   for tx_id in raw_train_node_list]
            self.validation_node_ids = [tx_id2taxon[tx_id]
                                        for tx_id in raw_validation_node_list]
            self.test_node_ids = [tx_id2taxon[tx_id]
                                  for tx_id in raw_test_node_list]
        else:
            print("Partition graph....")
            if self.partition_pattern == 'leaf':
                leaf_node_ids = []
                for node in self.taxonomy.nodes:
                    if self.taxonomy.out_degree(node) == 0:
                        leaf_node_ids.append(node.tx_id)
                random.seed(10)
                random.shuffle(leaf_node_ids)

                validation_size = min(
                    int(len(leaf_node_ids) * 0.1), MAX_VALIDATION_SIZE)
                test_size = min(
                    int(len(leaf_node_ids) * 0.1), MAX_TEST_SIZE)
                self.validation_node_ids = leaf_node_ids[:validation_size]
                self.test_node_ids = leaf_node_ids[validation_size:test_size+validation_size]
                self.train_node_ids = []
                for n in self.taxonomy.nodes:
                    if n.tx_id not in self.test_node_ids and n.tx_id not in self.validation_node_ids:
                        self.train_node_ids.append(n)

            elif self.partition_pattern == 'internal':
                print("Beginning internal partition...")
                root_node = [node for node in self.taxonomy.nodes(
                ) if self.taxonomy.in_degree(node) == 0]

                print(f"{len(root_node)} roots are present.")

                sampled_node_ids = [node.tx_id for node in self.taxonomy.nodes(
                ) if node.tx_id != root_node[0].tx_id]

                random.seed(20)
                random.shuffle(sampled_node_ids)

                validation_size = min(
                    int(len(sampled_node_ids) * 0.1), MAX_VALIDATION_SIZE)
                test_size = max(
                    int(len(sampled_node_ids) * 0.1), MAX_TEST_SIZE)

                self.test_node_ids = sampled_node_ids[:test_size]
                self.validation_node_ids = sampled_node_ids[test_size:test_size+validation_size]
                self.train_node_ids = list()

                for n in list(self.taxonomy.nodes()):
                    if n.tx_id not in self.test_node_ids and n.tx_id not in self.validation_node_ids:
                        self.train_node_ids.append(n)

        print("Graph Partitioning Complete..")
        print("Train Nodes Size: ", len(self.train_node_ids))
        print("Val Node Size: ", len(self.validation_node_ids))
        print("Test Node Size: ", len(self.test_node_ids))

        self.train_node_ids.append(self.root)
        self.train_subgraph = self._get_holdout_subgraph(
            self.train_node_ids)

        self.validation_nodes = [tx_id2taxon[tx_id]
                                 for tx_id in self.validation_node_ids]
        self.test_nodes = [tx_id2taxon[tx_id] for tx_id in self.test_node_ids]

        self.val_graph = self._get_holdout_subgraph(
            self.train_node_ids+self.validation_nodes)
        self.test_graph = self._get_holdout_subgraph(
            self.train_node_ids+self.test_nodes)

        print("Train subgraph size..")
        print(self.train_subgraph)

        print("Val Subgraph size")
        print(self.val_graph)

        print("Test Subgraph size")
        print(self.test_graph)

        self._create_new_taxo_file()

        if not self.existing_partition:

            with open(f"../data/{self.name}/{self.name}_test.terms", "w") as f:
                for n in self.test_nodes:
                    f.write(f"{n.tx_id}\t{n.norm_name}\n")

    def _load_node_list(self, file_path):
        node_list = []
        with open(file_path, "r") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    node_list.append(line.split('\t')[0])
        return node_list

    def _get_holdout_subgraph(self, node_ids):
        tx_ids = [n.tx_id for n in node_ids]
        node_to_remove = [
            n for n in self.taxonomy.nodes if n.tx_id not in tx_ids]

        subgraph = self.taxonomy.subgraph(node_ids).copy()

        for node in node_to_remove:
            parents = set()
            children = set()

            ps = deque(self.taxonomy.predecessors(node))
            cs = deque(self.taxonomy.successors(node))

            while ps:
                p = ps.popleft()
                if p in subgraph:
                    parents.add(p)
                else:
                    ps += list(self.taxonomy.predecessors(p))

            while cs:
                c = cs.popleft()
                if c in subgraph:
                    children.add(c)
                else:
                    cs += list(self.taxonomy.successors(c))
            for p, c in product(parents, children):
                subgraph.add_edge(p, c)
        node2descendants = {n: set(descendants(subgraph, n))
                            for n in subgraph.nodes}
        for node in subgraph.nodes():
            if subgraph.out_degree(node) > 1:
                successors1 = set(subgraph.successors(node))
                successors2 = set(chain.from_iterable(
                    [node2descendants[n] for n in successors1]))
                checkset = successors1.intersection(successors2)
                if checkset:
                    for s in checkset:
                        subgraph.remove_edge(node, s)
        return subgraph

    def _check_cycle(self):
        ud_train = self.taxonomy.to_undirected()

        try:
            nx.find_cycle(ud_train)
            print("Cycle exists")
            return True
        except nx.exception.NetworkXNoCycle:
            print("No cycle found")

        return False

    def _replicate_and_attach_subgraph(self, tree, dag, parent_in_tree, dag_node_to_replicate, replication_counters):

        replication_queue = deque()

        replication_counters[dag_node_to_replicate] += 1
        count = replication_counters[dag_node_to_replicate]

        first_copy = copy.deepcopy(dag_node_to_replicate)
        first_copy.tx_id = f"{dag_node_to_replicate.tx_id}_rep_{count}"

        tree.add_edge(parent_in_tree, first_copy)
        replication_queue.append((first_copy, dag_node_to_replicate))

        while replication_queue:
            repl_parent, repl_dag_node = replication_queue.popleft()
            for repl_child_dag_node in dag.successors(repl_dag_node):
                new_child_copy = copy.deepcopy(repl_child_dag_node)

                replication_counters[repl_child_dag_node] += 1
                count = replication_counters[repl_child_dag_node]
                new_child_copy.tx_id = f"{repl_child_dag_node.tx_id}_rep_{count}"

                tree.add_edge(repl_parent, new_child_copy)
                replication_queue.append((new_child_copy, repl_child_dag_node))

    def _dag_to_tree_(self, dag):
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("Input graph must be a DAG.")

        roots = [n for n, d in dag.in_degree() if d == 0]
        if len(roots) != 1:
            raise ValueError(
                f"Input DAG has {len(roots)} roots; this function requires exactly one.")

        dag_root_node = roots[0]

        tree = nx.DiGraph()
        dag_to_tree_map = {}

        replication_counters = defaultdict(int)

        tree_root_copy = copy.deepcopy(dag_root_node)
        dag_to_tree_map[dag_root_node] = tree_root_copy

        queue = deque([(tree_root_copy, dag_root_node)])

        while queue:
            parent_tree_node, current_dag_node = queue.popleft()

            for child_dag_node in dag.successors(current_dag_node):
                if child_dag_node in dag_to_tree_map:
                    self._replicate_and_attach_subgraph(
                        tree, dag, parent_tree_node, child_dag_node, replication_counters)
                else:
                    new_child_copy = copy.deepcopy(child_dag_node)
                    tree.add_edge(parent_tree_node, new_child_copy)
                    dag_to_tree_map[child_dag_node] = new_child_copy
                    queue.append((new_child_copy, child_dag_node))

        return tree

    def _create_new_taxo_file(self):
        with open(f"../data/{self.name}/{self.name}_train.taxo", "w") as f:
            for u, v in self.train_subgraph.edges():
                if u.tx_id != 'root':
                    f.write(f"{u.tx_id}\t{v.tx_id}\n")

        with open(f"../data/{self.name}/{self.name}_val.taxo", "w") as f:
            for u, v in self.val_graph.edges():
                if u.tx_id != 'root':
                    f.write(f"{u.tx_id}\t{v.tx_id}\n")

        with open(f"../data/{self.name}/{self.name}_test.taxo", "w") as f:
            for u, v in self.test_graph.edges():
                if u.tx_id != 'root':
                    f.write(f"{u.tx_id}\t{v.tx_id}\n")

    def _append_new_terms_file(self):
        with open(f"../data/{self.name}/{self.name}.terms", "a") as f:

            f.write(f"{'root'}\t{'root'}\n")


if __name__ == '__main__':
    MultiParentDataset(name='semeval_food', path="../data/semeval_food",
                       existing_partition=False, partition_pattern='leaf')
