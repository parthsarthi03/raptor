from typing import Dict, List, Set


class Node:
    """
    Represents a node in the hierarchical tree structure.
    """

    def __init__(self, text: str, index: int, children: Set[int], embeddings) -> None:
        self.text = text
        self.index = index
        self.children = children
        self.embeddings = embeddings


class Tree:
    """
    Represents the entire hierarchical tree structure.
    """

    def __init__(
        self, all_nodes, root_nodes, leaf_nodes, num_layers, layer_to_nodes
    ) -> None:
        self.all_nodes = all_nodes
        self.root_nodes = root_nodes
        self.leaf_nodes = leaf_nodes
        self.num_layers = num_layers
        self.layer_to_nodes = layer_to_nodes
    def hang_tree(self, another_tree) -> None:
        """
        Hangs another tree on the current tree.
        """
        index_shift = len(self.all_nodes)
        for idx, node in another_tree.all_nodes.items():
            another_tree.all_nodes[idx].index += index_shift
            another_tree.all_nodes[idx].children = {i + index_shift for i in node.children}
        self.all_nodes.update({index_shift + i : j for i, j in another_tree.all_nodes.items()})
        self.root_nodes.update({index_shift + i : j for i, j in another_tree.root_nodes.items()})
        self.leaf_nodes.update({index_shift + i : j for i, j in another_tree.leaf_nodes.items()})
        self.num_layers = max(self.num_layers, another_tree.num_layers)
        for i in range(self.num_layers + 1):
            for node in another_tree.layer_to_nodes.get(i, []):
                node.index += index_shift
        self.layer_to_nodes = {i : self.layer_to_nodes.get(i, []) + another_tree.layer_to_nodes.get(i, []) for i in range(self.num_layers + 1)}
