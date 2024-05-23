"""Tests tree visualisation"""

from ..visualise import visualize_tree_structure
from ..tree_structures import Node, Tree
import random, string
from typing import List


def prep():
    """Does prep, creates tree from random pieces of text and 50 nodes."""

    def get_random_text():
        length = random.randint(50, 150)
        chars = string.ascii_letters + string.digits + string.punctuation
        return "".join(random.choices(chars, k=length))

    nodes = [
        Node(text=get_random_text(), index=idx, children=set(), embeddings=[])
        for idx in range(50)
    ]
    random.shuffle(nodes)
    root_nodes = nodes[:5]
    layer_1_nodes = nodes[5:15]
    leaf_nodes = nodes[15:]

    # Now make connections bw
    # root -> layer_1_nodes
    # layer_1_node -> leaf_nodes
    def randomly_connect_nodes(
        parent_nodes: List[Node], potential_children: List[Node]
    ):
        for child in potential_children:
            parent_node = random.sample(parent_nodes, k=1)[0]
            parent_node.children.add(child.index)

    randomly_connect_nodes(root_nodes, layer_1_nodes)
    randomly_connect_nodes(layer_1_nodes, leaf_nodes)

    return Tree(
        all_nodes={idx: node for idx, node in enumerate(nodes)},
        root_nodes={idx : node for idx, node in enumerate(root_nodes)},
        leaf_nodes={idx : node for idx, node in enumerate(leaf_nodes)},
        num_layers=2,
        layer_to_nodes={"0": root_nodes, "1": layer_1_nodes, "2": leaf_nodes},
    )


tree = prep()
# Now create a new root Node on top of all root nodes
root_node = Node(
    "SIRE HERE",
    index=-1,
    children=list(map(lambda x: x.index, tree.root_nodes.values())),
    embeddings=[],
)
visualize_tree_structure(root_node, tree)
