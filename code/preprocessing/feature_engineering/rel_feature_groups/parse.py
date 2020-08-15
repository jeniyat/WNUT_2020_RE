from nltk import ParentedTree, Tree

from corpus.ProtoFile import Relation
from preprocessing.feature_engineering.datasets import RelationWindow


class ParseFeatureGroup(object):

    def __init__(self):
        pass

    def convert_window(self, window):
        result = []
        assert isinstance(window, RelationWindow)
        if window.relations is not None:
            for rel in window.relations:
                assert isinstance(rel, Relation)
                result.append([self.ptp(rel),  # combination of mention entity types
                               self.ptph(rel),
                               ])

        # print("done")
        return result

    @staticmethod
    def get_words(tokens):
        l= [token.word for token in tokens]
        if len(l)==0:
            l = [""]
        return l

    @staticmethod
    def get_lca_length(location1, location2):
        i = 0
        while i < len(location1) and i < len(location2) and location1[i] == location2[i]:
            i += 1
        return i

    @staticmethod
    def get_labels_from_lca(ptree, lca_len, location):
        labels = []
        for i in range(lca_len, len(location)):
            labels.append(ptree[location[:i]].label())
        return labels

    def get_idx(self, list1d, leaf_values):
        try:
            single_str = "".join(list1d)
            leaf_index = [leaf_values.index(i) for i in leaf_values if (single_str in i or i in single_str)]
            return leaf_index[-1]
        except IndexError:
            return 0

    def find_path(self, ptree, text1, text2):
        assert isinstance(ptree, Tree)
        leaf_values = ptree.leaves()
        leaf_index1 = self.get_idx(text1, leaf_values)
        leaf_index2 = self.get_idx(text2, leaf_values)

        location1 = ptree.leaf_treeposition(leaf_index1)
        location2 = ptree.leaf_treeposition(leaf_index2)

        # find length of least common ancestor (lca)
        lca_len = self.get_lca_length(location1, location2)

        # find path from the node1 to lca

        labels1 = self.get_labels_from_lca(ptree, lca_len, location1)
        # ignore the first element, because it will be counted in the second part of the path
        result = labels1[1:]
        # inverse, because we want to go from the node to least common ancestor
        result = result[::-1]

        # add path from lca to node2
        result = result + self.get_labels_from_lca(ptree, lca_len, location2)
        return result

    def ptp(self, rel):

        ptree = ParentedTree.convert(rel.parse_tree)
        # print(ptree.pprint())
        arg1_tokens = rel.get_arg1_tokens()
        arg1_words = self.get_words(arg1_tokens)
        arg2_tokens = rel.get_arg2_tokens()
        arg2_words = self.get_words(arg2_tokens)
        return "ptp={0}".format(self.find_path(ptree, arg1_words, arg2_words))

    def ptph(self, rel):
        ptree = ParentedTree.convert(rel.parse_tree)
        # print(ptree.pprint())
        arg1_tokens = rel.get_arg1_tokens()
        arg1_words = self.get_words(arg1_tokens)
        arg2_tokens = rel.get_arg2_tokens()
        arg2_words = self.get_words(arg2_tokens)
        return "ptp={0}".format(self.find_path(ptree, arg1_words, arg2_words))