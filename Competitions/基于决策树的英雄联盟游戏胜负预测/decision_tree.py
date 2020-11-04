# encoding: utf8

import collections
import numpy as np


class Node(object):
    """Tree node"""
    def __init__(self, column=None, value=None, left=None, right=None, data=None):
        self.column = column
        self.value = value
        self.left = left
        self.right = right
        self.data = data

    @property
    def is_leaf(self):
        return self.data is not None

    def __str__(self):
        return 'Tree node column index: %s value:%s' % (self.column, self.value)


# sentinel node
empty = Node()


class DecisionTree(object):
    def __init__(self, classes, features, max_depth=10,
                 min_samples_split=10, impurity_t='entropy'):
        """
        :param classes: 表示模型分类总共有几类
        :param features: 每个特征的名字，也方便查询总的共特征数
        :param max_depth: 构建决策树时的最大深度
        :param min_samples_split: 构建决策树分裂节点时，如果到达该节点的样本数小于该值则不再分裂
        :param impurity_t: 计算混杂度（不纯度）的计算方式，例如entropy或gini
        """
        self.classes = classes
        self.features = features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.impurity_t = impurity_t
        self.root = empty

    @staticmethod
    def entropy(labels: np.ndarray):
        """Calculate entropy."""
        assert isinstance(labels, np.ndarray)

        n_labels = len(labels)
        counter_labels = list(collections.Counter(labels).values())
        probs = np.array(counter_labels) / n_labels # NOQA
        return -np.sum([p*np.log2(p) for p in probs])

    def gain(self, set1, set2):
        """Calculate split sets information gain."""
        assert isinstance(set1, np.ndarray)
        assert isinstance(set2, np.ndarray)

        total_set = np.concatenate((set1, set2))
        before_split_entropy = self.entropy(total_set)
        after_split_entropy = np.sum([self.entropy(s) * len(s) / len(total_set) for s in (set1, set2)])
        return before_split_entropy - after_split_entropy

    @staticmethod
    def _split_set(xs, column, value):
        """
        Split set.
        :param xs: split set
        :param column: column index
        :param value: compare value
        :return split set row index
        """

        set1_idx = []
        set2_idx = []

        for row in range(len(xs)):
            if xs[row, column] <= value:
                set1_idx.append(row)
            else:
                set2_idx.append(row)

        return set1_idx, set2_idx

    def build_tree(self, xs, ys, depth=1):
        """
        Build decision tree recursively.

        :param xs: features
        :param ys: labels
        :param depth: tree depth start from 1
        :return tree node.
        """
        max_gain = 0.0
        best_column = None
        best_value = None
        best_split_set1 = None
        best_split_set2 = None

        # stop split
        # case1 all the labels are same
        if len(np.unique(ys)) == 1:
            return Node(data=ys)

        # case2 all the input are same
        for col in range(xs.shape[1]):
            if len(np.unique(xs[:, col])) > 1:
                break
        else:
            return Node(data=ys)

        # pre-pruning
        # min_samples_split
        if len(ys) < self.min_samples_split:
            # print('pre-pruning min_samples_split')
            return Node(data=ys)

        # max_depth
        if depth > self.max_depth:
            # print('pre-pruning max_depth')
            return Node(data=ys)

        # find best split feature and value
        for col in range(len(self.features)):
            for val in np.unique(xs[:, col]):
                set1_idx, set2_idx = self._split_set(xs, col, val)

                gain = self.gain(ys[set1_idx], ys[set2_idx])
                if gain > max_gain:
                    max_gain = gain
                    best_column = col
                    best_value = val
                    best_split_set1 = set1_idx
                    best_split_set2 = set2_idx

        node = Node(best_column, best_value)
        node.left = self.build_tree(xs[best_split_set1, :], ys[best_split_set1], depth+1)
        node.right = self.build_tree(xs[best_split_set2, :], ys[best_split_set2], depth+1)

        return node

    def traverse_tree(self, x):
        """Traverse decision tree."""
        assert self.root != empty

        root = self.root
        while True:
            if root.is_leaf:
                # leaf node
                return collections.Counter(root.data).most_common(1)[0][0]

            if x[root.column] < root.value:
                root = root.left
            else:
                root = root.right

    def fit(self, xs, y):
        """
        训练模型
        xs为二维numpy（n*m）数组，每行表示一个样本，有m个特征
        label为一维numpy（n）数组，表示每个样本的分类标签

        提示：一种可能的实现方式为
        self.root = self.build_tree(xs, y, depth=1) # 从根节点开始分裂，模型记录根节点
        """
        assert len(self.features) == len(xs[0])
        self.root = self.build_tree(xs, y)

    def predict(self, xs):
        """
        预测
        输入feature可以是一个一维numpy数组也可以是一个二维numpy数组
        如果是一维numpy（m）数组则是一个样本，包含m个特征，返回一个类别值
        如果是二维numpy（n*m）数组则表示n个样本，每个样本包含m个特征，返回一个numpy一维数组

        提示：一种可能的实现方式为
        """
        assert len(xs.shape) in (1, 2)

        if len(xs.shape) == 1:
            return np.array([self.traverse_tree(xs)])

        return np.array([self.traverse_tree(x) for x in xs])


if __name__ == '__main__':
    xs = np.random.randint(0, 4, (100, 20))
    ys = np.random.randint(0, 2, 100)

    feats = ['feat' + str(i) for i in range(0, 20)]
    clf = DecisionTree(classes=[0, 1], features=feats)
    clf.fit(xs, ys)
    print('train done.')
