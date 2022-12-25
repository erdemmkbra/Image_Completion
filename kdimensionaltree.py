from __future__ import division, print_function, absolute_import
import sys
import numpy as np


def minkowski_distance_p(x, y, p=2):
    x = np.asarray(x)
    y = np.asarray(y)
    if p == np.inf:
        return np.amax(np.abs(y-x), axis=-1)
    elif p == 1:
        return np.sum(np.abs(y-x), axis=-1)
    else:
        return np.sum(np.abs(y-x)**p, axis=-1)

class KDTree(object):
    def __init__(self, data, leafsize=10, tau=0):
        self.data = np.asarray(data)
        self.n, self.m = np.shape(self.data)
        self.leafsize = int(leafsize)
        if self.leafsize < 1:
            raise ValueError("leafsize must be at least 1")
        self.maxes = np.amax(self.data,axis=0)
        self.mins = np.amin(self.data,axis=0)
        self.tau = tau
        self.tree = self.__build(np.arange(self.n), self.maxes, self.mins)

    ##, verilen bir veri kümesine ait bir KDTree (K-Dimensional Tree, K-Boyutlu Ağaç) yapısını oluşturur.
    # KDTree, veri kümesinde bulunan noktaların düzenlenmesine yardımcı olur ve bu noktaları hızlı bir şekilde arama,
    # sıralama gibi işlemlerde kullanılabilir. KDTree yapısı, self.__build fonksiyonu ile oluşturulur. Bu fonksiyon, verilen veri kümesinin
    #  indekslerini, veri kümesinin her bir özelliği için en yüksek ve en düşük değeri ve tau değişkenini alır. tau değişkeni, veri kümesinin
    #  nasıl düzenleneceğini belirler ve 0 olarak verilmiştir.
    # Yapı, veri kümesinin her bir özelliği için en yüksek ve en düşük değerlerini hesaplar ve bu değerleri 
    # self.maxes ve self.mins değişkenlerinde saklar. Daha sonra, self.__build fonksiyonu çağrılır ve yapı oluşturulur.
    #  Oluşturulan yapı, self.tree değişkeninde saklanır.



    class node(object):
        if sys.version_info[0] >= 3:
            def __lt__(self, other):
                return id(self) < id(other)

            def __gt__(self, other):
                return id(self) > id(other)

            def __le__(self, other):
                return id(self) <= id(other)

            def __ge__(self, other):
                return id(self) >= id(other)

            def __eq__(self, other):
                return id(self) == id(other)

    class leafnode(node):
        def __init__(self, idx):
            self.idx = idx
            self.children = len(idx)

    class innernode(node):
        def __init__(self, split_dim, split, less, greater):
            self.split_dim = split_dim
            self.split = split
            self.less = less
            self.greater = greater
            self.children = less.children+greater.children

    def __build(self, idx, maxes, mins):
        if len(idx) <= self.leafsize:
            return KDTree.leafnode(idx)
        else:
            data = self.data[idx]
            d = np.argmax(maxes-mins)
            maxval = maxes[d]
            minval = mins[d]
            if maxval == minval:
                return KDTree.leafnode(idx)
            data = data[:,d]
            split = (maxval+minval)/2
            less_idx = np.nonzero(data <= split)[0]
            greater_idx = np.nonzero(data > split)[0]
            if len(less_idx) == 0:
                split = np.amin(data)
                less_idx = np.nonzero(data <= split)[0]
                greater_idx = np.nonzero(data > split)[0]
            if len(greater_idx) == 0:
                split = np.amax(data)
                less_idx = np.nonzero(data < split)[0]
                greater_idx = np.nonzero(data >= split)[0]
            if len(less_idx) == 0:
                if not np.all(data == data[0]):
                    raise ValueError("Troublesome data array: %s" % data)
                split = data[0]
                less_idx = np.arange(len(data)-1)
                greater_idx = np.array([len(data)-1])

            lessmaxes = np.copy(maxes)
            lessmaxes[d] = split
            greatermins = np.copy(mins)
            greatermins[d] = split
            return KDTree.innernode(d, split, self.__build(idx[less_idx],lessmaxes,mins), self.__build(idx[greater_idx],maxes,greatermins))

def get_query_leaf(x, node):
    if isinstance(node, KDTree.leafnode):
        return node.idx
    else:
        if x[node.split_dim] < node.split:
            return get_query_leaf(x, node.less)
        else:
            return get_query_leaf(x, node.greater)

def get_annf_offsets(queries, indices, root, tau):
    leaves = [None]*len(queries)
    offsets = [None]*len(queries)
    distances = np.full(len(queries), np.inf)
    for i in range(len(queries)):
        leaves[i] = data = get_query_leaf(queries[i], root)
        if i-1 > 0:
            data = np.concatenate((data, leaves[i-1]))
        for j in range(len(data)):
            if np.abs(indices[i][0] - indices[data[j]][0]) > tau and np.abs(indices[i][1] - indices[data[j]][1]) > tau:
                dist = minkowski_distance_p(queries[i], queries[data[j]])
                if dist < distances[i]:
                    distances[i] = dist
                    offsets[i] = [indices[data[j]][0] - indices[i][0], indices[data[j]][1] - indices[i][1]]
    return distances, offsets    

