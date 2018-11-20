"""

Source Code from https://github.com/stefankoegl/kdtree

"""


class Node(object):
    """
    init a kdtree node
    """

    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right


class KDNode(Node):
    def __init__(self, data=Node, left=Node, right=Node, axis=Node, sel_axis=Node, dimensions=Node):
        """
        create a new kdtree node

        if the node is used in the kdtree, axis and sel_axis must to be applied
        sel_axis and axis will be used when it creates the now child node
        input is the father node's axis, output is the child node's axis

        :param data: data set
        :param left: data set left-child
        :param right: data set right-child
        :param axis: data axis
        :param sel_axis: next data axis(like right-child data axis)
        :param dimensions: data dimensions
        """
        super(KDNode, self).__init__(data, left, right)

        self.axis = axis
        self.sel_axis = sel_axis
        self.dimensions = dimensions


def create(point_list=None, dimensions=None, axis=0, sel_axis=0):
    """

    create a kdtree from a input points list
    assume all points is a same dimensions
    if the point_list is NULL, then will create a NULL kdtree
    if the point_list and dimensions are supllied, the dimension of the point_list must be the dimensions
    axis denotes the location where will be splited, sel_axis will be used when the child node will be create,
    and it will return the child's axis

    :param point_list: kdtree node list
    :param dimensions: node dimensions
    :param axis: node axis
    :param sel_axis: next node axis
    :return: kdtree
    """
    if point_list is None and dimensions is None:
        raise ValueError('either point_list or dimensions should be provided!')
    elif point_list:
        check_dimensionality(point_list, dimensions)

    # 这里每次切分直接取下个一维度，而不是取所有维度中方差最大的维度
    sel_axis = sel_axis or (lambda prev_axis: (prev_axis + 1) % dimensions)

    if point_list is None:
        return KDNode(sel_axis=sel_axis, axis=axis, dimensions=dimensions)

    # 对point_list 按照axis升序排列，取中位数对用的坐标点
    point_list = list(point_list)
    point_list.sort(key=lambda point: point[axis])
    median = len(point_list) // 2

    loc = point_list[median]
    left = create(point_list[:median], dimensions, sel_axis(axis))
    right = create(point_list[median + 1:], dimensions, sel_axis(axis))

    return KDNode(loc, left, right, axis=axis, sel_axis=sel_axis, dimensions=dimensions)


def check_dimensionality(point_list, dimensions=None):
    """

    check the point_list's dimension

    :param point_list:
    :param dimensions:
    :return: point_list's dimension
    """
    dimensions = dimensions or len(point_list[0])
    for p in point_list:
        if len(p) != dimensions:
            raise ValueError('All Points in the point_list must have the same dimensionality')

    return dimensions


class BoundedPriorityQueue:
    """
    max heap
    """

    def __init__(self, k):
        self.heap = []
        self.k = k

    def items(self):
        return self.heap

    def parent(self, index):
        """
        :param index:
        :return: index of the father node
        """
        return int(index / 2)

    def left_child(self, index):
        return 2 * index + 1

    def right_child(self, index):
        return 2 * index + 2

    def _dist(self, index):
        """返回index对应的距离"""
        return self.heap[index][3]

    def max_heapify(self, index):
        """
        负责维护最大堆的属性，即使当前节点的所有子节点值均小于该父节点
        """
        left_index = self.left_child(index)
        right_index = self.right_child(index)

        largest = index
        if left_index < len(self.heap) and self._dist(left_index) > self._dist(index):
            largest = left_index
        if right_index < len(self.heap) and self._dist(right_index) > self._dist(largest):
            largest = right_index
        if largest != index:
            self.heap[index], self.heap[largest] = self.heap[largest], self.heap[index]
            self.max_heapify(largest)

    def propagate_up(self, index):
        """
         在index位置添加新元素后，通过不断和父节点比较并交换
        维持最大堆的特性，即保持堆中父节点的值永远大于子节点
        """
        while index != 0 and self._dist(self.parent(index)) < self._dist(index):
            self.heap[index], self.heap[self.parent(index)] = self.heap[self.parent(index)], self.heap[index]
            index = self.parent(index)

    def add(self, obj):
        """
        如果当前值小于优先队列中的最大值，则将obj添加入队列，
        如果队列已满，则移除最大值再添加，这时原队列中的最大值、
        将被obj取代
        """
        size = self.size()
        if size == self.k:
            max_elem = self.max()
            if obj[1] < max_elem:
                self.extract_max()
                self.heap_append(obj)
        else:
            self.heap_append(obj)

    def heap_append(self, obj):
        """
        向队列中添加一个obj
        """
        self.heap.append(obj)
        self.propagate_up(self.size() - 1)

    def size(self):
        return len(self.heap)

    def max(self):
        return self.heap[0][4]

    def extract_max(self):
        """
        将最大值从队列中移除，同时从新对队列排序
        """
        max = self.heap[0]
        data = self.heap.pop()
        if len(self.heap) > 0:
            self.heap[0] = data
            self.max_heapify(0)

        return max

    def _search_node(self, point, k, results, get_dist):
        if not self:
            return
        nodeDist = get_dist(self)

        # 如果当前节点小于队列中至少一个节点，则将该节点添加入队列
        # 该功能由BoundedPriorityQueue类实现
        results.add(self, nodeDist)

        # 获得当前节点的切分平面
        split_plane = self.data[self.axis]
        plane_dist = point[self.axis] - split_plane
        plane_dist2 = plane_dist ** 2

        # 从根节点递归向下访问，若point的axis维小于且分点坐标
        # 则移动到左子节点，否则移动到右子节点
        if point[self.axis] < split_plane:
            if self.left is not None:
                self.left._search_node(point, k, results, get_dist)
        else:
            if self.right is not None:
                self.right._search_node(point, k, results, get_dist)

        # 检查父节点的另一子节点是否存在比当前子节点更近的点
        # 判断另一区域是否与当前最近邻的圆相交
        if plane_dist2 < results.max() or results.size() < k:
            if point[self.axis] < self.data[self.axis]:
                if self.right is not None:
                    self.right._search_node(point, k, results, get_dist)
            else:
                if self.left is not None:
                    self.left._search_node(point, k, results, get_dist)

    def search_knn(self, point, k, dist=None):
        """
        返回k个离point最近的点及它们的距离
        """

        if dist is None:
            get_dist = lambda n: n.dist(point)
        else:
            gen_dist = lambda n: dist(n.data, point)

        results = BoundedPriorityQueue(k)
        self._search_node(point, k, results, get_dist)

        # 将最后的结果按照距离排序
        BY_VALUE = lambda kv: kv[1]
        return sorted(results.items(), key=BY_VALUE)
