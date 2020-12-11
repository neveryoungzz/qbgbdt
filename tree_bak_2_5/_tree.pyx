# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Lars Buitinck
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

from cpython cimport Py_INCREF, PyObject
import scipy
from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.math cimport fabs
from libc.math cimport erf, exp, abs
from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdio cimport printf
import numpy as np
cimport numpy as np
np.import_array()

from scipy.sparse import issparse
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix

from ._utils cimport Stack
from ._utils cimport StackRecord
from ._utils cimport PriorityHeap
from ._utils cimport PriorityHeapRecord
from ._utils cimport safe_realloc
from ._utils cimport sizet_ptr_to_ndarray
from ._utils cimport PredStackRecord
from ._utils cimport Pred_Stack
from ._utils cimport PredGradStackRecord
from ._utils cimport Pred_Grad_Stack

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(object subtype, np.dtype descr,
                                int nd, np.npy_intp* dims,
                                np.npy_intp* strides,
                                void* data, int flags, object obj)

# =============================================================================
# Types and constants
# =============================================================================

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

cdef double INFINITY = np.inf
cdef double EPSILON = np.finfo('double').eps

# Some handy constants (BestFirstTreeBuilder)
cdef int IS_FIRST = 1
cdef int IS_NOT_FIRST = 0
cdef int IS_LEFT = 1
cdef int IS_NOT_LEFT = 0

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef SIZE_t INITIAL_STACK_SIZE = 10
cdef double SQRT05 = 0.7071067811865476
cdef double SQRT2 = 1.4142135623730951
cdef double INVSQRT2PI = 0.3989422804014327

# Repeat struct definition for numpy
NODE_DTYPE = np.dtype({
    'names': ['left_child', 'right_child', 'feature', 'threshold', 'impurity',
              'n_node_samples', 'weighted_n_node_samples', 
              'up_prob_bound', 'low_prob_bound'],
    'formats': [np.intp, np.intp, np.intp, np.float64, np.float64, np.intp,
                np.float64, np.float32, np.float32],
    'offsets': [
        <Py_ssize_t> &(<Node*> NULL).left_child,
        <Py_ssize_t> &(<Node*> NULL).right_child,
        <Py_ssize_t> &(<Node*> NULL).feature,
        <Py_ssize_t> &(<Node*> NULL).threshold,
        <Py_ssize_t> &(<Node*> NULL).impurity,
        <Py_ssize_t> &(<Node*> NULL).n_node_samples,
        <Py_ssize_t> &(<Node*> NULL).weighted_n_node_samples,
        <Py_ssize_t> &(<Node*> NULL).up_prob_bound,
        <Py_ssize_t> &(<Node*> NULL).low_prob_bound
    ]
})

# =============================================================================
# TreeBuilder
# =============================================================================

cdef class TreeBuilder:
    """Interface for different tree building strategies."""

    cpdef build(self, Tree tree, object X, object invsig, 
                np.ndarray y,
                np.ndarray sample_weight=None,
                np.ndarray X_idx_sorted=None):
        """Build a decision tree from the training set (X, y)."""
        pass

    cdef inline _check_input(self, object X, object invsig, np.ndarray y,
                             np.ndarray sample_weight):
        """Check input dtype, layout and format"""
        if issparse(X):
            X = X.tocsc()
            X.sort_indices()

            if X.data.dtype != DTYPE:
                X.data = np.ascontiguousarray(X.data, dtype=DTYPE)

            if X.indices.dtype != np.int32 or X.indptr.dtype != np.int32:
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")

        elif X.dtype != DTYPE:
            # since we have to copy we will make it fortran for efficiency
            X = np.asfortranarray(X, dtype=DTYPE)
        
        if invsig.dtype != DTYPE:
            invsig = np.asfortranarray(invsig, dtype = DTYPE)

        if y.dtype != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if (sample_weight is not None and
            (sample_weight.dtype != DOUBLE or
            not sample_weight.flags.contiguous)):
                sample_weight = np.asarray(sample_weight, dtype=DOUBLE,
                                           order="C")

        return X, invsig, y, sample_weight

# Depth first builder ---------------------------------------------------------

cdef class DepthFirstTreeBuilder(TreeBuilder):
    """Build a decision tree in depth-first fashion."""

    def __cinit__(self, Splitter splitter, SIZE_t min_samples_split,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  SIZE_t max_depth, double min_impurity_decrease,
                  double min_impurity_split):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        #self.alpha = alpha
        #self.beta = beta
        #self.y_bar = NULL
        #self.d = NULL

    cpdef build(self, Tree tree, object X, object invsig, 
                np.ndarray y,
                np.ndarray sample_weight=None,
                np.ndarray X_idx_sorted=None):
        """Build a decision tree from the training set (X, y)."""
        # check input
        X, invsig, y, sample_weight = self._check_input(X, invsig, y, sample_weight)

        cdef DOUBLE_t* sample_weight_ptr = NULL
        if sample_weight is not None:
            sample_weight_ptr = <DOUBLE_t*> sample_weight.data

        # Initial capacity
        cdef int init_capacity

        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)
        cdef int init_prob_capacity = 2047
       
        # Parameters
        cdef Splitter splitter = self.splitter
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split
        cdef double min_impurity_decrease = self.min_impurity_decrease
        cdef double min_impurity_split = self.min_impurity_split

        # Recursive partition (without actual recursion)
        #splitter.init(X, invsig, y, sample_weight_ptr, X_idx_sorted)

        #cdef SIZE_t start
        #cdef SIZE_t end
        cdef SIZE_t* samples
        cdef SIZE_t** all_samples
        cdef double* prob_weight
        cdef double* _prob_weight
        cdef double** all_prob_weight
        cdef double* left_prob_weight
        cdef SIZE_t weight_node_id

        cdef DTYPE_t* up_prob_bounds = NULL
        cdef DTYPE_t* low_prob_bounds = NULL
        cdef DTYPE_t* _up_prob_bounds = NULL
        cdef DTYPE_t* _low_prob_bounds = NULL
        cdef DTYPE_t** all_prob_bounds = NULL
        cdef SIZE_t prob_bound_id = 0

        cdef SIZE_t depth
        cdef SIZE_t parent
        cdef bint is_left
        #cdef SIZE_t n_node_samples = splitter.n_samples
        #cdef double weighted_n_samples = splitter.weighted_n_samples
        cdef SIZE_t n_node_samples = 0
        cdef double weighted_n_node_samples
        cdef SplitRecord split
        cdef SIZE_t node_id

        cdef double threshold
        cdef double impurity = INFINITY
        cdef SIZE_t n_constant_features
        cdef bint is_leaf
        cdef bint first = 1
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0
        cdef SIZE_t i = 0
        cdef Stack stack = Stack(INITIAL_STACK_SIZE)
        cdef StackRecord stack_record

        cdef SIZE_t* left_samples = NULL
        cdef SIZE_t* right_samples = NULL
        cdef SIZE_t left_n_node_samples = 0
        cdef SIZE_t right_n_node_samples = 0 
        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t n_invsigs = invsig.shape[1]
        cdef bint is_prob = 0
        #cdef bint is_past_prob = 0
        #cdef DOUBLE_t* y_bar = NULL
        #cdef DOUBLE_t* d = NULL
        #cdef double alpha = self.alpha
        self.n_samples = n_samples
        self.n_invsigs = n_invsigs
        
        samples = <SIZE_t*> malloc(n_samples * sizeof(SIZE_t))
        all_samples = <SIZE_t**> malloc(init_prob_capacity * sizeof(SIZE_t*))
        weight_node_id = 0
        all_samples[weight_node_id] = samples
        prob_weight = <double*> malloc(n_samples * sizeof(double))
        for i in range(n_samples):
            prob_weight[i] = 1
        all_prob_weight = <double**> malloc(init_prob_capacity * sizeof(double*))
        all_prob_weight[weight_node_id] = prob_weight
        weight_node_id += 1
        left_prob_weight = <double*> malloc(n_samples * sizeof(double))
        memset(left_prob_weight, 0, n_samples * sizeof(double))
        splitter.init(X, invsig, y, samples, sample_weight_ptr, X_idx_sorted)
        up_prob_bounds = <DTYPE_t*> malloc(n_invsigs * sizeof(DTYPE_t))
        for i in range(n_invsigs):
            up_prob_bounds[i] = <DTYPE_t> np.inf
        low_prob_bounds = <DTYPE_t*> malloc(n_invsigs * sizeof(DTYPE_t))
        for i in range(n_invsigs):
            low_prob_bounds[i] = <DTYPE_t> (- np.inf)
        prob_bound_id = 0
        all_prob_bounds = <DTYPE_t**> malloc(init_prob_capacity * 2 * sizeof(DTYPE_t*))
        all_prob_bounds[prob_bound_id] = up_prob_bounds
        prob_bound_id += 1
        all_prob_bounds[prob_bound_id] = low_prob_bounds
        prob_bound_id += 1
        #y_bar = <DOUBLE_t*> malloc(n_samples * sizeof(DOUBLE_t))
        #d = <DOUBLE_t*> malloc(n_samples * sizeof(DOUBLE_t))
        #for i in range(n_samples):
        #    y_bar[i] = 0
        #    d[i] = 0
        with nogil:
            # push root node onto stack
            #rc = stack.push(0, n_node_samples, 0, _TREE_UNDEFINED, 0, INFINITY, 0)
            rc = stack.push(samples, n_samples, 0, _TREE_UNDEFINED, 0, INFINITY, 
                            0, prob_weight, up_prob_bounds, low_prob_bounds, 0)
            if rc == -1:
                # got return code -1 - out-of-memory
                with gil:
                    raise MemoryError()

            while not stack.is_empty():
                stack.pop(&stack_record)
                #start = stack_record.start
                #end = stack_record.end
                samples = stack_record.samples
                n_node_samples = stack_record.n_node_samples
                depth = stack_record.depth
                parent = stack_record.parent
                is_left = stack_record.is_left
                impurity = stack_record.impurity
                n_constant_features = stack_record.n_constant_features
                prob_weight = stack_record.prob_weight
                up_prob_bounds = stack_record.up_prob_bounds
                low_prob_bounds = stack_record.low_prob_bounds
                is_past_prob = stack_record.is_past_prob
                #n_node_samples = end - start
                #splitter.node_reset(start, end, &weighted_n_node_samples)
                splitter.node_reset(samples, n_node_samples, &weighted_n_node_samples, 
                                    prob_weight)
                
                is_leaf = (depth >= max_depth or
                           n_node_samples < min_samples_split or
                           n_node_samples < 2 * min_samples_leaf or
                           weighted_n_node_samples < 2 * min_weight_leaf)

                if first:
                    impurity = splitter.node_impurity()
                    first = 0

                is_leaf = (is_leaf or
                           (impurity <= min_impurity_split))

                if not is_leaf:
                    memset(left_prob_weight, 0, n_samples * sizeof(double))
                    left_samples = NULL
                    right_samples = NULL
                    splitter.node_split(impurity, &split, &n_constant_features,
                                        &left_samples, &left_n_node_samples, &right_samples,
                                        &right_n_node_samples, left_prob_weight, &is_prob,
                                        up_prob_bounds, low_prob_bounds)
                    # If EPSILON=0 in the below comparison, float precision
                    # issues stop splitting, producing trees that are
                    # dissimilar to v0.18
                    #is_leaf = (is_leaf or split.pos >= end or
                    #           (split.improvement + EPSILON <
                    #            min_impurity_decrease))
                    
                    is_leaf = (is_leaf or split.pos >= n_node_samples or
                                (split.improvement + EPSILON <
                                 min_impurity_decrease))
                
                if split.feature < n_invsigs:
                    node_id = tree._add_node(parent, is_left, is_leaf, split.feature,
                                            split.threshold, impurity, n_node_samples,
                                            weighted_n_node_samples, 
                                            up_prob_bounds[split.feature],
                                            low_prob_bounds[split.feature])
                else:
                    node_id = tree._add_node(parent, is_left, is_leaf, split.feature,
                                            split.threshold, impurity, n_node_samples,
                                            weighted_n_node_samples,
                                            -10000, 10000)
                if node_id == <SIZE_t>(-1):
                    rc = -1
                    break

                    # Store value for all nodes, to facilitate tree/model
                    # inspection and interpretation
                splitter.node_value(tree.value + node_id * tree.value_stride)

                if not is_leaf:
                    if is_prob:
                        _prob_weight = <double*> malloc(n_samples * sizeof(double))
                        memset(_prob_weight, 0, n_samples * sizeof(double))
                        for p in range(left_n_node_samples):
                            _prob_weight[left_samples[p]] = prob_weight[left_samples[p]] * left_prob_weight[left_samples[p]]
                        all_prob_weight[weight_node_id] = _prob_weight
                        all_samples[weight_node_id] = left_samples
                        weight_node_id += 1
                        _up_prob_bounds = <DTYPE_t*> malloc(n_invsigs * sizeof(DTYPE_t))
                        all_prob_bounds[prob_bound_id] = _up_prob_bounds
                        prob_bound_id += 1
                        memcpy(_up_prob_bounds, up_prob_bounds, n_invsigs * sizeof(DTYPE_t))
                        _low_prob_bounds = <DTYPE_t*> malloc(n_invsigs * sizeof(DTYPE_t))
                        all_prob_bounds[prob_bound_id] = _low_prob_bounds
                        prob_bound_id += 1
                        memcpy(_low_prob_bounds, low_prob_bounds, n_invsigs * sizeof(DTYPE_t))
                        _up_prob_bounds[split.feature] = split.threshold
                        # Push right child on stack
                        #rc = stack.push(split.pos, end, depth + 1, node_id, 0,
                        #                split.impurity_right, n_constant_features,
                        #                left_prob_weight, new_prob_weight)
                        rc = stack.push(left_samples, left_n_node_samples, depth + 1, node_id, 1,
                                        split.impurity_left, n_constant_features,
                                        _prob_weight, _up_prob_bounds, _low_prob_bounds, is_prob)
                        if rc == -1:
                            break
                        _prob_weight = <double*> malloc(n_samples * sizeof(double))
                        memset(_prob_weight, 0, n_samples * sizeof(double))
                        for p in range(right_n_node_samples):
                            _prob_weight[right_samples[p]] = prob_weight[right_samples[p]] * (1 - left_prob_weight[right_samples[p]])
                        all_prob_weight[weight_node_id] = _prob_weight
                        all_samples[weight_node_id] = right_samples
                        weight_node_id += 1
                        _up_prob_bounds = <DTYPE_t*> malloc(n_invsigs * sizeof(DTYPE_t))
                        all_prob_bounds[prob_bound_id] = _up_prob_bounds
                        prob_bound_id += 1
                        memcpy(_up_prob_bounds, up_prob_bounds, n_invsigs * sizeof(DTYPE_t))
                        _low_prob_bounds = <DTYPE_t*> malloc(n_invsigs * sizeof(DTYPE_t))
                        all_prob_bounds[prob_bound_id] = _low_prob_bounds
                        prob_bound_id += 1
                        memcpy(_low_prob_bounds, low_prob_bounds, n_invsigs * sizeof(DTYPE_t))
                        _low_prob_bounds[split.feature] = split.threshold
                        # Push left child on stack
                        #rc = stack.push(start, split.pos, depth + 1, node_id, 1,
                        #                split.impurity_left, n_constant_features,
                        #                left_prob_weight, new_prob_weight)
                        rc = stack.push(right_samples, right_n_node_samples, depth + 1, node_id, 0,
                                        split.impurity_right, n_constant_features,
                                        _prob_weight, _up_prob_bounds, _low_prob_bounds, is_prob)
                        if rc == -1:
                            break
                    else:
                        _prob_weight = <double*> malloc(n_samples * sizeof(double))
                        memset(_prob_weight, 0, n_samples * sizeof(double))
                        for p in range(left_n_node_samples):
                            _prob_weight[left_samples[p]] = prob_weight[left_samples[p]]
                        all_prob_weight[weight_node_id] = _prob_weight
                        all_samples[weight_node_id] = left_samples
                        weight_node_id += 1
                        # Push right child on stack
                        #rc = stack.push(split.pos, end, depth + 1, node_id, 0,
                        #                split.impurity_right, n_constant_features,
                        #                left_prob_weight, new_prob_weight)
                        rc = stack.push(left_samples, left_n_node_samples, depth + 1, node_id, 1,
                                        split.impurity_left, n_constant_features,
                                        _prob_weight, up_prob_bounds, low_prob_bounds, 
                                        is_past_prob)
                        if rc == -1:
                            break
                        _prob_weight = <double*> malloc(n_samples * sizeof(double))
                        memset(_prob_weight, 0, n_samples * sizeof(double))
                        for p in range(right_n_node_samples):
                            _prob_weight[right_samples[p]] = prob_weight[right_samples[p]]
                        all_prob_weight[weight_node_id] = _prob_weight
                        all_samples[weight_node_id] = right_samples
                        weight_node_id += 1
                        # Push left child on stack
                        #rc = stack.push(start, split.pos, depth + 1, node_id, 1,
                        #                split.impurity_left, n_constant_features,
                        #                left_prob_weight, new_prob_weight)
                        rc = stack.push(right_samples, right_n_node_samples, depth + 1, node_id, 0,
                                        split.impurity_right, n_constant_features,
                                        _prob_weight, up_prob_bounds, low_prob_bounds, 
                                        is_past_prob)
                        if rc == -1:
                            break
                
                if depth > max_depth_seen:
                    max_depth_seen = depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen
        if rc == -1:
            raise MemoryError()

    #    self.y_bar = y_bar
    #    self.d = d
        #tree._add_d(d, n_samples)
        free(left_prob_weight)
        for i in range(weight_node_id):
            free(all_prob_weight[i])
            free(all_samples[i])
        for i in range(prob_bound_id):
            free(all_prob_bounds[i])
        self.all_samples = all_samples
        self.all_prob_weight = all_prob_weight
        self.all_prob_bounds = all_prob_bounds

    def __dealloc__(self):
        free(self.all_samples)
        free(self.all_prob_weight)
        free(self.all_prob_bounds)
        
# =============================================================================
# Tree
# =============================================================================

cdef class Tree:
    """Array-based representation of a binary decision tree.

    The binary tree is represented as a number of parallel arrays. The i-th
    element of each array holds information about the node `i`. Node 0 is the
    tree's root. You can find a detailed description of all arrays in
    `_tree.pxd`. NOTE: Some of the arrays only apply to either leaves or split
    nodes, resp. In this case the values of nodes of the other type are
    arbitrary!

    Attributes
    ----------
    node_count : int
        The number of nodes (internal nodes + leaves) in the tree.

    capacity : int
        The current capacity (i.e., size) of the arrays, which is at least as
        great as `node_count`.

    max_depth : int
        The depth of the tree, i.e. the maximum depth of its leaves.

    children_left : array of int, shape [node_count]
        children_left[i] holds the node id of the left child of node i.
        For leaves, children_left[i] == TREE_LEAF. Otherwise,
        children_left[i] > i. This child handles the case where
        X[:, feature[i]] <= threshold[i].

    children_right : array of int, shape [node_count]
        children_right[i] holds the node id of the right child of node i.
        For leaves, children_right[i] == TREE_LEAF. Otherwise,
        children_right[i] > i. This child handles the case where
        X[:, feature[i]] > threshold[i].

    feature : array of int, shape [node_count]
        feature[i] holds the feature to split on, for the internal node i.

    threshold : array of double, shape [node_count]
        threshold[i] holds the threshold for the internal node i.

    value : array of double, shape [node_count, n_outputs, max_n_classes]
        Contains the constant prediction value of each node.

    impurity : array of double, shape [node_count]
        impurity[i] holds the impurity (i.e., the value of the splitting
        criterion) at node i.

    n_node_samples : array of int, shape [node_count]
        n_node_samples[i] holds the number of training samples reaching node i.

    weighted_n_node_samples : array of int, shape [node_count]
        weighted_n_node_samples[i] holds the weighted number of training samples
        reaching node i.
    """
    # Wrap for outside world.
    # WARNING: these reference the current `nodes` and `value` buffers, which
    # must not be freed by a subsequent memory allocation.
    # (i.e. through `_resize` or `__setstate__`)
    property n_classes:
        def __get__(self):
            return sizet_ptr_to_ndarray(self.n_classes, self.n_outputs)

    property children_left:
        def __get__(self):
            return self._get_node_ndarray()['left_child'][:self.node_count]

    property children_right:
        def __get__(self):
            return self._get_node_ndarray()['right_child'][:self.node_count]

    property n_leaves:
        def __get__(self):
            return np.sum(np.logical_and(
                self.children_left == -1,
                self.children_right == -1))

    property feature:
        def __get__(self):
            return self._get_node_ndarray()['feature'][:self.node_count]

    property threshold:
        def __get__(self):
            return self._get_node_ndarray()['threshold'][:self.node_count]

    property impurity:
        def __get__(self):
            return self._get_node_ndarray()['impurity'][:self.node_count]

    property n_node_samples:
        def __get__(self):
            return self._get_node_ndarray()['n_node_samples'][:self.node_count]

    property weighted_n_node_samples:
        def __get__(self):
            return self._get_node_ndarray()['weighted_n_node_samples'][:self.node_count]

    property value:
        def __get__(self):
            return self._get_value_ndarray()[:self.node_count]

    #property d:
    #    def __get__(self):
    #        return self._get_d_ndarray()[:self.n]

    def __cinit__(self, int n_features, np.ndarray[SIZE_t, ndim=1] n_classes,
                  int n_outputs):
        """Constructor."""
        # Input/Output layout
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_classes = NULL
        safe_realloc(&self.n_classes, n_outputs)

        self.max_n_classes = np.max(n_classes)
        self.value_stride = n_outputs * self.max_n_classes

        cdef SIZE_t k
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

        # Inner structures
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        self.value = NULL
        self.nodes = NULL
        self.n_predict_samples = 0
        cdef init_prob_capacity = 1024
        self.pred_stack = Pred_Stack(init_prob_capacity)
        self.pred_grad_stack = Pred_Grad_Stack(init_prob_capacity)
        #self.n = 0
        #self.d = NULL

    def __dealloc__(self):
        """Destructor."""
        # Free all inner structures
        free(self.n_classes)
        free(self.value)
        free(self.nodes)
        #if self.n > 0:
        #    free(self.d)

    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        return (Tree, (self.n_features,
                       sizet_ptr_to_ndarray(self.n_classes, self.n_outputs),
                       self.n_outputs), self.__getstate__())

    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        d = {}
        # capacity is inferred during the __setstate__ using nodes
        d["max_depth"] = self.max_depth
        d["node_count"] = self.node_count
        d["nodes"] = self._get_node_ndarray()
        d["values"] = self._get_value_ndarray()
        return d

    def __setstate__(self, d):
        """Setstate re-implementation, for unpickling."""
        self.max_depth = d["max_depth"]
        self.node_count = d["node_count"]

        if 'nodes' not in d:
            raise ValueError('You have loaded Tree version which '
                             'cannot be imported')

        node_ndarray = d['nodes']
        value_ndarray = d['values']

        value_shape = (node_ndarray.shape[0], self.n_outputs,
                       self.max_n_classes)
        if (node_ndarray.ndim != 1 or
                node_ndarray.dtype != NODE_DTYPE or
                not node_ndarray.flags.c_contiguous or
                value_ndarray.shape != value_shape or
                not value_ndarray.flags.c_contiguous or
                value_ndarray.dtype != np.float64):
            raise ValueError('Did not recognise loaded array layout')

        self.capacity = node_ndarray.shape[0]
        if self._resize_c(self.capacity) != 0:
            raise MemoryError("resizing tree to %d" % self.capacity)
        nodes = memcpy(self.nodes, (<np.ndarray> node_ndarray).data,
                       self.capacity * sizeof(Node))
        value = memcpy(self.value, (<np.ndarray> value_ndarray).data,
                       self.capacity * self.value_stride * sizeof(double))

    cdef int _resize(self, SIZE_t capacity) nogil except -1:
        """Resize all inner arrays to `capacity`, if `capacity` == -1, then
           double the size of the inner arrays.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if self._resize_c(capacity) != 0:
            # Acquire gil only if we need to raise
            with gil:
                raise MemoryError()

    # XXX using (size_t)(-1) is ugly, but SIZE_MAX is not available in C89
    # (i.e., older MSVC).
    cdef int _resize_c(self, SIZE_t capacity=<SIZE_t>(-1)) nogil except -1:
        """Guts of _resize

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if capacity == self.capacity and self.nodes != NULL:
            return 0

        if capacity == <SIZE_t>(-1):
            if self.capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * self.capacity

        safe_realloc(&self.nodes, capacity)
        safe_realloc(&self.value, capacity * self.value_stride)

        # value memory is initialised to 0 to enable classifier argmax
        if capacity > self.capacity:
            memset(<void*>(self.value + self.capacity * self.value_stride), 0,
                   (capacity - self.capacity) * self.value_stride *
                   sizeof(double))

        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity

        self.capacity = capacity
        return 0

    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, double threshold, double impurity,
                          SIZE_t n_node_samples,
                          double weighted_n_node_samples,
                          DTYPE_t up_prob_bound, DTYPE_t low_prob_bound) nogil except -1:
        """Add a node to the tree.

        The new node registers itself as the child of its parent.

        Returns (size_t)(-1) on error.
        """
        cdef SIZE_t node_id = self.node_count

        if node_id >= self.capacity:
            if self._resize_c() != 0:
                return <SIZE_t>(-1)

        cdef Node* node = &self.nodes[node_id]
        node.impurity = impurity
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples
        node.up_prob_bound = up_prob_bound
        node.low_prob_bound = low_prob_bound

        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id

        if is_leaf:
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = _TREE_UNDEFINED
            node.threshold = _TREE_UNDEFINED

        else:
            # left_child and right_child will be set later
            node.feature = feature
            node.threshold = threshold

        self.node_count += 1

        return node_id

    cpdef np.ndarray predict(self, object X):
        """Predict target for X."""
        out = self._get_value_ndarray().take(self.apply(X), axis=0,
                                             mode='clip')
        if self.n_outputs == 1:
            out = out.reshape(X.shape[0], self.max_n_classes)
        return out

    cpdef np.ndarray apply(self, object X):
        """Finds the terminal region (=leaf node) for each sample in X."""
        if issparse(X):
            return self._apply_sparse_csr(X)
        else:
            return self._apply_dense(X)

    cdef inline np.ndarray _apply_dense(self, object X):
        """Finds the terminal region (=leaf node) for each sample in X."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray X_ndarray = X
        cdef DTYPE_t* X_ptr = <DTYPE_t*> X_ndarray.data
        cdef SIZE_t X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
        cdef SIZE_t X_fx_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize
        cdef SIZE_t n_samples = X.shape[0]

        # Initialize output
        cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.intp)
        cdef SIZE_t* out_ptr = <SIZE_t*> out.data

        # Initialize auxiliary data-structure
        cdef Node* node = NULL
        cdef SIZE_t i = 0

        with nogil:
            for i in range(n_samples):
                node = self.nodes
                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    if X_ptr[X_sample_stride * i +
                             X_fx_stride * node.feature] <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset

        return out
    
    #cpdef np.ndarray predict_prob(self, object X, object invsig):
    #    value_arr = self._get_value_ndarray().transpose((1, 2, 0))
    #    value_arr[np.isnan(value_arr)] = 0
    #    out = np.inner(self._apply_prob_dense(X, invsig), value_arr)
    #    if self.n_outputs == 1:
    #        out = out.reshape(X.shape[0], self.max_n_classes)
    #    return out
    cpdef int reweight(self, object X, object invsig, object grad, object hess):
        cdef SIZE_t leaf = 0
        cdef SIZE_t node_count = self.node_count
        cdef DOUBLE_t* grad_node = <DOUBLE_t*> malloc(node_count * sizeof(DOUBLE_t))
        cdef DOUBLE_t* hess_node = <DOUBLE_t*> malloc(node_count * sizeof(DOUBLE_t))
        memset(grad_node, 0, node_count * sizeof(DOUBLE_t))
        memset(hess_node, 0, node_count * sizeof(DOUBLE_t))
        cdef np.ndarray grad_ndarray = grad
        cdef DOUBLE_t* grad_ptr = <DOUBLE_t*> grad_ndarray.data
        cdef np.ndarray hess_ndarray = hess
        cdef DOUBLE_t* hess_ptr = <DOUBLE_t*> hess_ndarray.data
        children_left = self._get_node_ndarray()['left_child'][:self.node_count]
        leaves = np.where(children_left == -1)[0]
        self._apply_prob_dense(X, invsig, grad_node, hess_node, grad_ptr, hess_ptr)
        for leaf in leaves:
            if abs(hess[leaf]) < 1e-6:
                self.value[leaf] = 0
            else:
                self.value[leaf] = grad_node[leaf] / hess_node[leaf]
            self.value[leaf] = max(self.value[leaf], - 1.0e6)
            self.value[leaf] = min(self.value[leaf], 1.0e6)
        free(grad_node)
        free(hess_node)
        return 0

    #cpdef int apply_prob(self, object X, object invsig, DOUBLE_t* grad_node, 
    #                     DOUBLE_t* hess_node, DOUBLE_t* grad, DOUBLE_t* hess):
    #    self._apply_prob_dense(X, invsig, grad_node, hess_node, grad, hess)
    #    return 1
    
    cdef inline int _apply_prob_dense(self, object X, object invsig, 
                                      DOUBLE_t* grad_node, DOUBLE_t* hess_node,
                                      DOUBLE_t* grad, DOUBLE_t* hess):
        """Finds the terminal region (=leaf node) for each sample in X."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray X_ndarray = X
        cdef DTYPE_t* X_ptr = <DTYPE_t*> X_ndarray.data
        cdef SIZE_t X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
        cdef SIZE_t X_fx_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize
        cdef SIZE_t n_samples = X.shape[0]

        cdef np.ndarray invsig_ndarray = invsig
        cdef DTYPE_t* invsig_ptr = <DTYPE_t*> invsig_ndarray.data
        cdef SIZE_t invsig_sample_stride = <SIZE_t> invsig.strides[0] / <SIZE_t> invsig.itemsize
        cdef SIZE_t invsig_fx_stride = <SIZE_t> invsig.strides[1] / <SIZE_t> invsig.itemsize
        cdef SIZE_t n_invsigs = invsig.shape[1]

        # Initialize output
        #cdef np.ndarray out
        #cdef int init_prob_capacity = 1024
        #if self.first_predict:
        #    self.predict_out = <double*> malloc(n_samples * self.node_count * sizeof(double))
            #self.pred_stack = Pred_Stack(init_prob_capacity)
            #self.pred_grad_stack = Pred_Grad_Stack(init_prob_capacity)
        #    self.first_predict = 0
        #elif self.n_predict_samples != n_samples:
        #    free(self.predict_out)
        #    self.predict_out = <double*> malloc(n_samples * self.node_count * sizeof(double))
        #self.n_predict_samples = n_samples
        #cdef double* out_data = self.predict_out
        cdef Pred_Stack pred_stack = self.pred_stack
        #memset(out_data, 0, n_samples * self.node_count * sizeof(double))
        #cdef SIZE_t shape[2]
        #shape[0] = n_samples
        #shape[1] = self.node_count
        # Initialize auxiliary data-structure
        cdef Node* _node = NULL
        cdef SIZE_t i = 0
        cdef double _prob_weight_value = 1
        cdef DTYPE_t _x
        cdef DTYPE_t _invsig = 1
        cdef DTYPE_t _z
        cdef double _left_prob_weight_value = 0
        cdef PredStackRecord predstackrecord
        cdef int rc = 0
        cdef double _left_weight
        cdef double _right_weight
        with nogil:
            for i in range(n_samples):
                _node = self.nodes
                    # While node not a leaf
                    #while node.left_child != _TREE_LEAF:
                        # ... and node.right_child != _TREE_LEAF:
                    #    if X_ptr[X_sample_stride * i +
                    #             X_fx_stride * node.feature] <= node.threshold:
                    #        node = &self.nodes[node.left_child]
                    #    else:
                    #        node = &self.nodes[node.right_child]
                _prob_weight_value = 1
                rc = pred_stack.reset()
                rc = pred_stack.push(_node, _prob_weight_value)
                while not pred_stack.is_empty():
                    pred_stack.pop(&predstackrecord)
                    _node = predstackrecord.node
                    _prob_weight_value = predstackrecord.prob_weight
                    _x = X_ptr[X_sample_stride * i + 
                            X_fx_stride * _node.feature]
                    if _node.left_child != _TREE_LEAF:
                        if _node.feature < n_invsigs:
                            _z = _x - _node.threshold
                            _invsig = invsig_ptr[invsig_sample_stride * i +
                                                invsig_fx_stride * _node.feature]
                            #_left_prob_weight_value = 0.5 * erf(-_z * SQRT05 * _invsig) + 0.5
                            _left_weight = 0.5 * erf(-_z * SQRT05 * _invsig) \
                                           - 0.5 * erf((_node.low_prob_bound - _x) * SQRT05 * _invsig)
                            _right_weight = 0.5 * erf((_node.up_prob_bound - _x) * SQRT05 * _invsig) \
                                            - 0.5 * erf(-_z * SQRT05 * _invsig)
                            _left_prob_weight_value = _left_weight / (_left_weight + _right_weight)
                            
                            #if _z > 3 * _invsig:
                            if _left_prob_weight_value < 0.001:
                                rc = pred_stack.push(&self.nodes[_node.right_child], _prob_weight_value)
                            #elif _z < -3 * _invsig:
                            elif _left_prob_weight_value > 0.999:
                                rc = pred_stack.push(&self.nodes[_node.left_child], _prob_weight_value)
                            else:
                                rc = pred_stack.push(&self.nodes[_node.left_child], _prob_weight_value * _left_prob_weight_value)
                                rc = pred_stack.push(&self.nodes[_node.right_child], _prob_weight_value * (1 - _left_prob_weight_value))
                        else:
                            if _x <= _node.threshold:
                                rc = pred_stack.push(&self.nodes[_node.left_child], _prob_weight_value)
                            else:
                                rc = pred_stack.push(&self.nodes[_node.right_child], _prob_weight_value)
                    else:
                        #out_arr += _prob_weight_value * value_arr.take(<SIZE_t> (_node - self.nodes), axis = 0, mode = 'clip')
                        #out_data[<SIZE_t> (i * self.node_count + _node - self.nodes)] = _prob_weight_value
                        grad_node[<SIZE_t> (_node - self.nodes)] += grad[i] * _prob_weight_value
                        hess_node[<SIZE_t> (_node - self.nodes)] += hess[i] * _prob_weight_value
        return 1
        #out = np.PyArray_SimpleNewFromData(2, shape, np.NPY_DOUBLE, out_data)
        #Py_INCREF(self)
        #out.base = <PyObject*> self
        #return out

    cpdef np.ndarray predict_prob(self, object X, object invsig):
        cdef SIZE_t n_samples = X.shape[0]
        pred = np.zeros(n_samples, dtype = np.double)
        cdef np.ndarray pred_ndarray = pred
        cdef DOUBLE_t* pred_ptr = <DOUBLE_t*> pred_ndarray.data
        self._predict_prob_dense(X, invsig, pred_ptr)
        return pred.reshape((X.shape[0], self.max_n_classes))

    cdef inline int _predict_prob_dense(self, object X, object invsig, DOUBLE_t* pred):
        """Finds the terminal region (=leaf node) for each sample in X."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray X_ndarray = X
        cdef DTYPE_t* X_ptr = <DTYPE_t*> X_ndarray.data
        cdef SIZE_t X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
        cdef SIZE_t X_fx_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize
        cdef SIZE_t n_samples = X.shape[0]

        cdef np.ndarray invsig_ndarray = invsig
        cdef DTYPE_t* invsig_ptr = <DTYPE_t*> invsig_ndarray.data
        cdef SIZE_t invsig_sample_stride = <SIZE_t> invsig.strides[0] / <SIZE_t> invsig.itemsize
        cdef SIZE_t invsig_fx_stride = <SIZE_t> invsig.strides[1] / <SIZE_t> invsig.itemsize
        cdef SIZE_t n_invsigs = invsig.shape[1]

        value_arr = self._get_value_ndarray()
        cdef np.ndarray value_ndarray = value_arr
        cdef DOUBLE_t* value_ptr = <DOUBLE_t*> value_ndarray.data
        value_arr[np.isnan(value_arr)] = 0
        cdef SIZE_t value_node_stride = <SIZE_t> value_arr.strides[0] / <SIZE_t> value_arr.itemsize
        cdef SIZE_t value_outputs_stride = <SIZE_t> value_arr.strides[1] / <SIZE_t> value_arr.itemsize
        cdef SIZE_t value_classes_stride = <SIZE_t> value_arr.strides[2] / <SIZE_t> value_arr.itemsize
        
        # Initialize output
        cdef int init_prob_capacity = 1024
        #if self.first_predict:
            #self.predict_out = <double*> malloc(n_samples * self.node_count * sizeof(double))
        #    self.pred_stack = Pred_Stack(init_prob_capacity)
        #    self.pred_grad_stack = Pred_Grad_Stack(init_prob_capacity)
        #    self.first_predict = 0
        #elif self.n_predict_samples != n_samples:
        #    free(self.predict_out)
        #    self.predict_out = <double*> malloc(n_samples * self.node_count * sizeof(double))
        #self.n_predict_samples = n_samples
        #cdef double* out_data = self.predict_out
        cdef Pred_Stack pred_stack = self.pred_stack
        #memset(out_data, 0, n_samples * self.node_count * sizeof(double))
        #cdef SIZE_t shape[2]
        #shape[0] = n_samples
        #shape[1] = self.node_count
        # Initialize auxiliary data-structure
        cdef Node* _node = NULL
        cdef SIZE_t i = 0
        cdef double _prob_weight_value = 1
        cdef DTYPE_t _x
        cdef DTYPE_t _invsig = 1
        cdef DTYPE_t _z
        cdef double _left_prob_weight_value = 0
        cdef PredStackRecord predstackrecord
        cdef int rc = 0
        cdef double _left_weight
        cdef double _right_weight
        cdef DOUBLE_t _value
        with nogil:
            for i in range(n_samples):
                _node = self.nodes
                    # While node not a leaf
                    #while node.left_child != _TREE_LEAF:
                        # ... and node.right_child != _TREE_LEAF:
                    #    if X_ptr[X_sample_stride * i +
                    #             X_fx_stride * node.feature] <= node.threshold:
                    #        node = &self.nodes[node.left_child]
                    #    else:
                    #        node = &self.nodes[node.right_child]
                _prob_weight_value = 1
                rc = pred_stack.reset()
                rc = pred_stack.push(_node, _prob_weight_value)
                while not pred_stack.is_empty():
                    pred_stack.pop(&predstackrecord)
                    _node = predstackrecord.node
                    _prob_weight_value = predstackrecord.prob_weight
                    _x = X_ptr[X_sample_stride * i + 
                            X_fx_stride * _node.feature]
                    if _node.left_child != _TREE_LEAF:
                        if _node.feature < n_invsigs:
                            _z = _x - _node.threshold
                            _invsig = invsig_ptr[invsig_sample_stride * i +
                                                invsig_fx_stride * _node.feature]
                            #_left_prob_weight_value = 0.5 * erf(-_z * SQRT05 * _invsig) + 0.5
                            _left_weight = 0.5 * erf(-_z * SQRT05 * _invsig) \
                                           - 0.5 * erf((_node.low_prob_bound - _x) * SQRT05 * _invsig)
                            _right_weight = 0.5 * erf((_node.up_prob_bound - _x) * SQRT05 * _invsig) \
                                            - 0.5 * erf(-_z * SQRT05 * _invsig)
                            _left_prob_weight_value = _left_weight / (_left_weight + _right_weight)
                            
                            #if _z > 3 * _invsig:
                            if _left_prob_weight_value < 0.001:
                                rc = pred_stack.push(&self.nodes[_node.right_child], _prob_weight_value)
                            #elif _z < -3 * _invsig:
                            elif _left_prob_weight_value > 0.999:
                                rc = pred_stack.push(&self.nodes[_node.left_child], _prob_weight_value)
                            else:
                                rc = pred_stack.push(&self.nodes[_node.left_child], _prob_weight_value * _left_prob_weight_value)
                                rc = pred_stack.push(&self.nodes[_node.right_child], _prob_weight_value * (1 - _left_prob_weight_value))
                        else:
                            if _x <= _node.threshold:
                                rc = pred_stack.push(&self.nodes[_node.left_child], _prob_weight_value)
                            else:
                                rc = pred_stack.push(&self.nodes[_node.right_child], _prob_weight_value)
                    else:
                        #out_arr += _prob_weight_value * value_arr.take(<SIZE_t> (_node - self.nodes), axis = 0, mode = 'clip')
                        #out_data[<SIZE_t> (i * self.node_count + _node - self.nodes)] = _prob_weight_value
                        _value = value_ptr[<SIZE_t> ((_node - self.nodes) * value_node_stride)]
                        pred[i] += _prob_weight_value * _value
        #out = np.PyArray_SimpleNewFromData(2, shape, np.NPY_DOUBLE, out_data)
        #Py_INCREF(self)
        #out.base = <PyObject*> self
        return 1

    cpdef apply_prob_grad(self, object X, object invsig, SIZE_t feature):
        cdef SIZE_t n_samples = X.shape[0]
        pred = np.zeros((n_samples, self.max_n_classes), dtype = np.double)
        grad1 = np.zeros((n_samples, self.max_n_classes), dtype = np.double)
        grad2 = np.zeros((n_samples, self.max_n_classes), dtype = np.double)
        grad4 = np.zeros((n_samples, self.max_n_classes), dtype = np.double)
        #d = np.zeros((n_samples, self.max_n_classes), dtype = np.double)
        #dgrad1 = np.zeros((n_samples, self.max_n_classes), dtype = np.double)
        #dgrad2 = np.zeros((n_samples, self.max_n_classes), dtype = np.double)
        #dgrad4 = np.zeros((n_samples, self.max_n_classes), dtype = np.double)
        cdef np.ndarray pred_ndarray = pred
        cdef np.ndarray grad1_ndarray = grad1
        cdef np.ndarray grad2_ndarray = grad2
        cdef np.ndarray grad4_ndarray = grad4
        #cdef np.ndarray d_ndarray = d
        #cdef np.ndarray dgrad1_ndarray = dgrad1
        #cdef np.ndarray dgrad2_ndarray = dgrad2
        #cdef np.ndarray dgrad4_ndarray = dgrad4
        cdef DOUBLE_t* pred_ptr = <DOUBLE_t*> pred_ndarray.data
        cdef DOUBLE_t* grad1_ptr = <DOUBLE_t*> grad1_ndarray.data
        cdef DOUBLE_t* grad2_ptr = <DOUBLE_t*> grad2_ndarray.data
        cdef DOUBLE_t* grad4_ptr = <DOUBLE_t*> grad4_ndarray.data
        #cdef DOUBLE_t* d_ptr = <DOUBLE_t*> d_ndarray.data
        #cdef DOUBLE_t* dgrad1_ptr = <DOUBLE_t*> dgrad1_ndarray.data
        #cdef DOUBLE_t* dgrad2_ptr = <DOUBLE_t*> dgrad2_ndarray.data
        #cdef DOUBLE_t* dgrad4_ptr = <DOUBLE_t*> dgrad4_ndarray.data
        #self._apply_prob_grad_dense(X, invsig, feature, pred_ptr, 
        #                            grad1_ptr, grad2_ptr, grad4_ptr, d_ptr,
        #                            dgrad1_ptr, dgrad2_ptr, dgrad4_ptr)
        #return pred, grad1, grad2, grad4, d, dgrad1, dgrad2, dgrad4
        self._apply_prob_grad_dense(X, invsig, feature, pred_ptr, 
                                    grad1_ptr, grad2_ptr, grad4_ptr)
        return pred, grad1, grad2, grad4

    #cdef inline int _apply_prob_grad_dense(self, object X, object invsig, 
    #                                       SIZE_t feature, DOUBLE_t* pred, 
    #                                       DOUBLE_t* grad1, DOUBLE_t* grad2, 
    #                                       DOUBLE_t* grad4, DOUBLE_t* d, 
    #                                       DOUBLE_t* dgrad1, DOUBLE_t* dgrad2,
    #                                       DOUBLE_t* dgrad4):
    cdef inline int _apply_prob_grad_dense(self, object X, object invsig, 
                                           SIZE_t feature, DOUBLE_t* pred, 
                                           DOUBLE_t* grad1, DOUBLE_t* grad2, 
                                           DOUBLE_t* grad4):
        """Finds the terminal region (=leaf node) for each sample in X."""
        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray X_ndarray = X
        cdef DTYPE_t* X_ptr = <DTYPE_t*> X_ndarray.data
        cdef SIZE_t X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
        cdef SIZE_t X_fx_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize
        cdef SIZE_t n_samples = X.shape[0]

        cdef np.ndarray invsig_ndarray = invsig
        cdef DTYPE_t* invsig_ptr = <DTYPE_t*> invsig_ndarray.data
        cdef SIZE_t invsig_sample_stride = <SIZE_t> invsig.strides[0] / <SIZE_t> invsig.itemsize
        cdef SIZE_t invsig_fx_stride = <SIZE_t> invsig.strides[1] / <SIZE_t> invsig.itemsize
        cdef SIZE_t n_invsigs = invsig.shape[1]

        value_arr = self._get_value_ndarray()
        cdef np.ndarray value_ndarray = value_arr
        cdef DOUBLE_t* value_ptr = <DOUBLE_t*> value_ndarray.data
        value_arr[np.isnan(value_arr)] = 0
        cdef SIZE_t value_node_stride = <SIZE_t> value_arr.strides[0] / <SIZE_t> value_arr.itemsize
        cdef SIZE_t value_outputs_stride = <SIZE_t> value_arr.strides[1] / <SIZE_t> value_arr.itemsize
        cdef SIZE_t value_classes_stride = <SIZE_t> value_arr.strides[2] / <SIZE_t> value_arr.itemsize
        
        #up_bounds_array = np.zeros((n_samples, self.node_count), dtype = np.float32)
        #low_bounds_array = np.zeros((n_samples, self.node_count), dtype = np.float32)
        #cdef np.ndarray up_bounds_ndarray = up_bounds_array
        #cdef np.ndarray low_bounds_ndarray = low_bounds_array
        #cdef DTYPE_t* up_bounds = <DTYPE_t*> up_bounds_ndarray.data
        #cdef DTYPE_t* low_bounds = <DTYPE_t*> low_bounds_ndarray.data
        #for i in range(n_samples):
        #    up_bounds[i * self.node_count] = 100000
        #    low_bounds[i * self.node_count] = - 100000
        cdef DOUBLE_t base_up_bound = 100000.0
        cdef DOUBLE_t base_low_bound = -100000.0

        # Initialize output
        #cdef np.ndarray out
        cdef int init_prob_capacity = 1024
        #if self.first_predict:
            #self.predict_out = <double*> malloc(n_samples * self.node_count * sizeof(double))
        #    self.pred_stack = Pred_Stack(init_prob_capacity)
        #    self.pred_grad_stack = Pred_Grad_Stack(init_prob_capacity)
        #    self.first_predict = 0
        #if self.n_predict_samples != n_samples:
            #free(self.predict_out)
            #self.predict_out = <double*> malloc(n_samples * self.node_count * sizeof(double))
        #self.n_predict_samples = n_samples
        #cdef double* out_data = self.predict_out
        cdef Pred_Grad_Stack pred_grad_stack = self.pred_grad_stack
        # Initialize auxiliary data-structure
        cdef Node* _node = NULL
        cdef SIZE_t i = 0
        cdef double _prob_weight_value = 1
        cdef DTYPE_t _x
        cdef DTYPE_t _invsig = 1
        cdef DTYPE_t _z
        cdef double _left_prob_weight_value = 0
        cdef PredGradStackRecord predgradstackrecord
        cdef int rc = 0
        cdef DTYPE_t up_bound
        cdef DTYPE_t low_bound
        cdef double _left_weight
        cdef double _right_weight
        cdef DOUBLE_t _value
        cdef DOUBLE_t _grad2
        cdef DOUBLE_t _grad41, _grad42, _grad43
        #cdef DOUBLE_t _sq_d_sum, _d_sum
        with nogil:
            for i in range(n_samples):
                _node = self.nodes
                    # While node not a leaf
                    #while node.left_child != _TREE_LEAF:
                        # ... and node.right_child != _TREE_LEAF:
                    #    if X_ptr[X_sample_stride * i +
                    #             X_fx_stride * node.feature] <= node.threshold:
                    #        node = &self.nodes[node.left_child]
                    #    else:
                    #        node = &self.nodes[node.right_child]
                _prob_weight_value = 1
                #up_bound = up_bounds[i * self.node_count]
                #low_bound = low_bounds[i * self.node_count]
                up_bound = base_up_bound
                low_bound = base_low_bound
                #_sq_d_sum = 0
                #_d_sum = 0
                rc = pred_grad_stack.reset()
                rc = pred_grad_stack.push(_node, _prob_weight_value, up_bound, low_bound)
                while not pred_grad_stack.is_empty():
                    pred_grad_stack.pop(&predgradstackrecord)
                    _node = predgradstackrecord.node
                    _prob_weight_value = predgradstackrecord.prob_weight
                    up_bound = predgradstackrecord.up_bound
                    low_bound = predgradstackrecord.low_bound
                    _x = X_ptr[X_sample_stride * i + 
                            X_fx_stride * _node.feature]
                    if _node.left_child != _TREE_LEAF:
                        if _node.feature < n_invsigs:
                            if _node.feature != feature:
                                _z = _x - _node.threshold
                                _invsig = invsig_ptr[invsig_sample_stride * i +
                                                    invsig_fx_stride * _node.feature]
                                #_left_prob_weight_value = 0.5 * erf(-_z * SQRT05 * _invsig) + 0.5
                                _left_weight = 0.5 * erf(-_z * SQRT05 * _invsig) \
                                               - 0.5 * erf((_node.low_prob_bound - _x) * SQRT05 * _invsig)
                                _right_weight = 0.5 * erf((_node.up_prob_bound - _x) * SQRT05 * _invsig) \
                                                - 0.5 * erf(-_z * SQRT05 * _invsig)
                                _left_prob_weight_value = _left_weight / (_left_weight + _right_weight)
                                
                                #if _z > 3 * _invsig:
                                if _left_prob_weight_value < 0.001:
                                    rc = pred_grad_stack.push(&self.nodes[_node.right_child], _prob_weight_value, 
                                                        up_bound, low_bound)
                                #elif _z < -3 * _invsig:
                                elif _left_prob_weight_value > 0.999:
                                    rc = pred_grad_stack.push(&self.nodes[_node.left_child], _prob_weight_value,
                                                         up_bound, low_bound)
                                else:
                                    rc = pred_grad_stack.push(&self.nodes[_node.left_child], 
                                                              _prob_weight_value * _left_prob_weight_value,
                                                              up_bound, low_bound)
                                    rc = pred_grad_stack.push(&self.nodes[_node.right_child], 
                                                              _prob_weight_value * (1 - _left_prob_weight_value),
                                                              up_bound, low_bound)
                            else:
                                _z = _x - _node.threshold
                                _invsig = invsig_ptr[invsig_sample_stride * i +
                                                    invsig_fx_stride * _node.feature]
                                #_left_prob_weight_value = 0.5 * erf(-_z * SQRT05 * _invsig) + 0.5
                                _left_weight = 0.5 * erf(-_z * SQRT05 * _invsig) \
                                            - 0.5 * erf((_node.low_prob_bound - _x) * SQRT05 * _invsig)
                                _right_weight = 0.5 * erf((_node.up_prob_bound - _x) * SQRT05 * _invsig) \
                                                - 0.5 * erf(-_z * SQRT05 * _invsig)
                                _left_prob_weight_value = _left_weight / (_left_weight + _right_weight)
                                
                                #if _z > 3 * _invsig:
                                if _left_prob_weight_value < 0.001:
                                    rc = pred_grad_stack.push(&self.nodes[_node.right_child], _prob_weight_value, 
                                                              up_bound, low_bound)
                                #elif _z < -3 * _invsig:
                                elif _left_prob_weight_value > 0.999:
                                    rc = pred_grad_stack.push(&self.nodes[_node.left_child], _prob_weight_value,
                                                              up_bound, low_bound)
                                else:
                                    rc = pred_grad_stack.push(&self.nodes[_node.left_child], 
                                                              _prob_weight_value * _left_prob_weight_value,
                                                              -_z * SQRT05 * _invsig, low_bound)
                                    rc = pred_grad_stack.push(&self.nodes[_node.right_child], 
                                                              _prob_weight_value * (1 - _left_prob_weight_value),
                                                              up_bound, -_z * SQRT05 * _invsig)
                        else:
                            if _x <= _node.threshold:
                                rc = pred_grad_stack.push(&self.nodes[_node.left_child], _prob_weight_value,
                                                          up_bound, low_bound)
                            else:
                                rc = pred_grad_stack.push(&self.nodes[_node.right_child], _prob_weight_value,
                                                          up_bound, low_bound)
                    else:
                        #out_arr += _prob_weight_value * value_arr.take(<SIZE_t> (_node - self.nodes), axis = 0, mode = 'clip')
                        #out_data[<SIZE_t> (i * self.node_count + _node - self.nodes)] = _prob_weight_value
                        #up_bounds[<SIZE_t> (i * self.node_count + _node - self.nodes)] = up_bound
                        #low_bounds[<SIZE_t> (i * self.node_count + _node - self.nodes)] = low_bound
                        if _prob_weight_value > 0:
                            _value = value_ptr[<SIZE_t> ((_node - self.nodes) * value_node_stride)]
                            #_invsig = invsig_ptr[invsig_sample_stride * i + invsig_fx_stride * _node.feature]
                            _invsig = invsig_ptr[invsig_sample_stride * i + invsig_fx_stride * feature]
                            pred[i] += _prob_weight_value * _value
                            grad1[i] += (- exp(- up_bound ** 2) + exp(- low_bound ** 2)) * _value * INVSQRT2PI * _invsig
                            _grad2 = (- up_bound * exp(- up_bound ** 2) + low_bound * exp(- low_bound ** 2)) * SQRT2 * INVSQRT2PI
                            _grad2 += 0.5 * (erf(up_bound) - erf(low_bound))
                            grad2[i] += _grad2 * _value
                            _grad41 = - up_bound ** 3 * exp(- up_bound ** 2) + low_bound ** 3 * exp(- low_bound ** 2)
                            _grad41 *= 2 * SQRT2 * INVSQRT2PI 
                            _grad42 = - up_bound * exp(- up_bound ** 2) + low_bound * exp(- low_bound ** 2)
                            _grad42 *= 3 * SQRT2 * INVSQRT2PI
                            _grad43 = 0.5 * (erf(up_bound) - erf(low_bound))
                            _grad43 *= 3
                            grad4[i] += (_grad41 + _grad42 + _grad43) * _value
                            #dgrad1[i] += (-exp(-up_bound ** 2) + exp(- low_bound ** 2)) * _value * _value * INVSQRT2PI * _invsig
                            #dgrad2[i] += _grad2 * _value *_value
                            #dgrad4[i] += (_grad41 + _grad42 + _grad43) * _value * _value
                            #_sq_d_sum += _value * _value * _prob_weight_value
                            #_d_sum += _value * _prob_weight_value
                #d[i] = _sq_d_sum - _d_sum * _d_sum
        #out = np.PyArray_SimpleNewFromData(2, shape, np.NPY_DOUBLE, out_data)
        #Py_INCREF(self)
        #out.base = <PyObject*> self
        return 0

    cdef inline np.ndarray _apply_sparse_csr(self, object X):
        """Finds the terminal region (=leaf node) for each sample in sparse X.
        """
        # Check input
        if not isinstance(X, csr_matrix):
            raise ValueError("X should be in csr_matrix format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray[ndim=1, dtype=DTYPE_t] X_data_ndarray = X.data
        cdef np.ndarray[ndim=1, dtype=INT32_t] X_indices_ndarray  = X.indices
        cdef np.ndarray[ndim=1, dtype=INT32_t] X_indptr_ndarray  = X.indptr

        cdef DTYPE_t* X_data = <DTYPE_t*>X_data_ndarray.data
        cdef INT32_t* X_indices = <INT32_t*>X_indices_ndarray.data
        cdef INT32_t* X_indptr = <INT32_t*>X_indptr_ndarray.data

        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t n_features = X.shape[1]

        # Initialize output
        cdef np.ndarray[SIZE_t, ndim=1] out = np.zeros((n_samples,),
                                                       dtype=np.intp)
        cdef SIZE_t* out_ptr = <SIZE_t*> out.data

        # Initialize auxiliary data-structure
        cdef DTYPE_t feature_value = 0.
        cdef Node* node = NULL
        cdef DTYPE_t* X_sample = NULL
        cdef SIZE_t i = 0
        cdef INT32_t k = 0

        # feature_to_sample as a data structure records the last seen sample
        # for each feature; functionally, it is an efficient way to identify
        # which features are nonzero in the present sample.
        cdef SIZE_t* feature_to_sample = NULL

        safe_realloc(&X_sample, n_features)
        safe_realloc(&feature_to_sample, n_features)

        with nogil:
            memset(feature_to_sample, -1, n_features * sizeof(SIZE_t))

            for i in range(n_samples):
                node = self.nodes

                for k in range(X_indptr[i], X_indptr[i + 1]):
                    feature_to_sample[X_indices[k]] = i
                    X_sample[X_indices[k]] = X_data[k]

                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    if feature_to_sample[node.feature] == i:
                        feature_value = X_sample[node.feature]

                    else:
                        feature_value = 0.

                    if feature_value <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset

            # Free auxiliary arrays
            free(X_sample)
            free(feature_to_sample)

        return out

    cpdef object decision_path(self, object X):
        """Finds the decision path (=node) for each sample in X."""
        if issparse(X):
            return self._decision_path_sparse_csr(X)
        else:
            return self._decision_path_dense(X)

    cdef inline object _decision_path_dense(self, object X):
        """Finds the decision path (=node) for each sample in X."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray X_ndarray = X
        cdef DTYPE_t* X_ptr = <DTYPE_t*> X_ndarray.data
        cdef SIZE_t X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
        cdef SIZE_t X_fx_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize
        cdef SIZE_t n_samples = X.shape[0]

        # Initialize output
        cdef np.ndarray[SIZE_t] indptr = np.zeros(n_samples + 1, dtype=np.intp)
        cdef SIZE_t* indptr_ptr = <SIZE_t*> indptr.data

        cdef np.ndarray[SIZE_t] indices = np.zeros(n_samples *
                                                   (1 + self.max_depth),
                                                   dtype=np.intp)
        cdef SIZE_t* indices_ptr = <SIZE_t*> indices.data

        # Initialize auxiliary data-structure
        cdef Node* node = NULL
        cdef SIZE_t i = 0

        with nogil:
            for i in range(n_samples):
                node = self.nodes
                indptr_ptr[i + 1] = indptr_ptr[i]

                # Add all external nodes
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                    indptr_ptr[i + 1] += 1

                    if X_ptr[X_sample_stride * i +
                             X_fx_stride * node.feature] <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                # Add the leave node
                indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                indptr_ptr[i + 1] += 1

        indices = indices[:indptr[n_samples]]
        cdef np.ndarray[SIZE_t] data = np.ones(shape=len(indices),
                                               dtype=np.intp)
        out = csr_matrix((data, indices, indptr),
                         shape=(n_samples, self.node_count))

        return out

    cdef inline object _decision_path_sparse_csr(self, object X):
        """Finds the decision path (=node) for each sample in X."""

        # Check input
        if not isinstance(X, csr_matrix):
            raise ValueError("X should be in csr_matrix format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray[ndim=1, dtype=DTYPE_t] X_data_ndarray = X.data
        cdef np.ndarray[ndim=1, dtype=INT32_t] X_indices_ndarray  = X.indices
        cdef np.ndarray[ndim=1, dtype=INT32_t] X_indptr_ndarray  = X.indptr

        cdef DTYPE_t* X_data = <DTYPE_t*>X_data_ndarray.data
        cdef INT32_t* X_indices = <INT32_t*>X_indices_ndarray.data
        cdef INT32_t* X_indptr = <INT32_t*>X_indptr_ndarray.data

        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t n_features = X.shape[1]

        # Initialize output
        cdef np.ndarray[SIZE_t] indptr = np.zeros(n_samples + 1, dtype=np.intp)
        cdef SIZE_t* indptr_ptr = <SIZE_t*> indptr.data

        cdef np.ndarray[SIZE_t] indices = np.zeros(n_samples *
                                                   (1 + self.max_depth),
                                                   dtype=np.intp)
        cdef SIZE_t* indices_ptr = <SIZE_t*> indices.data

        # Initialize auxiliary data-structure
        cdef DTYPE_t feature_value = 0.
        cdef Node* node = NULL
        cdef DTYPE_t* X_sample = NULL
        cdef SIZE_t i = 0
        cdef INT32_t k = 0

        # feature_to_sample as a data structure records the last seen sample
        # for each feature; functionally, it is an efficient way to identify
        # which features are nonzero in the present sample.
        cdef SIZE_t* feature_to_sample = NULL

        safe_realloc(&X_sample, n_features)
        safe_realloc(&feature_to_sample, n_features)

        with nogil:
            memset(feature_to_sample, -1, n_features * sizeof(SIZE_t))

            for i in range(n_samples):
                node = self.nodes
                indptr_ptr[i + 1] = indptr_ptr[i]

                for k in range(X_indptr[i], X_indptr[i + 1]):
                    feature_to_sample[X_indices[k]] = i
                    X_sample[X_indices[k]] = X_data[k]

                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:

                    indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                    indptr_ptr[i + 1] += 1

                    if feature_to_sample[node.feature] == i:
                        feature_value = X_sample[node.feature]

                    else:
                        feature_value = 0.

                    if feature_value <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                # Add the leave node
                indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                indptr_ptr[i + 1] += 1

            # Free auxiliary arrays
            free(X_sample)
            free(feature_to_sample)

        indices = indices[:indptr[n_samples]]
        cdef np.ndarray[SIZE_t] data = np.ones(shape=len(indices),
                                               dtype=np.intp)
        out = csr_matrix((data, indices, indptr),
                         shape=(n_samples, self.node_count))

        return out


    cpdef compute_feature_importances(self, normalize=True):
        """Computes the importance of each feature (aka variable)."""
        cdef Node* left
        cdef Node* right
        cdef Node* nodes = self.nodes
        cdef Node* node = nodes
        cdef Node* end_node = node + self.node_count

        cdef double normalizer = 0.

        cdef np.ndarray[np.float64_t, ndim=1] importances
        importances = np.zeros((self.n_features,))
        cdef DOUBLE_t* importance_data = <DOUBLE_t*>importances.data

        with nogil:
            while node != end_node:
                if node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    left = &nodes[node.left_child]
                    right = &nodes[node.right_child]

                    importance_data[node.feature] += (
                        node.weighted_n_node_samples * node.impurity -
                        left.weighted_n_node_samples * left.impurity -
                        right.weighted_n_node_samples * right.impurity)
                node += 1

        importances /= nodes[0].weighted_n_node_samples

        if normalize:
            normalizer = np.sum(importances)

            if normalizer > 0.0:
                # Avoid dividing by zero (e.g., when root is pure)
                importances /= normalizer

        return importances

    cdef np.ndarray _get_value_ndarray(self):
        """Wraps value as a 3-d NumPy array.

        The array keeps a reference to this Tree, which manages the underlying
        memory.
        """
        cdef np.npy_intp shape[3]
        shape[0] = <np.npy_intp> self.node_count
        shape[1] = <np.npy_intp> self.n_outputs
        shape[2] = <np.npy_intp> self.max_n_classes
        cdef np.ndarray arr
        arr = np.PyArray_SimpleNewFromData(3, shape, np.NPY_DOUBLE, self.value)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr

    cdef np.ndarray _get_node_ndarray(self):
        """Wraps nodes as a NumPy struct array.

        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        """
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.node_count
        cdef np.npy_intp strides[1]
        strides[0] = sizeof(Node)
        cdef np.ndarray arr
        Py_INCREF(NODE_DTYPE)
        arr = PyArray_NewFromDescr(np.ndarray, <np.dtype> NODE_DTYPE, 1, shape,
                                   strides, <void*> self.nodes,
                                   np.NPY_DEFAULT, None)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr
    '''
    cdef int _add_d(self, DOUBLE_t* d, SIZE_t n):
        cdef SIZE_t i
        if self.n > 0:
            free(self.d)
        self.d = <DOUBLE_t*> malloc(n * sizeof(DOUBLE_t))
        self.n = n
        memcpy(self.d, d, n * sizeof(DOUBLE_t))
        for i in range(n):
            if self.d[i] < 0:
                self.d[i] = 0
        return (0)

    cdef np.ndarray _get_d_ndarray(self):
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.n
        cdef np.ndarray arr
        arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, self.d)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr
    '''