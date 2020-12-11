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
#
# License: BSD 3 clause

from ._criterion cimport Criterion

from libc.stdlib cimport free
from libc.stdlib cimport qsort
from libc.stdlib cimport malloc
from libc.string cimport memcpy
from libc.string cimport memset
from libc.math cimport erf
from libc.stdio cimport printf

import numpy as np
cimport numpy as np
np.import_array()

from scipy.sparse import csc_matrix

from ._utils cimport log
from ._utils cimport rand_int
from ._utils cimport rand_uniform
from ._utils cimport RAND_R_MAX
from ._utils cimport safe_realloc

cdef double INFINITY = np.inf
cdef double SQRT05 = 0.7071067811865476
# Mitigate precision differences between 32 bit and 64 bit
cdef DTYPE_t FEATURE_THRESHOLD = 1e-7

# Constant to switch between algorithm non zero value extract algorithm
# in SparseSplitter
cdef DTYPE_t EXTRACT_NNZ_SWITCH = 0.1

cdef inline void _init_split(SplitRecord* self, SIZE_t start_pos) nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY

cdef class Splitter:
    """Abstract splitter class.

    Splitters are called by tree builders to find the best splits on both
    sparse and dense data, one split at a time.
    """

    def __cinit__(self, Criterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  object random_state, bint presort):
        """
        Parameters
        ----------
        criterion : Criterion
            The criterion to measure the quality of a split.

        max_features : SIZE_t
            The maximal number of randomly selected features which can be
            considered for a split.

        min_samples_leaf : SIZE_t
            The minimal number of samples each leaf can have, where splits
            which would result in having less samples in a leaf are not
            considered.

        min_weight_leaf : double
            The minimal weight each leaf can have, where the weight is the sum
            of the weights of each sample in it.

        random_state : object
            The user inputted random state to be used for pseudo-randomness
        """

        self.criterion = criterion

        self.samples = NULL
        self.n_samples = 0
        self.features = NULL
        self.n_features = 0
        self.feature_values = NULL

        self.y = NULL
        self.y_stride = 0
        self.sample_weight = NULL

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state
        self.presort = presort

    def __dealloc__(self):
        """Destructor."""

        #free(self.samples)
        free(self.features)
        free(self.constant_features)
        free(self.feature_values)

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int init(self,
                   object X, object invsig,
                   np.ndarray[DOUBLE_t, ndim=2, mode="c"] y, SIZE_t* samples, 
                   DOUBLE_t* sample_weight,
                   np.ndarray X_idx_sorted=None) except -1:
        """Initialize the splitter.

        Take in the input data X, the target Y, and optional sample weights.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        X : object
            This contains the inputs. Usually it is a 2d numpy array.

        y : numpy.ndarray, dtype=DOUBLE_t
            This is the vector of targets, or true labels, for the samples

        sample_weight : numpy.ndarray, dtype=DOUBLE_t (optional)
            The weights of the samples, where higher weighted samples are fit
            closer than lower weight samples. If not provided, all samples
            are assumed to have uniform weight.
        """
        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        cdef SIZE_t n_samples = X.shape[0]
        # Create a new array which will be used to store nonzero
        # samples from the feature of interest
        #cdef SIZE_t* samples = safe_realloc(&self.samples, n_samples)

        cdef SIZE_t i, j
        cdef double weighted_n_samples = 0.0
        j = 0

        for i in range(n_samples):
            # Only work with positively weighted samples
            if sample_weight == NULL or sample_weight[i] != 0.0:
                samples[j] = i
                j += 1

            if sample_weight != NULL:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0

        # Number of samples is number of positively weighted samples
        self.n_samples = j
        self.weighted_n_samples = weighted_n_samples

        cdef SIZE_t n_features = X.shape[1]
        cdef SIZE_t* features = safe_realloc(&self.features, n_features)

        for i in range(n_features):
            features[i] = i

        self.n_features = n_features
        self.n_invsigs = invsig.shape[1]
        safe_realloc(&self.feature_values, n_samples)
        safe_realloc(&self.constant_features, n_features)

        self.y = <DOUBLE_t*> y.data
        self.y_stride = <SIZE_t> y.strides[0] / <SIZE_t> y.itemsize
        
        self.sample_weight = sample_weight
        return 0

    cdef int node_reset(self, #SIZE_t start, SIZE_t end,
                        SIZE_t* samples, SIZE_t n_node_samples,
                        double* weighted_n_node_samples,
                        double* prob_weight) nogil except -1:
        """Reset splitter on node samples[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        start : SIZE_t
            The index of the first sample to consider
        end : SIZE_t
            The index of the last sample to consider
        weighted_n_node_samples : numpy.ndarray, dtype=double pointer
            The total weight of those samples
        """
        #self.start = start
        #self.end = end
        self.samples = samples
        self.n_node_samples = n_node_samples
        #self.prob_weight = prob_weight

        self.criterion.init(self.y,
                            self.y_stride,
                            self.sample_weight,
                            self.weighted_n_samples,
                            self.samples, self.n_node_samples, prob_weight)
                            #start,
                            #end, prob_weight)

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples
        return 0

    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features, SIZE_t** left_samples,
                        SIZE_t* left_n_node_samples, SIZE_t** right_samples,
                        SIZE_t* right_n_node_samples, double* left_prob_weight,
                        bint* is_prob, DTYPE_t* up_prob_bounds,
                        DTYPE_t* low_prob_bounds) nogil except -1:
        """Find the best split on node samples[start:end].

        This is a placeholder method. The majority of computation will be done
        here.

        It should return -1 upon errors.
        """

        pass

    cdef void node_value(self, double* dest) nogil:
        """Copy the value of node samples[start:end] into dest."""

        self.criterion.node_value(dest)

    cdef double node_impurity(self) nogil:
        """Return the impurity of the current node."""

        return self.criterion.node_impurity()


cdef class BaseDenseSplitter(Splitter):
    cdef DTYPE_t* X
    cdef SIZE_t X_sample_stride
    cdef SIZE_t X_feature_stride

    cdef DTYPE_t* invsig
    cdef SIZE_t invsig_sample_stride
    cdef SIZE_t invsig_feature_stride

    cdef np.ndarray X_idx_sorted
    cdef INT32_t* X_idx_sorted_ptr
    cdef SIZE_t X_idx_sorted_stride
    cdef SIZE_t n_total_samples
    cdef SIZE_t* sample_mask

    def __cinit__(self, Criterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  object random_state, bint presort):

        self.X = NULL
        self.X_sample_stride = 0
        self.X_feature_stride = 0
        self.X_idx_sorted_ptr = NULL
        self.X_idx_sorted_stride = 0
        self.invsig = NULL
        self.invsig_sample_stride = 0
        self.invsig_feature_stride = 0
        self.sample_mask = NULL
        self.presort = presort

    def __dealloc__(self):
        """Destructor."""
        if self.presort == 1:
            free(self.sample_mask)

    cdef int init(self,
                  object X, object invsig, 
                  np.ndarray[DOUBLE_t, ndim=2, mode="c"] y,
                  SIZE_t* samples,
                  DOUBLE_t* sample_weight,
                  np.ndarray X_idx_sorted=None) except -1:
        """Initialize the splitter

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """

        # Call parent init
        Splitter.init(self, X, invsig, y, samples, sample_weight)

        # Initialize X
        cdef np.ndarray X_ndarray = X

        self.X = <DTYPE_t*> X_ndarray.data
        self.X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
        self.X_feature_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize

        cdef np.ndarray invsig_ndarray = invsig

        self.invsig = <DTYPE_t*> invsig_ndarray.data
        self.invsig_sample_stride = <SIZE_t> invsig.strides[0] / <SIZE_t> invsig.itemsize
        self.invsig_feature_stride = <SIZE_t> invsig.strides[1] / <SIZE_t> invsig.itemsize
        
        if self.presort == 1:
            self.X_idx_sorted = X_idx_sorted
            self.X_idx_sorted_ptr = <INT32_t*> self.X_idx_sorted.data
            self.X_idx_sorted_stride = (<SIZE_t> self.X_idx_sorted.strides[1] /
                                        <SIZE_t> self.X_idx_sorted.itemsize)

            self.n_total_samples = X.shape[0]
            safe_realloc(&self.sample_mask, self.n_total_samples)
            memset(self.sample_mask, 0, self.n_total_samples*sizeof(SIZE_t))

        return 0


cdef class BestSplitter(BaseDenseSplitter):
    """Splitter for finding the best split."""
    def __reduce__(self):
        return (BestSplitter, (self.criterion,
                               self.max_features,
                               self.min_samples_leaf,
                               self.min_weight_leaf,
                               self.random_state,
                               self.presort), self.__getstate__())

    #cdef int node_split(self, double impurity, SplitRecord* split,
    #                    SIZE_t* n_constant_features) nogil except -1:
    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features, SIZE_t** left_samples,
                        SIZE_t* left_n_node_samples, SIZE_t** right_samples,
                        SIZE_t* right_n_node_samples, double* left_prob_weight,
                        bint* is_prob, DTYPE_t* up_prob_bounds,
                        DTYPE_t* low_prob_bounds) nogil except -1:
        """Find the best split on node samples[start:end]

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # Find the best split
        cdef SIZE_t* samples = self.samples
        #cdef SIZE_t start = self.start
        #cdef SIZE_t end = self.end
        cdef SIZE_t n_node_samples = self.n_node_samples

        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t* X = self.X
        cdef DTYPE_t* invsig = self.invsig
        cdef DTYPE_t* Xf = self.feature_values

        cdef SIZE_t X_sample_stride = self.X_sample_stride
        cdef SIZE_t X_feature_stride = self.X_feature_stride
        cdef SIZE_t invsig_sample_stride = self.invsig_sample_stride
        cdef SIZE_t invsig_feature_stride = self.invsig_feature_stride
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef INT32_t* X_idx_sorted = self.X_idx_sorted_ptr
        cdef SIZE_t* sample_mask = self.sample_mask

        cdef SplitRecord best, current
        cdef double current_proxy_improvement = -INFINITY
        cdef double best_proxy_improvement = -INFINITY
        cdef double left_proxy_improvement = -INFINITY
        cdef double right_proxy_improvement = -INFINITY
        cdef double current_d_reg = 0

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j
        cdef SIZE_t tmp
        cdef SIZE_t p
        cdef SIZE_t feature_idx_offset
        cdef SIZE_t feature_offset
        cdef SIZE_t invsig_offset
        cdef SIZE_t i
        cdef SIZE_t j

        cdef SIZE_t n_visited_features = 0
        # Number of features discovered to be constant during the split search
        cdef SIZE_t n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        # n_total_constants = n_known_constants + n_found_constants
        cdef SIZE_t n_total_constants = n_known_constants
        cdef DTYPE_t current_feature_value
        cdef SIZE_t partition_end
        cdef bint _is_prob = 0
        cdef DTYPE_t _invsig = 0
        cdef SIZE_t _left_n_node_samples = 0
        cdef SIZE_t _right_n_node_samples = 0
        cdef SIZE_t* _left_samples = NULL
        cdef SIZE_t* _right_samples = NULL
        cdef DTYPE_t _X = 0
        cdef DTYPE_t _thres
        cdef DTYPE_t _left_thres
        cdef DTYPE_t _right_thres
        cdef double left_weight
        cdef double right_weight
        #_init_split(&best, ends)
        _init_split(&best, n_node_samples)
        if self.presort == 1:
            #for p in range(start, end):
            for p in range(n_node_samples):
                sample_mask[samples[p]] = 1

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.
        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                (n_visited_features < max_features or
                 # At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)):

            n_visited_features += 1

            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.

            # Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)

            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants[
                tmp = features[f_j]
                features[f_j] = features[n_drawn_constants]
                features[n_drawn_constants] = tmp

                n_drawn_constants += 1

            else:
                # f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants
                # f_j in the interval [n_total_constants, f_i[
                current.feature = features[f_j]
                feature_offset = self.X_feature_stride * current.feature
                if features[f_j] < self.n_invsigs:
                    _is_prob = 1
                    invsig_offset = self.invsig_feature_stride * features[f_j]
                else:
                    _is_prob = 0
                # Sort samples along that feature; either by utilizing
                # presorting, or by copying the values into an array and
                # sorting the array in a manner which utilizes the cache more
                # effectively.
                if self.presort == 1:
                    #p = start
                    p = 0
                    feature_idx_offset = self.X_idx_sorted_stride * current.feature

                    for i in range(self.n_total_samples): 
                        j = X_idx_sorted[i + feature_idx_offset]
                        if sample_mask[j] == 1:
                            samples[p] = j
                            Xf[p] = X[self.X_sample_stride * j + feature_offset]
                            p += 1
                else:
                    #for i in range(start, end):
                    for i in range(n_node_samples):
                        Xf[i] = X[self.X_sample_stride * samples[i] + feature_offset]
                    #sort(Xf + start, samples + start, end - start)
                    sort(Xf, samples, n_node_samples)

                #if Xf[end - 1] <= Xf[start] + FEATURE_THRESHOLD:
                if Xf[n_node_samples - 1] <= Xf[0] + FEATURE_THRESHOLD:
                    features[f_j] = features[n_total_constants]
                    features[n_total_constants] = current.feature

                    n_found_constants += 1
                    n_total_constants += 1

                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]
                
                    # Evaluate all splits
                    self.criterion.reset()
                    #p = start
                    p = 0
                    #while p < end:
                    while p < n_node_samples:
                        #while (p + 1 < end and
                        #       Xf[p + 1] <= Xf[p] + FEATURE_THRESHOLD):
                        while (p + 1 < n_node_samples and Xf[p + 1] <= Xf[p] + FEATURE_THRESHOLD):
                            p += 1

                        # (p + 1 >= end) or (X[samples[p + 1], current.feature] >
                        #                    X[samples[p], current.feature])
                        p += 1
                        # (p >= end) or (X[samples[p], current.feature] >
                        #                X[samples[p - 1], current.feature])

                        #if p < end:
                        if p < n_node_samples:
                            current.pos = p

                            # Reject if min_samples_leaf is not guaranteed
                            #if (((current.pos - start) < min_samples_leaf) or
                            #        ((end - current.pos) < min_samples_leaf)):
                            if ((current.pos < min_samples_leaf) or 
                                 ((n_node_samples - current.pos) < min_samples_leaf)):
                                continue

                            if _is_prob:
                                _thres = max(Xf[p], low_prob_bounds[current.feature])
                                _thres = min(_thres, up_prob_bounds[current.feature])
                                if _thres != Xf[p]:
                                    continue
                                if p < 1:
                                    continue
                                if p > n_node_samples - 2:
                                    continue
                            #    # left:
                            #    memset(left_prob_weight, 0, self.n_samples * sizeof(double))
                            #    _left_thres = max(Xf[p - 1], low_prob_bounds[current.feature])
                            #    _left_thres = min(_left_thres, up_prob_bounds[current.feature])
                            #    for q in range(n_node_samples):
                            #        _invsig = invsig[samples[q] * self.invsig_sample_stride + invsig_offset]
                            #        #left_prob_weight[samples[q]] = 0.5 * erf((Xf[p] - Xf[q]) * SQRT05 * _invsig) + 0.5
                            #        left_weight = 0.5 * erf((_left_thres - Xf[q]) * SQRT05 * _invsig) - 0.5 * erf((low_prob_bounds[current.feature] - Xf[q]) * SQRT05 * _invsig)
                            #        right_weight = 0.5 * erf((up_prob_bounds[current.feature] - Xf[q]) * SQRT05 * _invsig) - 0.5 * erf((_left_thres - Xf[q]) * SQRT05 * _invsig)
                            #        left_prob_weight[samples[q]] = left_weight / (left_weight + right_weight)
                            #    self.criterion.prob_update(current.pos, left_prob_weight)
                            #    left_proxy_improvement = self.criterion.prob_proxy_impurity_improvement(left_prob_weight)
                            #    # right:
                            #    memset(left_prob_weight, 0, self.n_samples * sizeof(double))
                            #    _right_thres = max(Xf[p + 1], low_prob_bounds[current.feature])
                            #    _right_thres = min(_left_thres, up_prob_bounds[current.feature])
                            #    for q in range(n_node_samples):
                            #        _invsig = invsig[samples[q] * self.invsig_sample_stride + invsig_offset]
                            #        #left_prob_weight[samples[q]] = 0.5 * erf((Xf[p] - Xf[q]) * SQRT05 * _invsig) + 0.5
                            #        left_weight = 0.5 * erf((_right_thres - Xf[q]) * SQRT05 * _invsig) - 0.5 * erf((low_prob_bounds[current.feature] - Xf[q]) * SQRT05 * _invsig)
                            #        right_weight = 0.5 * erf((up_prob_bounds[current.feature] - Xf[q]) * SQRT05 * _invsig) - 0.5 * erf((_right_thres - Xf[q]) * SQRT05 * _invsig)
                            #        left_prob_weight[samples[q]] = left_weight / (left_weight + right_weight)
                            #    self.criterion.prob_update(current.pos, left_prob_weight)
                            #    right_proxy_improvement = self.criterion.prob_proxy_impurity_improvement(left_prob_weight)
                                # current:
                                memset(left_prob_weight, 0, self.n_samples * sizeof(double))
                                for q in range(n_node_samples):
                                    _invsig = invsig[samples[q] * self.invsig_sample_stride + invsig_offset]
                                    #left_prob_weight[samples[q]] = 0.5 * erf((Xf[p] - Xf[q]) * SQRT05 * _invsig) + 0.5
                                    left_weight = 0.5 * erf((_thres - Xf[q]) * SQRT05 * _invsig) - 0.5 * erf((low_prob_bounds[current.feature] - Xf[q]) * SQRT05 * _invsig)
                                    right_weight = 0.5 * erf((up_prob_bounds[current.feature] - Xf[q]) * SQRT05 * _invsig) - 0.5 * erf((_thres - Xf[q]) * SQRT05 * _invsig)
                                    left_prob_weight[samples[q]] = left_weight / (left_weight + right_weight)
                                self.criterion.prob_update(current.pos, left_prob_weight)
                                current_proxy_improvement = self.criterion.prob_proxy_impurity_improvement(left_prob_weight)
                                #if alpha > 0:
                                #    cpy_inplace = 0
                                #    current_d_reg = self.criterion.d_improvement(_is_prob, current.pos, alpha,
                                #                                                self.gamma, left_prob_weight, y_bar, 
                                #                                                d, cpy_inplace)
                                #    current_proxy_improvement -= beta * current_d_reg
                            else:
                                self.criterion.update(current.pos)
                                current_proxy_improvement = self.criterion.proxy_impurity_improvement()
                                #if alpha > 0:
                                #    if is_past_prob:
                                #        cpy_inplace = 0
                                #        current_d_reg = self.criterion.d_improvement(_is_prob, current.pos, alpha,
                                #                                                    self.gamma, left_prob_weight, y_bar,
                                #                                                    d, cpy_inplace = 0)
                                #        current_proxy_improvement -= beta * current_d_reg
                            # Reject if min_weight_leaf is not satisfied
                            if ((self.criterion.weighted_n_left < min_weight_leaf) or
                                    (self.criterion.weighted_n_right < min_weight_leaf)):
                                continue

                            if _is_prob:
                                if current_proxy_improvement > best_proxy_improvement:
                                    best_proxy_improvement = current_proxy_improvement
                                    # sum of halves is used to avoid infinite value
                                    #current.threshold = Xf[p - 1] / 2.0 + Xf[p] / 2.0
                                    current.threshold = Xf[p]
                                    #current.threshold = ((Xf[p] ** 2 - Xf[p + 1] ** 2) * left_proxy_improvement + 
                                    #                     (Xf[p + 1] ** 2 - Xf[p - 1] ** 2) * current_proxy_improvement + 
                                    #                     (Xf[p - 1] ** 2 - Xf[p] ** 2) * right_proxy_improvement)
                                    #current.threshold /= ((Xf[p] - Xf[p + 1]) * left_proxy_improvement + 
                                    #                      (Xf[p + 1] - Xf[p - 1]) * current_proxy_improvement + 
                                    #                      (Xf[p - 1] - Xf[p]) * right_proxy_improvement)

                                    #if ((current.threshold == Xf[p]) or
                                    #    (current.threshold == INFINITY) or
                                    #    (current.threshold == -INFINITY)):
                                    #    current.threshold = Xf[p - 1]
                                    if ((current.threshold == INFINITY) or (current.threshold == - INFINITY)):
                                        current.threshold = Xf[p - 1]

                                    best = current  # copy
                            else:
                                if current_proxy_improvement > best_proxy_improvement:
                                    best_proxy_improvement = current_proxy_improvement
                                    # sum of halves is used to avoid infinite value
                                    current.threshold = Xf[p - 1] / 2.0 + Xf[p] / 2.0

                                    if ((current.threshold == Xf[p]) or
                                        (current.threshold == INFINITY) or
                                        (current.threshold == -INFINITY)):
                                        current.threshold = Xf[p - 1]

                                    best = current  # copy

        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        #if best.pos < end:
        if best.pos < n_node_samples:
            feature_offset = X_feature_stride * best.feature
            #partition_end = end
            partition_end = n_node_samples
            #p = start
            p = 0
            if best.feature < self.n_invsigs:
                _is_prob = 1
                invsig_offset = self.invsig_feature_stride * best.feature
            else:
                _is_prob = 0

            while p < partition_end:
                if X[X_sample_stride * samples[p] + feature_offset] <= best.threshold:
                    p += 1

                else:
                    partition_end -= 1

                    tmp = samples[partition_end]
                    samples[partition_end] = samples[p]
                    samples[p] = tmp

            self.criterion.reset()
            _left_n_node_samples = 0
            _right_n_node_samples = 0
            
            if _is_prob:
                _left_samples = <SIZE_t*> malloc(n_node_samples * sizeof(SIZE_t))
                _right_samples = <SIZE_t*> malloc(n_node_samples * sizeof(SIZE_t))
                memset(left_prob_weight, 0, self.n_samples * sizeof(double))
                for p in range(n_node_samples):
                    _invsig = invsig[invsig_sample_stride * samples[p] + invsig_offset]
                    _X = X[X_sample_stride * samples[p] + feature_offset]
                    #left_prob_weight[samples[p]] = 0.5 * erf((best.threshold - _X) * SQRT05 * _invsig) + 0.5
                    left_weight = 0.5 * erf((best.threshold - _X) * SQRT05 * _invsig) \
                                  - 0.5 * erf((low_prob_bounds[best.feature] - _X) * SQRT05 * _invsig)
                    right_weight = 0.5 * erf((up_prob_bounds[best.feature] - _X) * SQRT05 * _invsig) \
                                   - 0.5 * erf((best.threshold - _X) * SQRT05 * _invsig)
                    left_prob_weight[samples[p]] = left_weight / (left_weight + right_weight)
                    #if _X - 3 * _invsig < best.threshold:
                    if left_prob_weight[samples[p]] > 0.0001:
                        _left_samples[_left_n_node_samples] = samples[p]
                        _left_n_node_samples += 1
                    else:
                        left_prob_weight[samples[p]] = 0
                    #if _X + 3 * _invsig > best.threshold:
                    if left_prob_weight[samples[p]] < 0.9999:
                        _right_samples[_right_n_node_samples] = samples[p]
                        _right_n_node_samples += 1
                    else:
                        left_prob_weight[samples[p]] = 1
                    
                self.criterion.prob_update(best.pos, left_prob_weight)
                best.improvement = self.criterion.prob_impurity_improvement(impurity, left_prob_weight)
                self.criterion.prob_children_impurity(&best.impurity_left,
                                                      &best.impurity_right, left_prob_weight)
                left_samples[0] = <SIZE_t*> malloc(_left_n_node_samples * sizeof(SIZE_t))
                memcpy(left_samples[0], _left_samples, _left_n_node_samples * sizeof(SIZE_t))
                free(_left_samples)
                right_samples[0] = <SIZE_t*> malloc(_right_n_node_samples * sizeof(SIZE_t))
                memcpy(right_samples[0], _right_samples, _right_n_node_samples * sizeof(SIZE_t))
                free(_right_samples)
                #if alpha > 0:
                    #new_y_bar = <double*> malloc(self.n_samples * sizeof(double))
                    #new_d = <double*> malloc(self.n_samples * sizeof(double))
                    #memcpy(new_y_bar, y_bar, self.n_samples * sizeof(double))
                    #memcpy(new_d, d, self.n_samples * sizeof(double))
                #    cpy_inplace = 1
                #    current_d_reg = self.criterion.d_improvement(_is_prob, best.pos, alpha, self.gamma,
                #                                                left_prob_weight, y_bar, 
                #                                                d, cpy_inplace)
                    #memcpy(y_bar, new_y_bar, self.n_samples * sizeof(double))
                    #memcpy(d, new_d, self.n_samples * sizeof(double))
                    #free(new_y_bar)
                    #free(new_d)
                
            else:
                self.criterion.update(best.pos)
                _left_n_node_samples = best.pos
                _right_n_node_samples = n_node_samples - best.pos
                left_samples[0] = <SIZE_t*> malloc(_left_n_node_samples * sizeof(SIZE_t))
                memcpy(left_samples[0], samples, _left_n_node_samples * sizeof(SIZE_t))
                right_samples[0] = <SIZE_t*> malloc(_right_n_node_samples * sizeof(SIZE_t))
                memcpy(right_samples[0], samples + best.pos, _right_n_node_samples * sizeof(SIZE_t))
                best.improvement = self.criterion.impurity_improvement(impurity)
                self.criterion.children_impurity(&best.impurity_left,
                                                 &best.impurity_right)
                #if alpha > 0:
                #    if is_past_prob:
                        #new_y_bar = <double*> malloc(self.n_samples * sizeof(double))
                        #new_d = <double*> malloc(self.n_samples * sizeof(double))
                #        cpy_inplace = 1
                #        current_d_reg = self.criterion.d_improvement(_is_prob, best.pos, alpha, self.gamma,
                #                                                    left_prob_weight, y_bar, 
                #                                                    d, cpy_inplace)
                        #memcpy(y_bar, new_y_bar, self.n_samples * sizeof(double))
                        #memcpy(d, new_d, self.n_samples * sizeof(double))
                        #free(new_y_bar)
                        #free(new_d)
                #    else:
                #        self.criterion.d_init(best.pos, y_bar, d)
                    
            left_n_node_samples[0] = _left_n_node_samples
            right_n_node_samples[0] = _right_n_node_samples
            is_prob[0] = _is_prob
            

        # Reset sample mask
        if self.presort == 1:
            #for p in range(start, end):
            for p in range(n_node_samples):
                sample_mask[samples[p]] = 0

        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)

        # Copy newly found constant features
        memcpy(constant_features + n_known_constants,
               features + n_known_constants,
               sizeof(SIZE_t) * n_found_constants)

        # Return values
        split[0] = best
        n_constant_features[0] = n_total_constants
        return 0


# Sort n-element arrays pointed to by Xf and samples, simultaneously,
# by the values in Xf. Algorithm: Introsort (Musser, SP&E, 1997).
cdef inline void sort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    if n == 0:
      return
    cdef int maxd = 2 * <int>log(n)
    introsort(Xf, samples, n, maxd)


cdef inline void swap(DTYPE_t* Xf, SIZE_t* samples,
        SIZE_t i, SIZE_t j) nogil:
    # Helper for sort
    Xf[i], Xf[j] = Xf[j], Xf[i]
    samples[i], samples[j] = samples[j], samples[i]


cdef inline DTYPE_t median3(DTYPE_t* Xf, SIZE_t n) nogil:
    # Median of three pivot selection, after Bentley and McIlroy (1993).
    # Engineering a sort function. SP&E. Requires 8/3 comparisons on average.
    cdef DTYPE_t a = Xf[0], b = Xf[n / 2], c = Xf[n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b


# Introsort with median of 3 pivot selection and 3-way partition function
# (robust to repeated elements, e.g. lots of zero features).
cdef void introsort(DTYPE_t* Xf, SIZE_t *samples,
                    SIZE_t n, int maxd) nogil:
    cdef DTYPE_t pivot
    cdef SIZE_t i, l, r

    while n > 1:
        if maxd <= 0:   # max depth limit exceeded ("gone quadratic")
            heapsort(Xf, samples, n)
            return
        maxd -= 1

        pivot = median3(Xf, n)

        # Three-way partition.
        i = l = 0
        r = n
        while i < r:
            if Xf[i] < pivot:
                swap(Xf, samples, i, l)
                i += 1
                l += 1
            elif Xf[i] > pivot:
                r -= 1
                swap(Xf, samples, i, r)
            else:
                i += 1

        introsort(Xf, samples, l, maxd)
        Xf += r
        samples += r
        n -= r


cdef inline void sift_down(DTYPE_t* Xf, SIZE_t* samples,
                           SIZE_t start, SIZE_t end) nogil:
    # Restore heap order in Xf[start:end] by moving the max element to start.
    cdef SIZE_t child, maxind, root

    root = start
    while True:
        child = root * 2 + 1

        # find max of root, left child, right child
        maxind = root
        if child < end and Xf[maxind] < Xf[child]:
            maxind = child
        if child + 1 < end and Xf[maxind] < Xf[child + 1]:
            maxind = child + 1

        if maxind == root:
            break
        else:
            swap(Xf, samples, root, maxind)
            root = maxind


cdef void heapsort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    cdef SIZE_t start, end

    # heapify
    start = (n - 2) / 2
    end = n
    while True:
        sift_down(Xf, samples, start, end)
        if start == 0:
            break
        start -= 1

    # sort by shrinking the heap, putting the max element immediately after it
    end = n - 1
    while end > 0:
        swap(Xf, samples, 0, end)
        sift_down(Xf, samples, 0, end)
        end = end - 1

