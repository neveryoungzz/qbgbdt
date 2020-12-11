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
#          Zhe Zheng <zzheng16@163.com>
#
# License: BSD 3 clause

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset
from libc.math cimport fabs, pow

import numpy as np
cimport numpy as np
np.import_array()

from ._utils cimport log
from ._utils cimport safe_realloc
from ._utils cimport sizet_ptr_to_ndarray
from ._utils cimport WeightedMedianCalculator


cdef class Criterion:
    """Interface for impurity criteria.

    This object stores methods on how to calculate how good a split is using
    different metrics.
    """

    def __dealloc__(self):
        """Destructor."""

        free(self.sum_total)
        free(self.sum_left)
        free(self.sum_right)

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    #cdef int init(self, DOUBLE_t* y, SIZE_t y_stride, DOUBLE_t* sample_weight,
    #              double weighted_n_samples, SIZE_t* samples, SIZE_t start,
    #              SIZE_t end, double* prob_weight) nogil except -1:
    cdef int init(self, DOUBLE_t* y, SIZE_t y_stride, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, 
                  SIZE_t n_node_samples, double* prob_weight) nogil except -1:
    
        """Placeholder for a method which will initialize the criterion.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        y : array-like, dtype=DOUBLE_t
            y is a buffer that can store values for n_outputs target variables
        y_stride : SIZE_t
            y_stride is used to index the kth output value as follows:
            y[i, k] = y[i * y_stride + k]
        sample_weight : array-like, dtype=DOUBLE_t
            The weight of each sample
        weighted_n_samples : DOUBLE_t
            The total weight of the samples being considered
        samples : array-like, dtype=DOUBLE_t
            Indices of the samples in X and y, where samples[start:end]
            correspond to the samples in this node
        start : SIZE_t
            The first sample to be used on this node
        end : SIZE_t
            The last sample used on this node

        """

        pass

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start.

        This method must be implemented by the subclass.
        """

        pass

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end.

        This method must be implemented by the subclass.
        """
        pass

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left child.

        This updates the collected statistics by moving samples[pos:new_pos]
        from the right child to the left child. It must be implemented by
        the subclass.

        Parameters
        ----------
        new_pos : SIZE_t
            New starting index position of the samples in the right child
        """

        pass

    cdef int prob_update(self, SIZE_t new_pos, double* left_prob_weight) nogil except -1:
        pass

    cdef double node_impurity(self) nogil:
        """Placeholder for calculating the impurity of the node.

        Placeholder for a method which will evaluate the impurity of
        the current node, i.e. the impurity of samples[start:end]. This is the
        primary function of the criterion class.
        """

        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Placeholder for calculating the impurity of children.

        Placeholder for a method which evaluates the impurity in
        children nodes, i.e. the impurity of samples[start:pos] + the impurity
        of samples[pos:end].

        Parameters
        ----------
        impurity_left : double pointer
            The memory address where the impurity of the left child should be
            stored.
        impurity_right : double pointer
            The memory address where the impurity of the right child should be
            stored
        """

        pass
    
    cdef void prob_children_impurity(self, double* impurity_left, 
                                     double* impurity_right,
                                     double* left_prob_weight) nogil:
        pass

    cdef void node_value(self, double* dest) nogil:
        """Placeholder for storing the node value.

        Placeholder for a method which will compute the node value
        of samples[start:end] and save the value into dest.

        Parameters
        ----------
        dest : double pointer
            The memory address where the node value should be stored.
        """

        pass

    #cdef int d_init(self, SIZE_t pos, double* y_bar, double* d) nogil:
    #    pass

    #cdef double d_improvement(self, bint is_prob, SIZE_t pos, double alpha,
    #                                     double* gamma,
    #                                     double* left_prob_weight,
    #                                     double* y_bar, 
    #                                     double* d, bint cpy_inplace) nogil:
    #    pass

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """
        cdef double impurity_left
        cdef double impurity_right
        self.children_impurity(&impurity_left, &impurity_right)

        return (- self.weighted_n_right * impurity_right
                - self.weighted_n_left * impurity_left)

    cdef double prob_proxy_impurity_improvement(self, double* left_prob_weight) nogil:
        cdef double impurity_left
        cdef double impurity_right
        self.prob_children_impurity(&impurity_left, &impurity_right, left_prob_weight)

        return (- self.weighted_n_right * impurity_right
                - self.weighted_n_left * impurity_left)

    cdef double impurity_improvement(self, double impurity) nogil:
        """Compute the improvement in impurity

        This method computes the improvement in impurity when a split occurs.
        The weighted impurity improvement equation is the following:

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where N is the total number of samples, N_t is the number of samples
        at the current node, N_t_L is the number of samples in the left child,
        and N_t_R is the number of samples in the right child,

        Parameters
        ----------
        impurity : double
            The initial impurity of the node before the split

        Return
        ------
        double : improvement in impurity after the split occurs
        """

        cdef double impurity_left
        cdef double impurity_right

        self.children_impurity(&impurity_left, &impurity_right)

        if self.weighted_n_node_samples > 0:
            return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                    (impurity - (self.weighted_n_right / 
                                self.weighted_n_node_samples * impurity_right)
                            - (self.weighted_n_left / 
                                self.weighted_n_node_samples * impurity_left)))
        else:
            return 0.0

    cdef double prob_impurity_improvement(self, double impurity, 
                                          double* left_prob_weight) nogil:
        cdef double impurity_left
        cdef double impurity_right

        self.prob_children_impurity(&impurity_left, &impurity_right, left_prob_weight)

        if self.weighted_n_node_samples > 0:
            return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                    (impurity - (self.weighted_n_right / 
                                self.weighted_n_node_samples * impurity_right)
                            - (self.weighted_n_left / 
                                self.weighted_n_node_samples * impurity_left)))
        else:
            return 0.0

cdef class RegressionCriterion(Criterion):
    r"""Abstract regression criterion.

    This handles cases where the target is a continuous value, and is
    evaluated by computing the variance of the target values left and right
    of the split point. The computation takes linear time with `n_samples`
    by using ::

        var = \sum_i^n (y_i - y_bar) ** 2
            = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2
    """

    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_samples):
        """Initialize parameters for this criterion.

        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets to be predicted

        n_samples : SIZE_t
            The total number of samples to fit on
        """

        # Default values
        self.y = NULL
        self.y_stride = 0
        self.sample_weight = NULL

        self.samples = NULL
        #self.start = 0
        self.pos = 0
        #self.end = 0
        self.prob_weight = NULL
        
        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.sq_sum_total = 0.0

        # Allocate accumulators. Make sure they are NULL, not uninitialized,
        # before an exception can be raised (which triggers __dealloc__).
        self.sum_total = NULL
        self.sum_left = NULL
        self.sum_right = NULL

        # Allocate memory for the accumulators
        self.sum_total = <double*> calloc(n_outputs, sizeof(double))
        self.sum_left = <double*> calloc(n_outputs, sizeof(double))
        self.sum_right = <double*> calloc(n_outputs, sizeof(double))

        if (self.sum_total == NULL or 
                self.sum_left == NULL or
                self.sum_right == NULL):
            raise MemoryError()

    def __reduce__(self):
        return (type(self), (self.n_outputs, self.n_samples), self.__getstate__())

    #cdef int init(self, DOUBLE_t* y, SIZE_t y_stride, DOUBLE_t* sample_weight,
    #              double weighted_n_samples, SIZE_t* samples, SIZE_t start,
    #              SIZE_t end, DOUBLE_t* prob_weight) nogil except -1:
    cdef int init(self, DOUBLE_t* y, SIZE_t y_stride, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t n_node_samples,
                  double* prob_weight) nogil except -1:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""
        # Initialize fields
        self.y = y
        self.y_stride = y_stride
        self.sample_weight = sample_weight
        self.samples = samples
        #self.start = start
        #self.end = end
        self.n_node_samples = n_node_samples
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.
        self.prob_weight = prob_weight
        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t y_ik
        cdef DOUBLE_t w_y_ik
        cdef DOUBLE_t w = 1.0

        self.sq_sum_total = 0.0
        memset(self.sum_total, 0, self.n_outputs * sizeof(double))

        #for p in range(start, end):
        for p in range(n_node_samples):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i] * self.prob_weight[i]
            else:
                w = self.prob_weight[i]

            for k in range(self.n_outputs):
                y_ik = y[i * y_stride + k]
                w_y_ik = w * y_ik
                self.sum_total[k] += w_y_ik
                self.sq_sum_total += w_y_ik * y_ik

            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start."""
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
        memset(self.sum_left, 0, n_bytes)
        memcpy(self.sum_right, self.sum_total, n_bytes)

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        #self.pos = self.start
        self.pos = 0
        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end."""
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
        memset(self.sum_right, 0, n_bytes)
        memcpy(self.sum_left, self.sum_total, n_bytes)

        self.weighted_n_right = 0.0
        self.weighted_n_left = self.weighted_n_node_samples
        #self.pos = self.end
        self.pos = self.n_node_samples
        return 0

    cdef int prob_update(self, SIZE_t new_pos, double* left_prob_weight) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left."""
        
        ## reset node each time
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total
        cdef double* prob_weight = self.prob_weight
        
        cdef double* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples

        cdef DOUBLE_t* y = self.y
        cdef SIZE_t pos = self.pos
        #cdef SIZE_t end = self.end
        cdef SIZE_t n_node_samples = self.n_node_samples
        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0
        cdef DOUBLE_t y_ik
        
        # Update statistics up to new_pos
        #
        # Given that
        #           sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_pos.

        #if (new_pos - pos) <= (end - new_pos):
        self.reset()
        for p in range(n_node_samples):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i] * left_prob_weight[i] * prob_weight[i]
            else:
                w = left_prob_weight[i] * prob_weight[i]

            for k in range(self.n_outputs):
                y_ik = y[i * self.y_stride + k]
                sum_left[k] += w * y_ik

            self.weighted_n_left += w
        #else:
        #    self.reverse_reset()

        #    for p in range(end - 1, new_pos - 1, -1):
        #        i = samples[p]

        #        if sample_weight != NULL:
        #            w = sample_weight[i] * self.prob_weight[i] * 

        #        for k in range(self.n_outputs):
        #            y_ik = y[i * self.y_stride + k]
        #            sum_left[k] -= w * y_ik

        #        self.weighted_n_left -= w
        
        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        for k in range(self.n_outputs):
            sum_right[k] = sum_total[k] - sum_left[k]

        self.pos = new_pos
        return 0

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left."""

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total
        cdef double* prob_weight = self.prob_weight

        cdef double* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples

        cdef DOUBLE_t* y = self.y
        #cdef SIZE_t pos = self.pos
        #cdef SIZE_t end = self.end
        cdef SIZE_t pos = self.pos
        cdef SIZE_t n_node_samples = self.n_node_samples
        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0
        cdef DOUBLE_t y_ik

        # Update statistics up to new_pos
        #
        # Given that
        #           sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_pos.

        #if (new_pos - pos) <= (end - new_pos):
        if (new_pos - pos) <= (n_node_samples - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i] * prob_weight[i]
                else:
                    w = prob_weight[i]

                for k in range(self.n_outputs):
                    y_ik = y[i * self.y_stride + k]
                    sum_left[k] += w * y_ik

                self.weighted_n_left += w
        else:
            self.reverse_reset()

            #for p in range(end - 1, new_pos - 1, -1):
            for p in range(n_node_samples - 1, new_pos - 1, -1):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i] * prob_weight[i]
                else:
                    w = prob_weight[i]

                for k in range(self.n_outputs):
                    y_ik = y[i * self.y_stride + k]
                    sum_left[k] -= w * y_ik

                self.weighted_n_left -= w

        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        for k in range(self.n_outputs):
            sum_right[k] = sum_total[k] - sum_left[k]

        self.pos = new_pos
        return 0
    
    #cdef int d_init(self, SIZE_t pos, double* y_bar, double* d) nogil:
    #    pass

    #cdef double d_improvement(self, bint is_prob, SIZE_t pos, double alpha,
    #                                     double* gamma,
    #                                     double* left_prob_weight,
    #                                     double* y_bar, 
    #                                     double* d, bint cpy_inplace) nogil:
    #    pass

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        pass
    
    cdef void prob_children_impurity(self, double* impurity_left,
                                     double* impurity_right,
                                     double* left_prob_weight) nogil:
        pass

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""

        cdef SIZE_t k

        for k in range(self.n_outputs):
            if self.weighted_n_node_samples > 0:
                dest[k] = self.sum_total[k] / self.weighted_n_node_samples
            else:
                dest[k] = 0

cdef class MSE(RegressionCriterion):
    """Mean squared error impurity criterion.

        MSE = var_left + var_right
    """

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""

        cdef double* sum_total = self.sum_total
        cdef double impurity
        cdef SIZE_t k
        if self.weighted_n_node_samples > 0:
            impurity = self.sq_sum_total / self.weighted_n_node_samples
            for k in range(self.n_outputs):
                impurity -= (sum_total[k] / self.weighted_n_node_samples)**2.0

            return impurity / self.n_outputs
        else:
            impurity = 0
            return impurity

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef SIZE_t k
        cdef double proxy_impurity_left = 0.0
        cdef double proxy_impurity_right = 0.0
        cdef double out = 0.0
        for k in range(self.n_outputs):
            proxy_impurity_left += sum_left[k] * sum_left[k]
            proxy_impurity_right += sum_right[k] * sum_right[k]

        if self.weighted_n_left > 0:
            out += proxy_impurity_left / self.weighted_n_left
        if self.weighted_n_right > 0:
            out += proxy_impurity_right / self.weighted_n_right
        #return (proxy_impurity_left / self.weighted_n_left +
        #        proxy_impurity_right / self.weighted_n_right)
        return out

    cdef double prob_proxy_impurity_improvement(self, double* left_prob_weight) nogil:
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef SIZE_t k
        cdef double proxy_impurity_left = 0.0
        cdef double proxy_impurity_right = 0.0
        cdef double out = 0.0
        for k in range(self.n_outputs):
            proxy_impurity_left += sum_left[k] * sum_left[k]
            proxy_impurity_right += sum_right[k] * sum_right[k]

        if self.weighted_n_left > 0:
            out += proxy_impurity_left / self.weighted_n_left
        if self.weighted_n_right > 0:
            out += proxy_impurity_right / self.weighted_n_right
        #return (proxy_impurity_left / self.weighted_n_left +
        #        proxy_impurity_right / self.weighted_n_right)
        return out
    
    #cdef int d_init(self, SIZE_t pos, double* y_bar, double* d) nogil:
    #    cdef SIZE_t p = 0
    #    cdef SIZE_t i = 0
    #    cdef SIZE_t k = 0
    #    cdef SIZE_t* samples = self.samples
    #    cdef double* left_value = <double*> calloc(self.n_outputs, sizeof(double))
    #    cdef double* right_value = <double*> calloc(self.n_outputs, sizeof(double))
    #    memset(left_value, 0, self.n_outputs * sizeof(double))
    #    memset(right_value, 0, self.n_outputs * sizeof(double))
    #    for k in range(self.n_outputs):
    #        if self.weighted_n_left > 0:
    #            left_value[k] = self.sum_left[k]/self.weighted_n_left
    #        else:
    #            left_value[k] = 0
    #        if self.weighted_n_right > 0:
    #            right_value[k] = self.sum_right[k]/self.weighted_n_right
    #        else:
    #            right_value[k] = 0
    #    for p in range(pos):
    #        i = samples[p]
    #        y_bar[i] = 0
    #        for k in range(self.n_outputs):
    #            y_bar[i] += left_value[k]
    #        d[i] = 0
    #    for p in range(pos, self.n_node_samples):
    #        i = samples[p]
    #        y_bar[i] = 0
    #        for k in range(self.n_outputs):
    #            y_bar[i] += right_value[k]
    #        d[i] = 0
    #    free(left_value)
    #    free(right_value)
    #    return 0
    
    #cdef double d_improvement(self, bint is_prob, SIZE_t pos, double alpha,
    #                                     double* gamma,
    #                                     double* left_prob_weight,
    #                                     double* y_bar,
    #                                     double* d, bint cpy_inplace) nogil:
        # d = sum_i(gamma[i] * pow(sum_b(w_b * d_ib), 1/alpha))
        # sum_b(w_b * d_ib) = sum_b(w_b * y ** 2) - sum_b(w_b * y) ** 2
        #                   = old_d + w_b * delta_y ** 2 - delta sum_b y_bar ** 2
    #    cdef SIZE_t k = 0
    #    cdef SIZE_t p = 0
    #    cdef SIZE_t* samples = self.samples
    #    cdef double* left_value = <double*> calloc(self.n_outputs, sizeof(double))
    #    cdef double* right_value = <double*> calloc(self.n_outputs, sizeof(double))
    #    cdef double* value = <double*> calloc(self.n_outputs, sizeof(double))
    #    cdef double* _d = <double*> calloc(self.n_samples, sizeof(double))
    #    cdef double _d_output = 0
    #    cdef double _delta_y_bar = 0
    #    memset(left_value, 0, self.n_outputs * sizeof(double))
    #    memset(right_value, 0, self.n_outputs * sizeof(double))
    #    memset(value, 0, self.n_outputs * sizeof(double))
    #    memset(_d, 0, self.n_samples * sizeof(double))

    #    for k in range(self.n_outputs):
    #        if self.weighted_n_node_samples > 0:
    #            value[k] = self.sum_total[k]/self.weighted_n_node_samples
    #        else:
    #            value[k] = 0
    #        if self.weighted_n_left > 0:
    #            left_value[k] = self.sum_left[k]/self.weighted_n_left
    #        else:
    #            left_value[k] = 0
    #        if self.weighted_n_right > 0:
    #            right_value[k] = self.sum_right[k]/self.weighted_n_right
    #        else:
    #            right_value[k] = 0

    #    if alpha != 1:
            #if new_y_bar != NULL:
    #        if cpy_inplace:
    #            if is_prob:
    #                for p in range(self.n_node_samples):
    #                    i = samples[p]
    #                    for k in range(self.n_outputs):
    #                        _d[i] += left_prob_weight[i] * left_value[k] * left_value[k]
    #                        _d[i] += (1- left_prob_weight[i]) * right_value[k] * right_value[k]
    #                        _d[i] -= value[k] * value[k]
    #                        _delta_y_bar = left_prob_weight[i] * left_value[k] + (1 - left_prob_weight[i]) * right_value[k] - value[k]
    #                        _d[i] -= 2 * y_bar[i] * _delta_y_bar
    #                        _d[i] -= self.prob_weight[i] * (_delta_y_bar) ** 2
                            #new_y_bar[i] += self.prob_weight[i] * _delta_y_bar
    #                        y_bar[i] += self.prob_weight[i] * _delta_y_bar
    #                    _d[i] *= self.prob_weight[i]
    #                    _d[i] += d[i]
                        #new_d[i] = _d[i]
    #                    d[i] = _d[i]
    #                    _d_output += gamma[i] * pow(_d[i], 1/alpha)
    #            else:
    #                for p in range(pos):
    #                    i = samples[p]
    #                    for k in range(self.n_outputs):
    #                        _d[i] += left_value[k] * left_value[k]
    #                        _d[i] -= value[k] * value[k]
    #                        _delta_y_bar = left_value[k] - value[k]
    #                        _d[i] -= 2 * y_bar[i] * _delta_y_bar
    #                        _d[i] -= self.prob_weight[i] * (_delta_y_bar) ** 2
                            #new_y_bar[i] += self.prob_weight[i] * _delta_y_bar
    #                        y_bar[i] += self.prob_weight[i] * _delta_y_bar
    #                    _d[i] *= self.prob_weight[i]
    #                    _d[i] += d[i]
                        #new_d[i] = _d[i]
    #                    d[i] = _d[i]
    #                    _d_output += gamma[i] * pow(_d[i], 1/alpha)

    #                for p in range(pos, self.n_node_samples):
    #                    i = samples[p]
    #                    for k in range(self.n_outputs):
    #                        _d[i] += right_value[k] * right_value[k]
    #                        _d[i] -= value[k] * value[k]
    #                        _delta_y_bar = right_value[k] - value[k]
    #                        _d[i] -= 2 * y_bar[i] * _delta_y_bar
    #                        _d[i] -= self.prob_weight[i] * (_delta_y_bar) ** 2
    #                        #new_y_bar[i] += self.prob_weight[i] * _delta_y_bar
    #                        y_bar[i] += self.prob_weight[i] * _delta_y_bar
    #                    _d[i] *= self.prob_weight[i]
    #                    _d[i] += d[i]
    #                    #new_d[i] = _d[i]
    #                    d[i] = _d[i]
    #                    _d_output += gamma[i] * pow(_d[i], 1/alpha)
    #        else:
    #            if is_prob:
    #                for p in range(self.n_node_samples):
    #                    i = samples[p]
    #                    for k in range(self.n_outputs):
    #                        _d[i] += left_prob_weight[i] * left_value[k] * left_value[k]
    #                        _d[i] += (1- left_prob_weight[i]) * right_value[k] * right_value[k]
    #                        _d[i] -= value[k] * value[k]
    #                        _delta_y_bar = left_prob_weight[i] * left_value[k] + (1 - left_prob_weight[i]) * right_value[k] - value[k]
    #                        _d[i] -= 2 * y_bar[i] * _delta_y_bar
    #                        _d[i] -= self.prob_weight[i] * (_delta_y_bar) ** 2
    #                    _d[i] *= self.prob_weight[i]
    #                    _d[i] += d[i]
    #                    _d_output += gamma[i] * pow(_d[i], 1/alpha)
    #            else:
    #                for p in range(pos):
    #                    i = samples[p]
    #                    for k in range(self.n_outputs):
    #                        _d[i] += left_value[k] * left_value[k]
    #                        _d[i] -= value[k] * value[k]
    #                        _delta_y_bar = left_value[k] - value[k]
    #                        _d[i] -= 2 * y_bar[i] * _delta_y_bar
    #                        _d[i] -= self.prob_weight[i] * (_delta_y_bar) ** 2
    #                    _d[i] *= self.prob_weight[i]
    #                    _d[i] += d[i]
    #                    _d_output += gamma[i] * pow(_d[i], 1/alpha)

    #                for p in range(pos, self.n_node_samples):
    #                    i = samples[p]
    #                    for k in range(self.n_outputs):
    #                        _d[i] += right_value[k] * right_value[k]
    #                        _d[i] -= value[k] * value[k]
    #                        _delta_y_bar = right_value[k] - value[k]
    #                        _d[i] -= 2 * y_bar[i] * _delta_y_bar
    #                        _d[i] -= self.prob_weight[i] * (_delta_y_bar) ** 2
    #                    _d[i] *= self.prob_weight[i]
    #                    _d[i] += d[i]
    #                    _d_output += gamma[i] * pow(_d[i], 1/alpha)
    #    else:
            #if new_y_bar != NULL:
    #        if cpy_inplace:
    #            if is_prob:
    #                for p in range(self.n_node_samples):
    #                    i = samples[p]
    #                    for k in range(self.n_outputs):
    #                        _d[i] += left_prob_weight[i] * left_value[k] * left_value[k]
    #                        _d[i] += (1- left_prob_weight[i]) * right_value[k] * right_value[k]
    #                        _d[i] -= value[k] * value[k]
    #                        _delta_y_bar = left_prob_weight[i] * left_value[k] + (1 - left_prob_weight[i]) * right_value[k] - value[k]
    #                        _d[i] -= 2 * y_bar[i] * _delta_y_bar
    #                        _d[i] -= self.prob_weight[i] * (_delta_y_bar) ** 2
                            #new_y_bar[i] += self.prob_weight[i] * _delta_y_bar
    #                        y_bar[i] += self.prob_weight[i] * _delta_y_bar
    #                    _d[i] *= self.prob_weight[i]
    #                    _d[i] += d[i]
                        #new_d[i] = _d[i]
    #                    d[i] = _d[i]
    #                    _d_output += gamma[i] * _d[i]
    #            else:
    #                for p in range(pos):
    #                    i = samples[p]
    #                    for k in range(self.n_outputs):
    #                        _d[i] += left_value[k] * left_value[k]
    #                        _d[i] -= value[k] * value[k]
    #                        _delta_y_bar = left_value[k] - value[k]
    #                        _d[i] -= 2 * y_bar[i] * _delta_y_bar
    #                        _d[i] -= self.prob_weight[i] * (_delta_y_bar) ** 2
                            #new_y_bar[i] += self.prob_weight[i] * _delta_y_bar
    #                        y_bar[i] += self.prob_weight[i] * _delta_y_bar
    #                    _d[i] *= self.prob_weight[i]
    #                    _d[i] += d[i]
                        #new_d[i] = _d[i]
    #                    d[i] = _d[i]
    #                    _d_output += gamma[i] * _d[i]

    #                for p in range(pos, self.n_node_samples):
    #                    i = samples[p]
    #                    for k in range(self.n_outputs):
    #                        _d[i] += right_value[k] * right_value[k]
    #                        _d[i] -= value[k] * value[k]
    #                        _delta_y_bar = right_value[k] - value[k]
    #                        _d[i] -= 2 * y_bar[i] * _delta_y_bar
    #                        _d[i] -= self.prob_weight[i] * (_delta_y_bar) ** 2
                            #new_y_bar[i] += self.prob_weight[i] * _delta_y_bar
    #                        y_bar[i] += self.prob_weight[i] * _delta_y_bar
    #                    _d[i] *= self.prob_weight[i]
    #                    _d[i] += d[i]
                        #new_d[i] = _d[i]
    #                    d[i] = _d[i]
    #                    _d_output += gamma[i] * _d[i]
    #        else:
    #            if is_prob:
    #                for p in range(self.n_node_samples):
    #                    i = samples[p]
    #                    for k in range(self.n_outputs):
    #                        _d[i] += left_prob_weight[i] * left_value[k] * left_value[k]
    #                        _d[i] += (1- left_prob_weight[i]) * right_value[k] * right_value[k]
    #                        _d[i] -= value[k] * value[k]
    #                        _delta_y_bar = left_prob_weight[i] * left_value[k] + (1 - left_prob_weight[i]) * right_value[k] - value[k]
    #                        _d[i] -= 2 * y_bar[i] * _delta_y_bar
    #                        _d[i] -= self.prob_weight[i] * (_delta_y_bar) ** 2
    #                    _d[i] *= self.prob_weight[i]
    #                    _d[i] += d[i]
    #                    _d_output += gamma[i] * _d[i]
    #            else:
    #                for p in range(pos):
    #                    i = samples[p]
    #                    for k in range(self.n_outputs):
    #                        _d[i] += left_value[k] * left_value[k]
    #                        _d[i] -= value[k] * value[k]
    #                        _delta_y_bar = left_value[k] - value[k]
    #                        _d[i] -= 2 * y_bar[i] * _delta_y_bar
    #                        _d[i] -= self.prob_weight[i] * (_delta_y_bar) ** 2
    #                    _d[i] *= self.prob_weight[i]
    #                    _d[i] += d[i]
    #                    _d_output += gamma[i] * _d[i]

    #                for p in range(pos, self.n_node_samples):
    #                    i = samples[p]
    #                    for k in range(self.n_outputs):
    #                        _d[i] += right_value[k] * right_value[k]
    #                        _d[i] -= value[k] * value[k]
    #                        _delta_y_bar = right_value[k] - value[k]
    #                        _d[i] -= 2 * y_bar[i] * _delta_y_bar
    #                        _d[i] -= self.prob_weight[i] * (_delta_y_bar) ** 2
    #                    _d[i] *= self.prob_weight[i]
    #                    _d[i] += d[i]
    #                    _d_output += gamma[i] * _d[i]
    #    free(left_value)
    #    free(right_value)
    #    free(value)
    #    free(_d)
    #    return _d_output
    #    return 0


    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end])."""


        cdef DOUBLE_t* y = self.y
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos
        #cdef SIZE_t start = self.start
        cdef SIZE_t n_node_samples = self.n_node_samples
        cdef double* prob_weight = self.prob_weight

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef double sq_sum_left = 0.0
        cdef double sq_sum_right

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0
        cdef DOUBLE_t y_ik

        #for p in range(start, pos):
        for p in range(0, pos):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i] * prob_weight[i]
            else:
                w = prob_weight[i]

            for k in range(self.n_outputs):
                y_ik = y[i * self.y_stride + k]
                sq_sum_left += w * y_ik * y_ik

        sq_sum_right = self.sq_sum_total - sq_sum_left

        if self.weighted_n_left > 0:
            impurity_left[0] = sq_sum_left / self.weighted_n_left
            for k in range(self.n_outputs):
                impurity_left[0] -= (sum_left[k] / self.weighted_n_left) ** 2.0
            impurity_left[0] /= self.n_outputs
        else:
            impurity_left[0] = 0
            
        if self.weighted_n_right > 0:
            impurity_right[0] = sq_sum_right / self.weighted_n_right
            for k in range(self.n_outputs):
                impurity_right[0] -= (sum_right[k] / self.weighted_n_right) ** 2.0
            impurity_right[0] /= self.n_outputs
        else:
            impurity_right[0] = 0

    cdef void prob_children_impurity(self, double* impurity_left,
                                double* impurity_right, 
                                double* left_prob_weight) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end])."""


        cdef DOUBLE_t* y = self.y
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        #cdef SIZE_t pos = self.pos
        #cdef SIZE_t start = self.start
        cdef SIZE_t n_node_samples = self.n_node_samples
        cdef double* prob_weight = self.prob_weight

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef double sq_sum_left = 0.0
        cdef double sq_sum_right

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0
        cdef DOUBLE_t y_ik

        #for p in range(start, pos):
        for p in range(n_node_samples):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i] * prob_weight[i] * left_prob_weight[i]
            else:
                w = prob_weight[i] * left_prob_weight[i]

            for k in range(self.n_outputs):
                y_ik = y[i * self.y_stride + k]
                sq_sum_left += w * y_ik * y_ik

        sq_sum_right = self.sq_sum_total - sq_sum_left

        #impurity_left[0] = sq_sum_left / self.weighted_n_left
        #impurity_right[0] = sq_sum_right / self.weighted_n_right

        #for k in range(self.n_outputs):
        #    impurity_left[0] -= (sum_left[k] / self.weighted_n_left) ** 2.0
        #    impurity_right[0] -= (sum_right[k] / self.weighted_n_right) ** 2.0

        #impurity_left[0] /= self.n_outputs
        #impurity_right[0] /= self.n_outputs

        if self.weighted_n_left > 0:
            impurity_left[0] = sq_sum_left / self.weighted_n_left
            for k in range(self.n_outputs):
                impurity_left[0] -= (sum_left[k] / self.weighted_n_left) ** 2.0
            impurity_left[0] /= self.n_outputs
        else:
            impurity_left[0] = 0
            
        if self.weighted_n_right > 0:
            impurity_right[0] = sq_sum_right / self.weighted_n_right
            for k in range(self.n_outputs):
                impurity_right[0] -= (sum_right[k] / self.weighted_n_right) ** 2.0
            impurity_right[0] /= self.n_outputs
        else:
            impurity_right[0] = 0

