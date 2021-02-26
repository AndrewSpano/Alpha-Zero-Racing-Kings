"""
Created on February 18 2021

@author: Andreas Spanopoulos

Utility functions for the Neural Network of the Alpha Zero algorithm.
"""


from src.utils.error_utils import InvalidArchitectureError


def compute_output_shape(current_shape, kernel_size, stride, padding):
    """
    :param tuple current_shape:  The current shape of the data before a convolution is applied.
    :param tuple kernel_size:    The kernel size of the current convolution operation.
    :param tuple stride:         The stride of the current convolution operation.
    :param tuple padding:        The padding of the current convolution operation.

    :return:  The shape after a convolution operation with the above parameters is applied.
    :rtype:   tuple

    The formula used to compute the final shape is:

        component[i] = floor((N[i] - K[i] + 2 * P[i]) / S[i]) + 1

        where, N = current shape of the data
               K = kernel size
               P = padding
               S = stride
    """
    # get the dimension of the data compute each component using the above formula
    dimensions = len(current_shape)
    return tuple((current_shape[i] - kernel_size[i] + 2 * padding[i]) // stride[i] + 1
                 for i in range(dimensions))


def same_padding(current_shape, kernel_size, stride):
    """
    :param tuple current_shape:            The current (and target) shape of the data before
                                            (and after) a convolution is applied.
    :param Union[int, tuple] kernel_size:  The kernel size of the current convolution operation.
    :param Union[int, tuple] stride:       The stride of the current convolution operation.

    :return:  The padding needed in order to have the same shape after the convolution.
    :rtype:   tuple

    :raises:
        InvalidArchitectureError:  If the hyperperameters given result in an invalid same padding.

    Computes the padding needed in order to have the same output shape as the input shape after a
    convolution operation is applied. The formula used is:

        padding[i] = ((N[i] - 1) * S[i] + k[i] - N[i]) / 2

        where, N = current shape of the data
               K = kernel size
               S = stride
    """
    # get the dimension of the data compute each component using the above formula
    dimensions = len(current_shape)

    # compute here the padding needed
    padding = []
    for i in range(dimensions):
        k = kernel_size if isinstance(kernel_size, int) else kernel_size[i]
        s = stride if isinstance(stride, int) else stride[i]

        if current_shape[i] < k:
            raise InvalidArchitectureError(
                msg=f'The kernel size {k} is smaller than respective component {current_shape[i]} '
                    f'of the current shape: {current_shape}.')
        elif ((current_shape[i] - 1) * s + k - current_shape[i]) % 2 != 0:
            raise InvalidArchitectureError(
                msg=f'The term: (N[i] - 1) * S[i] + k[i] - N[i] ( ({current_shape[i]} - 1) * {s} + '
                    f'{k} - {current_shape[i]}) is not divisible by 2, and therefore the padding '
                    f'will be uneven.')

        padding.append(((current_shape[i] - 1) * s + k - current_shape[i]) // 2)

    return tuple(padding)
