package NN.centipede.model.activation;

import NN.centipede.numpy.NDArray;

public interface Activation {
    NDArray forward(NDArray x);
    NDArray backward(NDArray delta);
}