package NN.centipede.model.activation;

import NN.centipede.numpy.NDArray;
import NN.centipede.numpy.Numpy.np;

/**
 * y = 1/(1 + e^-x)
 * y/dx = y(1-y)
 */
public class Sigmoid implements Activation{
    private NDArray x;

    /**
     * sigmoid(x)=1/(e^(-x)+1)
     */
    @Override
    public NDArray forward(NDArray x) {
        this.x = np.exp(x.negative()).add(1).reciprocal();
        return this.x;
    }

    /**
     * sigmoid′(x)=sigmoid(x)⋅(1−sigmoid(x))
     */
    @Override
    public NDArray backward(NDArray delta) {
        return np.multiply(delta, x).multiply(x.negative().add(1));
    }
}