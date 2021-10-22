package NN.centipede.model.activation;

import NN.centipede.numpy.NDArray;
import NN.centipede.numpy.Numpy.np;

public class Relu {
    private NDArray x;

    public NDArray forward(NDArray x) {
        this.x = x;
        return np.maximum(x, 0.);
    }

    public NDArray backward(NDArray delta) {
        //delta[self.x<0] = 0;
        np.checkset(delta, x, (n)->(double)n<0, 0.);
        return delta;
    }
}
