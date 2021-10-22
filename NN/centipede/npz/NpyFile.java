package NN.centipede.npz;

import NN.centipede.numpy.NDArray;
import NN.centipede.numpy.Numpy.np;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.util.stream.IntStream;

/**
 * A file in NPY format.
 *
 * Currently unsupported types:
 *
 *   * unsigned integral types (treated as signed)
 *   * bit field,
 *   * complex,
 *   * object,
 *   * Unicode
 *   * void*
 *   * intersections aka types for structured arrays.
 *
 * See http://docs.scipy.org/doc/numpy-dev/neps/npy-format.html
 */
public class NpyFile {

	public static NDArray read(Header header, ByteBuffer chunks) {
        int size = IntStream.of(header.shape).reduce(1, (a,b)->a*b);

        chunks.order(header.order);
        DoubleBuffer db = chunks.asDoubleBuffer();
        double[] data = new double[size];
        int offset = 0;
        while (db.hasRemaining()) {
            size = db.remaining();
            db.get(data, offset, size);
            offset += size;
        }
        return np.array(data, header.shape);
	}
}