package DL;

import java.net.InetAddress;

import javax.swing.Action;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.inverse.InvertMatrix;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.primitives.Pair;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.api.instrumentation.InMemoryInstrumentation;

public class PILoss implements ILossFunction {

	private static double eps = 0.0000000001;

	// for mask the selective actions
	private static INDArray getInMask(INDArray masks, int out_size) {
		int batch = (int) masks.shape()[0];
		float[][] mask_code = new float[batch][out_size];
		int[] sel_arg = masks.toIntVector();

		// initialize mask_code
		for (int i = 0; i < batch; i++) {
			for (int j = 0; j < out_size; j++) {
				mask_code[i][j] = 0;
			}

			mask_code[i][sel_arg[i]] = 1;
		}

		INDArray inMask = Nd4j.create(mask_code);

		return inMask;
	}

	private static INDArray safeLog(INDArray probs, INDArray inMask) {
		// System.out.println("probs:" + probs);
		// System.out.println("max probs:"+ probs.max(1));
		INDArray results = probs.mul(inMask).sum(1);
		double[] logs = results.toDoubleVector();
		int length = logs.length;
		// System.out.print("log:");
		for (int i = 0; i < length; i++) {
			// System.out.print(logs[i] + ", ");
			if (Math.abs(logs[i]) >= eps) {
				logs[i] = Math.log(logs[i]);
			} else {
				logs[i] = 0;
			}
		}
		// System.out.println("");
		INDArray indLOGS = Nd4j.create(logs, new int[] { length, 1 });
		return indLOGS;
	}

	// here
	// labels = rewards, masks = actions

	// Score Array - DIY------------------------------------------
	private INDArray scoreArray(INDArray rewards, INDArray preOutput, IActivation actFn, INDArray actions) {
		// action Mask
		int out_size = (int) (preOutput.shape()[1]);
		INDArray inMask = getInMask(actions, out_size);
		INDArray output = actFn.getActivation(preOutput.dup(), true);

		INDArray logvals = safeLog(output, inMask);
		INDArray PiLoss = rewards.mul(-1).mul(logvals).repeat(1, out_size).mul(inMask);

		// scoreArr.addi(yMinusyHat); // regulization part

		return PiLoss;
	}

	@Override
	public INDArray computeGradient(INDArray rewards, INDArray preOutput, IActivation actFn, INDArray actions) {
		int out_size = (int) (preOutput.shape()[1]);
		INDArray inMask = getInMask(actions, out_size);

		INDArray output = actFn.getActivation(preOutput.dup(), true);

		INDArray p_ = output.dup().add(1e-10); // output.add(eps);

		INDArray piGD = rewards.mul(-1).repeat(1, out_size).div(p_).mul(inMask);

		// System.out.println("piGD:" + piGD);
		// System.out.println("rewards:" + rewards);
		// System.out.println("inMask:" + inMask);
		// .sub(Transforms.sign(yMinusyHat)); //regularization part

		// everything below remain the same.
		INDArray dLdPreout;
		if (false) {
			// dLdPreout = computeNaturalPolicyGradient(p_, inMask, output, piGD);
			dLdPreout = actFn.backprop(preOutput.dup(), computeNaturalPolicyGradient(p_, inMask, output, piGD))
					.getFirst();
		} else {
			dLdPreout = actFn.backprop(preOutput.dup(), piGD).getFirst();
		}

		return dLdPreout;
	}

	public INDArray computeNaturalPolicyGradient(INDArray p_, INDArray inMask, INDArray output, INDArray piGD) {
		INDArray differential_softmax = p_.div(p_).div(p_).mul(inMask);
		// INDArray double_differential_softmax =
		// differential_softmax.mul(differential_softmax.transpose());
		// INDArray computedDifferential_softmax = computeDifferentialSofmax(output);
		/*
		 * INDArray double_differential_softmax = computedDifferential_softmax
		 * .prod(computedDifferential_softmax.transpose());
		 */

		INDArray double_differential_softmax = differential_softmax.mmul(differential_softmax.transpose());
		// INDArray double_differential_softmax = computedDifferential_softmax
		// .mmul(computedDifferential_softmax.transpose());
		System.out.println("double: " + double_differential_softmax.shapeInfoToString());

		// INDArray computed_piGD = InvertMatrix.invert(double_differential_softmax,
		// false).mul(piGD);
		INDArray computed_piGD = computeInverseArray(double_differential_softmax).mmul(piGD);

		return computed_piGD;
	}

	public INDArray computeInverseArray(INDArray target) {
		int outer_size = (int) (target.shape()[0]);
		int inner_size = (int) (target.shape()[1]);
		double[][] result = new double[outer_size][inner_size];
		for (int i = 0; i < outer_size; i++)
			for (int j = 0; j < inner_size; j++)
				result[i][j] = i == j ? 1 : 0;

		double buffer;
		double[][] converted_target = new double[outer_size][inner_size];
		for (int i = 0; i < outer_size; i++) {
			converted_target[i] = target.getRow(i).toDoubleVector();
		}

		for (int i = 0; i < outer_size; i++) {
			buffer = 1 / converted_target[i][i];
			for (int j = 0; j < inner_size; j++) {
				converted_target[i][j] *= buffer;
				result[i][j] *= buffer;
			}

			for (int j = 0; j < inner_size; j++) {
				buffer = converted_target[j][i];
				for (int k = 0; k < outer_size; k++) {
					converted_target[j][k] -= converted_target[i][k] * buffer;
					result[j][k] -= result[i][k] * buffer;
				}
			}
		}

		INDArray converted_result = Nd4j.create(result);
		return converted_result;
	}

	public INDArray computeDifferentialSofmax(INDArray output) {
		int batch_len = (int) (output.shape()[0]);
		int action_len = (int) (output.shape()[1]);
		double[][] result = new double[batch_len][action_len];

		for (int batch = 0; batch < batch_len; batch++) {
			for (int i = 0; i < action_len; i++) {
				for (int j = 0; j < action_len; j++) {
					if (i == j) {
						result[batch][i] += output.getRow(batch / batch_len).getFloat(i)
								- Math.pow(output.getRow(batch / batch_len).getFloat(i), 2);
					} else {
						result[batch][i] += -output.getRow(batch / batch_len).getFloat(i)
								* output.getRow(batch / batch_len).getFloat(j);
					}
				}
			}
		}
		INDArray converted_result = Nd4j.create(result);
		return converted_result;
	}

	// DIY-------------------------------------------------------

	@Override
	public double computeScore(INDArray labels, INDArray preOutput, IActivation actFn, INDArray mask, boolean average) {
		INDArray scoreArr = scoreArray(labels, preOutput, actFn, mask);
		double score = scoreArr.sumNumber().doubleValue();
		if (average)
			score = score / scoreArr.size(0);
		return score;
	}

	@Override
	public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation actFn, INDArray mask) {
		INDArray scoreArr = scoreArray(labels, preOutput, actFn, mask);
		return scoreArr.sum(1);
	}

	@Override
	public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation actFn,
			INDArray mask, boolean average) {

		return new Pair<>(computeScore(labels, preOutput, actFn, mask, average),
				computeGradient(labels, preOutput, actFn, mask));
	}

	@Override
	public String name() {
		return "RewardLoss";

	}

	public boolean equals(Object o) {
		if (o == this)
			return true;
		if (!(o instanceof PILoss))
			return false;
		final PILoss other = (PILoss) o;
		if (!other.canEqual((Object) this))
			return false;
		return true;
	}

	public int hashCode() {
		int result = 1;
		return result;
	}

	protected boolean canEqual(Object other) {
		return other instanceof PILoss;
	}

}
