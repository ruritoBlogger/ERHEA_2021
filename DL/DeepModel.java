package DL;

import java.io.File;

import org.deeplearning4j.nn.conf.layers.*;
import org.nd4j.linalg.activations.Activation;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import org.deeplearning4j.util.ModelSerializer;

import org.nd4j.linalg.api.ndarray.INDArray;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import org.nd4j.linalg.lossfunctions.LossFunctions;

import org.deeplearning4j.nn.conf.GradientNormalization;

import javax.swing.*;

import enumerate.Action;

import java.util.*;

public class DeepModel {
	static private MultiLayerNetwork net;
	static private int seed = 123; // seed is 123;

	static private double learningRate = 0.001;

	static private HashMap<Action, Integer> actToIndex;
	final static private int numOutputs = 56;
	final static private int n_epochs = 3;

	final static private String directory = "./data/aiData/ERHEA_PPO_PG";
	private static String oppName;
	private static String myName;
	private static String oppModelName;

	private static String lossType;
	private String character;

	public DeepModel(int input_size, String myName, String oppName, String lossType, String character) {
		this.myName = myName;
		this.oppName = oppName;
		this.lossType = lossType;
		this.character = character;
		this.oppModelName = "oppModel.zip";
		this.oppModelName = this.lossType + "-" + this.oppModelName;
		// network learning model
		System.out.println("Create " + this.lossType + " Model!");

		MultiLayerConfiguration conf = getConfig(input_size);
		net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));

		actToIndex = new HashMap<Action, Integer>();
		for (int i = 0; i < Action.values().length; i++) {
			actToIndex.put(Action.values()[i], i);
		}

	}

	// num input: my HP, my Energy, my Posx, my Posy, my state, diff posx, opp HP,
	// opp Energy, opp Posx, opp Posy, opp State,
	// my Last Act,opp Last Act
	private MultiLayerConfiguration getConfig(int input_size) {

		if (lossType.toLowerCase() == "pi") {
			return new NeuralNetConfiguration.Builder().seed(seed)
					// .weightInit(WeightInit.XAVIER)
					.weightInit(WeightInit.ONES).biasInit(0.0)
					.gradientNormalization(GradientNormalization.ClipL2PerLayer).gradientNormalizationThreshold(1.0)
					.updater(new Adam(learningRate, 0.99, 0.999, 0.00000001)) //
					.list().layer(0, new OutputLayer.Builder(new PILoss()) // best performance: PILoss()
							.activation(Activation.SOFTMAX) // pi: softmax, q: identity
							.nIn(input_size).nOut(numOutputs).build())
					.build();

		} else if (lossType.toLowerCase() == "sl") {
			return new NeuralNetConfiguration.Builder().seed(seed)
					// .weightInit(WeightInit.XAVIER)
					.weightInit(WeightInit.ONES).gradientNormalization(GradientNormalization.ClipL2PerLayer)
					.gradientNormalizationThreshold(1.0).updater(new Adam(learningRate, 0.99, 0.999, 0.00000001)) //
					.list().layer(0, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT) // best performance:
																								// PILoss()
							.activation(Activation.SOFTMAX) // pi: softmax, q: identity
							.nIn(input_size).nOut(numOutputs).build())
					.build();

		}
		System.err.printf("Not right loss:%s \n", lossType);
		return null;

	}

	public void train(float[][] cur_input_info, float[][] nx_input_info, float[] sel_actions, float[] opp_sel_actions,
			float win_signal) {
		long startTime = System.nanoTime();
		int batch_size = sel_actions.length;

		INDArray sel_actsND = Nd4j.create(sel_actions, new int[] { batch_size, 1 });
		INDArray opp_sel_actsND = Nd4j.create(opp_sel_actions, new int[] { batch_size, 1 });
		INDArray cur_nd = Nd4j.create(cur_input_info);

		if (lossType == "sl") {
			int[] labels = new int[opp_sel_actions.length];
			for (int i = 0; i < labels.length; i++)
				labels[i] = (int) (opp_sel_actions[i]);
			for (int i = 0; i < n_epochs; i++) {
				net.fit(cur_nd, labels);
				System.out.println("Loss:" + net.score());
			}
		} else {
			float discount = 0.8f;
			float[] targets_opp = new float[batch_size];
			float[] targets_sel = new float[batch_size];
			float[] opp_dp = new float[batch_size + 1];
			float[] sel_dp = new float[batch_size + 1];
			float reward_sel = 0;
			float reward_opp = 0;

			float lambda = 0.6f;
			int lambda_len = 3;

			float[] opp_lambda = new float[lambda_len];
			float[] sel_lambda = new float[lambda_len];

			targets_opp[batch_size - 1] = win_signal;
			targets_sel[batch_size - 1] = win_signal;
			opp_dp[0] = 0;
			sel_dp[0] = 0;

			for (int i = batch_size - 2; i >= 0; i--) {

				reward_sel = -Math.max(nx_input_info[i][0] - nx_input_info[i + 1][0], 0.f) - 0.01f;
				reward_opp = Math.max(nx_input_info[i][1] - nx_input_info[i + 1][1], 0.f) + 0.01f;

				// v1 -- 55.9
				// reward_sel = -Math.max(nx_input_info[i][0] - nx_input_info[i+1][0], 0.f) +
				// 0.01f;
				// reward_opp = Math.max(nx_input_info[i][1] - nx_input_info[i+1][1], 0.f) -
				// 0.01f;

				// // my player - soso-0
				// reward = Math.max(nx_input_info[i][0] - nx_input_info[i+1][0], 0.f);

				// // opp player - not so well
				// reward = Math.max(nx_input_info[i][1] - nx_input_info[i+1][1], 0.f);

				// opp player --not so well our player
				// reward = Math.max(nx_input_info[i][1] - nx_input_info[i+1][1], 0.f) -
				// Math.max(nx_input_info[i][0] - nx_input_info[i+1][0], 0.f);

				// no well
				// reward = Math.max(nx_input_info[i][0] - nx_input_info[i+1][0], 0.f) +
				// Math.max(nx_input_info[i][1] - nx_input_info[i+1][1], 0.f);

				// not well
				// reward = Math.max(nx_input_info[i][0] - nx_input_info[i+1][0], 0.f) -
				// Math.max(nx_input_info[i][1] - nx_input_info[i+1][1], 0.f);

				// our player hp diff opp player hp diff
				// reward = Math.max(nx_input_info[i][0] - nx_input_info[i+1][0], 0.f) -
				// Math.max(nx_input_info[i][1] - nx_input_info[i+1][1], 0.f);
				/*
				 * targets_opp[i] = reward_opp + discount * targets_opp[i + 1]; targets_sel[i] =
				 * reward_sel + discount * targets_sel[i + 1];
				 */

				// targets_opp[i] = reward_opp + discount * targets_opp[i + 1];
				// targets_sel[i] = reward_sel + discount * targets_sel[i + 1];
				opp_dp[batch_size - 1 - i] = reward_opp + opp_dp[batch_size - 2 - i];
				sel_dp[batch_size - 1 - i] = reward_sel + sel_dp[batch_size - 2 - i];

				opp_lambda[0] = reward_opp + discount * targets_opp[i + 1];
				sel_lambda[0] = reward_sel + discount * targets_sel[i + 1];

				if (i < batch_size - 2) {
					opp_lambda[1] = opp_dp[batch_size - 1 - i] - opp_dp[batch_size - 3 - i]
							+ discount * discount * targets_opp[i + 1];
					sel_lambda[1] = sel_dp[batch_size - 1 - i] - sel_dp[batch_size - 3 - i]
							+ discount * discount * targets_sel[i + 1];
				} else {
					opp_lambda[1] = 0;
					sel_lambda[1] = 0;
				}

				if (i < batch_size - 3) {
					opp_lambda[2] = opp_dp[batch_size - 1 - i] - opp_dp[batch_size - 4 - i]
							+ discount * discount * discount * targets_opp[i + 1];
					sel_lambda[2] = sel_dp[batch_size - 1 - i] - sel_dp[batch_size - 4 - i]
							+ discount * discount * discount * targets_sel[i + 1];
				} else {
					opp_lambda[2] = 0;
					sel_lambda[2] = 0;
				}

				if (i < batch_size - 4) {
					opp_lambda[3] = opp_dp[batch_size - 1 - i] - opp_dp[batch_size - 5 - i]
							+ (float) (Math.pow(discount, 4)) * targets_opp[i + 1];
					sel_lambda[3] = sel_dp[batch_size - 1 - i] - sel_dp[batch_size - 5 - i]
							+ (float) (Math.pow(discount, 4)) * targets_sel[i + 1];
				} else {
					opp_lambda[3] = 0;
					sel_lambda[3] = 0;
				}

				if (i < batch_size - 5) {
					opp_lambda[4] = opp_dp[batch_size - 1 - i] - opp_dp[batch_size - 6 - i]
							+ (float) (Math.pow(discount, 5)) * targets_opp[i + 1];
					sel_lambda[4] = sel_dp[batch_size - 1 - i] - sel_dp[batch_size - 6 - i]
							+ (float) (Math.pow(discount, 5)) * targets_sel[i + 1];
				} else {
					opp_lambda[4] = 0;
					sel_lambda[4] = 0;
				}

				if (i < batch_size - 6) {
					opp_lambda[5] = opp_dp[batch_size - 1 - i] - opp_dp[batch_size - 7 - i]
							+ (float) (Math.pow(discount, 6)) * targets_opp[i + 1];
					sel_lambda[5] = sel_dp[batch_size - 1 - i] - sel_dp[batch_size - 7 - i]
							+ (float) (Math.pow(discount, 6)) * targets_sel[i + 1];
				} else {
					opp_lambda[5] = 0;
					sel_lambda[5] = 0;
				}

				if (i < batch_size - 7) {
					opp_lambda[6] = opp_dp[batch_size - 1 - i] - opp_dp[batch_size - 8 - i]
							+ (float) (Math.pow(discount, 7)) * targets_opp[i + 1];
					sel_lambda[6] = sel_dp[batch_size - 1 - i] - sel_dp[batch_size - 8 - i]
							+ (float) (Math.pow(discount, 7)) * targets_sel[i + 1];
				} else {
					opp_lambda[6] = 0;
					sel_lambda[6] = 0;
				}

				for (int j = 0; j < lambda_len; j++) {
					targets_opp[i] += opp_lambda[j] * Math.pow(lambda, j);
					targets_sel[i] += sel_lambda[j] * Math.pow(lambda, j);
				}
				targets_opp[i] *= (1 - lambda);
				targets_sel[i] *= (1 - lambda);
			}

			System.out.println("");

			// // Normalize reward part
			// for (int i=0; i<batch_size-1;i++){
			// targets_f[i] = targets_f[i]/ batch_size;
			// }

			// target values
			INDArray target_sel = Nd4j.create(targets_sel, new int[] { batch_size, 1 });
			INDArray target_opp = Nd4j.create(targets_opp, new int[] { batch_size, 1 });

			// train repeatedly
			// for (int i = 0; i < n_epochs; i++) {

			int divide_key = batch_size - 1;
			int[] key = new int[divide_key];
			for (int i = 0; i < divide_key; i++) {
				key[i] = i + 1;
			}

			// INDArray divide_sel_actsND = Nd4j.create(new float[0], new int[] { 0 });
			INDArray divide_sel_actsND = sel_actsND.getRows(key);
			INDArray divide_opp_sel_actsND = opp_sel_actsND.getRows(key);
			INDArray divide_cur_nd = cur_nd.getRows(key);
			INDArray divide_target_sel = target_sel.getRows(key);
			INDArray divide_target_opp = target_opp.getRows(key);

			for (int i = 0; i < 3; i++) {
				net.setMask(divide_sel_actsND);
				net.fit(divide_cur_nd, divide_target_sel);
				// System.out.println("Loss:" + net.score());

				net.setMask(divide_opp_sel_actsND);
				net.fit(divide_cur_nd, divide_target_opp);
			}
			System.out.println("処理時間: " + (System.nanoTime() - startTime) + "ナノ秒");
		}

	}

	public int forward(float[] input_info) {
		int num_out = 0;
		INDArray input;
		INDArray out;

		input = Nd4j.create(input_info);
		out = net.output(input);

		INDArray argmax;
		// System.out.println("Out:" + out);
		argmax = Nd4j.argMax(out, 1);
		num_out = (int) (argmax.getInt(0));

		return num_out;
	}

	public int forward(float[] input_info, LinkedList<Action> validActs) {

		int num_out = 0;
		double maxV = -99999999.0;
		double val;
		INDArray input;
		INDArray out;

		input = Nd4j.create(input_info);
		out = net.output(input);

		double[] ds = out.toDoubleVector();
		for (Action act : validActs) {
			int i = actToIndex.get(act);
			val = ds[i];
			if (val > maxV) {
				maxV = val;
				num_out = i;
			}
		}

		return num_out;

	}

	private String getPathName() {
		String myname;
		String oppname;
		myname = this.myName == "" ? "" : this.myName + "-";
		oppname = this.oppName == "" ? "" : this.oppName + "-";
		return this.directory + "/" + myname + oppname + this.oppModelName + '-' + this.character;
	}

	public void save() {
		File file = new File(this.directory);
		if (!file.exists()) {
			file.mkdir();
		}

		String pathName;
		pathName = getPathName();
		try {
			System.out.println("Save Model!");
			ModelSerializer.writeModel(net, new File(pathName), true); // false is not saved optimization parameters.
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	public void load() {
		String pathName;
		pathName = getPathName();
		try {
			File loadFile = new File(pathName);
			if (loadFile.exists()) {
				System.out.println("Loaded Model!");
				net = ModelSerializer.restoreMultiLayerNetwork(loadFile);
			} else {
				System.out.println("Could not find out the model!");
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
