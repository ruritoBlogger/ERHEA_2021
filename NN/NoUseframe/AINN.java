package NN.NoUseframe;

import NN.centipede.numpy.NDArray;
import NN.centipede.numpy.Numpy.np;
import enumerate.Action;

import java.io.*;
import java.util.ArrayList;
import java.util.Date;
import java.util.LinkedList;
import java.util.Vector;


public class AINN {
	public ArrayList<NDArray> nnWeights;
	public LinkedList<String> softmaxBeforeLog = new LinkedList<>();
	public LinkedList<String> softmaxAfterLog = new LinkedList<>();

	public AINN(String path, String character, String mode) {
		if ("ZEN".equals(character)) {
			character = "1";
		} else if ("GARNET".equals(character)) {
			character = "2";
		} else {
			character = "3";
		}
		this.nnWeights = getWeight(path, mode + "_" + character);
	}

	public static ArrayList<NDArray> getWeight(String path, String name) {
		NDArray weight;
		NDArray bias;
		ArrayList<NDArray> nn = new ArrayList<NDArray>();
		weight = np.loadtxt(path + "W0_" + name + ".csv", ",");
		bias = np.loadtxt(path + "b0_" + name + ".csv", ",");
		nn.add(weight);
		nn.add(bias);
		weight = np.loadtxt(path + "W1_" + name + ".csv", ",");
		bias = np.loadtxt(path + "b1_" + name + ".csv", ",");
		nn.add(weight);
		nn.add(bias);
		weight = np.loadtxt(path + "W2_" + name + ".csv", ",");
		bias = np.loadtxt(path + "b2_" + name + ".csv", ",");
		nn.add(weight);
		nn.add(bias);
		return nn;
	}

	public static final Action[] ACTIONSAIR = new Action[] {
		Action.AIR_A, Action.AIR_B, Action.AIR_D_DB_BA, Action.AIR_D_DB_BB,
		Action.AIR_D_DF_FA, Action.AIR_D_DF_FB, Action.AIR_DA, Action.AIR_DB,
		Action.AIR_F_D_DFA, Action.AIR_F_D_DFB, Action.AIR_FA, Action.AIR_FB,
		Action.AIR_UA, Action.AIR_UB};

public static final Action[] ACTIONSGROUND = new Action[] {
		Action.BACK_JUMP, Action.BACK_STEP, Action.CROUCH_A, Action.CROUCH_B,
		Action.CROUCH_FA, Action.CROUCH_FB, Action.CROUCH_GUARD, Action.DASH,
		Action.FOR_JUMP, Action.FORWARD_WALK, Action.JUMP, Action.NEUTRAL,
		Action.STAND_A, Action.STAND_B, Action.STAND_D_DB_BA, Action.STAND_D_DB_BB,
		Action.STAND_D_DF_FA, Action.STAND_D_DF_FB, Action.STAND_D_DF_FC,
		Action.STAND_F_D_DFA, Action.STAND_F_D_DFB, Action.STAND_FA,
		Action.STAND_FB, Action.STAND_GUARD, Action.THROW_A, Action.THROW_B };


	public static final Action[] ACTIONS = new Action[] {
			Action.AIR_A, Action.AIR_B, Action.AIR_D_DB_BA, Action.AIR_D_DB_BB,
			Action.AIR_D_DF_FA, Action.AIR_D_DF_FB, Action.AIR_DA, Action.AIR_DB,
			Action.AIR_F_D_DFA, Action.AIR_F_D_DFB, Action.AIR_FA, Action.AIR_FB,
			Action.AIR_UA, Action.AIR_UB,
			Action.BACK_JUMP, Action.BACK_STEP, Action.CROUCH_A, Action.CROUCH_B, Action.CROUCH_FA, Action.CROUCH_FB,
			Action.CROUCH_GUARD, Action.DASH, Action.FOR_JUMP, Action.FORWARD_WALK, Action.JUMP, Action.NEUTRAL,
			Action.STAND_A, Action.STAND_B, Action.STAND_D_DB_BA, Action.STAND_D_DB_BB, Action.STAND_D_DF_FA,
			Action.STAND_D_DF_FB, Action.STAND_D_DF_FC, Action.STAND_F_D_DFA, Action.STAND_F_D_DFB, Action.STAND_FA,
			Action.STAND_FB, Action.STAND_GUARD, Action.THROW_A, Action.THROW_B };

	public NDArray nn_predict(NDArray obs) {
		int n_layers = this.nnWeights.size() / 2;
		NDArray h = obs;
		NDArray y;
		NDArray weight;
		NDArray bias;
		for (int i = 0; i < n_layers - 1; i++) {
			weight = this.nnWeights.get(i * 2);
			bias = this.nnWeights.get(i * 2 + 1);

			y = np.dot(h, weight).add(bias);
			h = np.maximum(y, 0.0);
		}
		weight = this.nnWeights.get((n_layers - 1) * 2);
		bias = this.nnWeights.get((n_layers - 1) * 2 + 1);
		y = np.dot(h, weight).add(bias);

		//向量转成1维
		y = y.get(0);
		return y;
	}

	public int mask_cal(NDArray y, Vector<Boolean> ownMasks) {
		//将不可能向量降到0
		for (int i = 0; i < ownMasks.size(); i++) {
			if (ownMasks.get(i) == false) {
				y.set(-Float.MAX_VALUE, i);
			}
		}
//		System.out.println(new Float(-Float.MIN_VALUE).toString());

		String s = "";
		for (int i = 0; i < ownMasks.size(); i++) {
//			if (ownMasks.get(i) == false) {
//				s += new Double(0).toString() + ",";
//			} else {
//				s += new Double(y.asDouble(i)).toString() + ",";
//			}
			s += new Double(y.asDouble(i)).toString() + ",";
		}

		// softmax
		y = np.exp(y).divide(np.sum(np.exp(y), 0));
		// no softmax
//		y = y.divide(np.sum(y, 0));

		String s1 = "";
		for (int i = 0; i < ownMasks.size(); i++) {
//			if (ownMasks.get(i) == false) {
//				s1 += new Double(0).toString() + ",";
//			} else {
//				s1 += new Double(y.asDouble(i)).toString() + ",";
//			}
			s1 += new Double(y.asDouble(i)).toString() + ",";
		}

		this.softmaxBeforeLog.add(s);
		this.softmaxAfterLog.add(s1);


//		System.out.println("EmcmAiMax action probability: " + s);


		// random select
		java.util.Random random = new java.util.Random(System.currentTimeMillis());
		double prob = random.nextDouble();
		double start_prob = 0;
		int sel_act = 0;
		for (int i = 0; i < ownMasks.size(); i++) {
			start_prob = start_prob + y.asDouble(i);
			if (start_prob > prob) {
				sel_act = i;
				break;
			}
		}

		return sel_act;
	}

	public int get_policy(NDArray obs, Vector<Boolean> ownMasks) {
		int n_layers = this.nnWeights.size() / 2;
		NDArray h = obs;
		NDArray y;
		NDArray weight;
		NDArray bias;
		for (int i = 0; i < n_layers - 1; i++) {
			weight = this.nnWeights.get(i * 2);
			bias = this.nnWeights.get(i * 2 + 1);

			y = np.dot(h, weight).add(bias);
			h = np.maximum(y, 0.0);
		}
		weight = this.nnWeights.get((n_layers - 1) * 2);
		bias = this.nnWeights.get((n_layers - 1) * 2 + 1);
		y = np.dot(h, weight).add(bias);

		//向量转成1维
		y = y.get(0);

		//将不可能向量降到0
		for (int i = 0; i < ownMasks.size(); i++) {
			if (ownMasks.get(i) == false) {
				y.set(-Float.MAX_VALUE, i);
			}
		}
//		System.out.println(new Float(-Float.MIN_VALUE).toString());

		String s = "";
		for (int i = 0; i < ownMasks.size(); i++) {
//			if (ownMasks.get(i) == false) {
//				s += new Double(0).toString() + ",";
//			} else {
//				s += new Double(y.asDouble(i)).toString() + ",";
//			}
			s += new Double(y.asDouble(i)).toString() + ",";
		}

		// softmax
		y = np.exp(y).divide(np.sum(np.exp(y), 0));
		// no softmax
//		y = y.divide(np.sum(y, 0));

		String s1 = "";
		for (int i = 0; i < ownMasks.size(); i++) {
//			if (ownMasks.get(i) == false) {
//				s1 += new Double(0).toString() + ",";
//			} else {
//				s1 += new Double(y.asDouble(i)).toString() + ",";
//			}
			s1 += new Double(y.asDouble(i)).toString() + ",";
		}

		this.softmaxBeforeLog.add(s);
		this.softmaxAfterLog.add(s1);


//		System.out.println("EmcmAiMax action probability: " + s);


		// random select
		java.util.Random random = new java.util.Random(System.currentTimeMillis());
		double prob = random.nextDouble();
		double start_prob = 0;
		int sel_act = 0;
		for (int i = 0; i < ownMasks.size(); i++) {
			start_prob = start_prob + y.asDouble(i);
			if (start_prob > prob) {
				sel_act = i;
				break;
			}
		}

		return sel_act;
	}
	public void log() {
		String softmaxStr = "";
		// 加入动作名称
		for (Action a : ACTIONS) {
			softmaxStr += a.name() + ",";
		}
		softmaxStr += "\n";

		for (int i = 0; i < softmaxBeforeLog.size(); i++) {
			softmaxStr += softmaxBeforeLog.get(i);
			softmaxStr += "\n";
			softmaxStr += softmaxAfterLog.get(i);
			softmaxStr += "\n";
		}
//		for (String s : softmaxBeforeLog) {
//			softmaxBeforeStr += s;
//			softmaxBeforeStr += "\n";
//		}
//		for (String s : softmaxAfterLog) {
//			softmaxAfterStr += s;
//			softmaxAfterStr += "\n";
//		}
		String fileName = new Long(new Date().getTime()).toString();
		// /root/deep_learning/lrq/emcmai/
		saveFile("/home/llt/", fileName + ".csv", softmaxStr); // /Users/zhongqian/Desktop/FTG/FTG4.50_rlhomework/
		softmaxBeforeLog.clear();
		softmaxAfterLog.clear();
	}

	public void saveFile(String path, String fileName, String text) {
		BufferedWriter writer = null;
		File file = new File(path + fileName);
		//如果文件不存在，则新建一个
		if (!file.exists()) {
			try {
				file.createNewFile();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		//写入
		try {
			writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file, false), "UTF-8"));
			writer.write(text);
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				if (writer != null) {
					writer.close();
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		System.out.println("文件写入成功！");
	}

}
