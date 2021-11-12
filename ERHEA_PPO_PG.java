import aiinterface.AIInterface;
import aiinterface.CommandCenter;

import enumValue.CharName;
import enumValue.GameMode;
import enumerate.Action;
import struct.FrameData;
import struct.GameData;
// modify by init.sh
import struct.Key;
import struct.ScreenData;

import org.apache.log4j.BasicConfigurator;

import AI.BaseAI;

public class ERHEA_PPO_PG implements AIInterface {
	private Key inputKey;
	private boolean player;
	private FrameData frameData;
	private CommandCenter cc;
	private GameData gd;

	// game Character Name
	private CharName charName;

	// RHEA AI
	BaseAI ai;

	private float win1 = 0, win2 = 0;
	private float win = 0;
	private int round = 0;
	private int frozenFrames = 0;

	// Set AI mode
	public void setAIMode() {
		System.out.println("Call First Get Information");
		// CommandCenter c = new CommandCenter();
		// c.setFrameData(frameData, player);

		String lossType = "pi"; // pi, sl
		boolean specificEnemy = true;

		// Just Set Game Node hold on.
		ai = new BaseAI(charName, gd, player, lossType, specificEnemy);

		// load AI model
		ai.gi.dlOpModel.loadModel();

	}

	@Override
	public void close() {
		System.gc();
		System.out.println("Close AI!");
	}

	@Override
	public void getInformation(FrameData fd, boolean control) {
		this.frameData = fd;
		ai.control = control;
		ai.getInformation(this.frameData);
		this.cc.setFrameData(frameData, this.player);
		ai.gi.isEmcmAiMaxFlag = false;
		ai.gi.isUseEmcmAiMaxFlag = false;
		timeFlag = true;

	}

	@Override
	public int initialize(GameData gd, boolean player) {
		System.out.println("Initialize!");
		BasicConfigurator.configure();
		this.inputKey = new Key();
		this.player = player;
		this.frameData = new FrameData();
		this.cc = new CommandCenter();
		this.gd = gd;
		String s_charName = this.gd.getCharacterName(this.player);
		if (s_charName.equals("ZEN"))
			charName = CharName.ZEN;
		else if (s_charName.equals("GARNET"))
			charName = CharName.GARNET;
		else
			charName = CharName.OTHER;

		// Frozen Frames
		this.frozenFrames = 0;

		// Set AI Mode
		setAIMode();

		return 0;
	}

	@Override
	public Key input() {
		return inputKey;
	}

	@Override
	public void processing() {

		// can process
		if (!frameData.getEmptyFlag() && frameData.getRemainingFramesNumber() > 0) {

			// keep do skill
			if (cc.getSkillFlag()) {
				inputKey = cc.getSkillKey();
			} else {
				inputKey.empty();
				cc.skillCancel();
				String key = ai.getDoAction();
				if (key != null) {
					cc.commandCall(key);
					if (ai.gi.isUseEmcmAiMaxFlag)
						inputKey = cc.getSkillKey();
				}
			}
			if (ai.use_ppo && ai.gi.isEmcmAiMaxFlag == false) {
				ai.gi.isEmcmAiMaxFlag = true;
				ai.gi.emcmAiMax.processing(false);
			}
			if (timeFlag) {
				timeFlag = false;
				long t = (System.nanoTime() - ai.start);
				if (t > 16500000) {
					delayTime++;
					avgTime += t;
					// System.out.println("remain time of ERHEA:" + new Double((t) /
					// 1000).toString());
				}
				allTime++;
				ai.gi.AllEvolveTimes += ai.gi.evolveTimes;
			}
		}
	}

	long avgTime = 0;
	int delayTime = 0;
	int allTime = 0;
	// int processingTime = 0;
	boolean timeFlag = true;

	@Override
	public void roundEnd(int p1_hp, int p2_hp, int frames) {
		System.out.println("ERHEA_PI_DJL avgtime: " + new Long(avgTime / (delayTime + 1)).toString());
		System.out.println("ERHEA_PI_DJL delayTime size: " + delayTime);
		System.out.println("ERHEA_PI_DJL time size: " + allTime);
		System.out.println("ERHEA_PI_DJL ERHEA size: " + ai.gi.processingTime);
		System.out.println("ERHEA_PI_DJL avg ai.gi.evolveTimes: " + ai.gi.AllEvolveTimes / (ai.gi.processingTime + 1));
		long startTime = System.nanoTime();
		ai.gi.AllEvolveTimes = 0;
		allTime = 0;
		delayTime = 0;
		avgTime = 0;
		ai.gi.processingTime = 0;

		round = round + 1;
		System.out.println("Round End!");
		float win_signal = 1;
		if (this.player && p1_hp < p2_hp) {
			win_signal = -1;
		} else if (!this.player && p1_hp > p2_hp) {
			win_signal = -1;
		}
		// 暴力切换器 zhongqian TODO 0720
		if (ai.SpeedModeFlag == false && win_signal == -1) {
			ai.use_ppo = !ai.use_ppo;
		}

		// train it.
		ai.gi.dlOpModel.train_batch(win_signal);
		ai.gi.dlOpModel.saveModel();

		// reset info
		ai.gi.resetInfo();

		// print statistics
		win1 = p1_hp > p2_hp ? win1 + 1 : win1;
		win2 = p1_hp < p2_hp ? win2 + 1 : win2;

		win = win1 + win2;
		if (win != 0) {
			win = this.player ? win1 / win : win2 / win;
		}
		String winstr = ("Round:\t" + round + "\t" + (win_signal > 0 ? "WIN" : "LOSE"));
		System.err.println("I'm:\t" + this.gd.getAiName(this.player));
		System.err.println("Opp:\t" + this.gd.getAiName(!this.player));
		System.err.println("Chara:\t" + this.gd.getCharacterName(this.player));
		System.err.println(winstr + "\twinrate\t" + win);

		boolean iswim = (this.player && p1_hp > p2_hp && p2_hp == 0) || (!this.player && p1_hp < p2_hp && p1_hp == 0);
		frames = (iswim) ? frames : 4200;
		System.err.println("Framenumber: " + frames);

		System.err.flush();

		// round end clean
		this.inputKey.empty();
		this.cc.skillCancel();
		System.out.println("全体の処理時間: " + (System.nanoTime() - startTime) + "ナノ秒");

		System.gc();

	}

	// modify by init.sh
	@Override
	public void getInformation(FrameData frameData) {

	}

	@Override
	public void getScreenData(ScreenData sd) {
		AIInterface.super.getScreenData(sd);
	}

}
