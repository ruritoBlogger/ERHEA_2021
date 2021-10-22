package NN;

import aiinterface.AIInterface;
import aiinterface.CommandCenter;
import NN.centipede.numpy.NDArray;
import NN.centipede.numpy.Numpy.np;
import enumerate.Action;
import enumerate.State;
import simulator.Simulator;
import struct.CharacterData;
import struct.*;

import java.util.*;

/**
 * @author Joshua
 */
public class EmcmAiMaxMinusOne implements AIInterface {

	private String PATH = "./data/aiData/EmcmAiMaxMinusOne/";
	private Key inputKey = new Key();
	private FrameData frameData;
	private CommandCenter commandCenter;
	private GameData gameData;
	private boolean player;
	private Simulator simulator;

	private Vector<Integer> commandsInDelaysCurrentFrame;
	private Vector<Action> commandsInDelaysAction;

	private int frozenFrames;
	private ArrayList<MotionData> ownMotions;
	private Vector<Integer> ownMotionsEnergy;
	private boolean isControl;
	private Vector<NDArray> frameObsDeque;
	private int delayedFrame;
	private int currentFrame;
	private Vector<Integer> oppMotionsEnergy;
	private int remainingTimeMilliseconds;
	private long startTime;

	// 这个是调整整个框架是否使用框架的代码, 以及选择模式
	public NN.NoUseframe.AINN AINN;

	Vector<Boolean> ownMasks;
	NDArray ownObs;

	public EmcmAiMaxMinusOne() {
	}
	
	private int[] ACTION_MASK = new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	private int actionsSize = ACTION_MASK.length;

	private static final int[] AIR_ACTION_MASK = new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	private static final int[] GROUND_ACTION_MASK = new int[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	private NDArray getObs() {
		LinkedList<NDArray> obsesList = new LinkedList<NDArray>();
		obsesList.add(this.frameObsDeque.get(24 - 1));
		obsesList.add(this.frameObsDeque.get(12));
		obsesList.add(this.frameObsDeque.get(0));
		NDArray obs = null;
		for (int i = 0; i < obsesList.size(); i++) {
			if (i == 0) {
				obs = obsesList.get(0);
			} else {
				obs = np.concatenate(obs, obsesList.get(i));
			}
		}
		return obs;
	}

	public NDArray getFrameObs(FrameData frameData, boolean player) {
		CharacterData my = frameData.getCharacter(player);
		CharacterData opp = frameData.getCharacter(!player);

		// my information
		float myEnergy = (float) my.getEnergy() / 300;
		float myX = ((float) (my.getLeft() + my.getRight()) / 2 - 960 / 2) / (960 / 2);
		float myY = ((float) (my.getBottom() + my.getTop()) / 2) / 640;
		float mySpeedX = (float) my.getSpeedX() / 20;
		float mySpeedY = (float) my.getSpeedY() / 28;
		float myState = my.getState().ordinal();
		float myAction = my.getAction().ordinal();
		float myRemainingFrame = (float) my.getRemainingFrame() / 70;

		// opp information
		float oppEnergy = (float) opp.getEnergy() / 300;
		float oppX = ((float) (opp.getLeft() + opp.getRight()) / 2 - (float) (my.getLeft() + my.getRight()) / 2) / 960;
		float oppY = ((float) (opp.getBottom() + opp.getTop()) / 2) / 640;
		float oppSpeedX = (float) opp.getSpeedX() / 20;
		float oppSpeedY = (float) opp.getSpeedY() / 28;
		float oppState = opp.getState().ordinal();
		float oppAction = opp.getAction().ordinal();
		float oppRemainingFrame = (float) opp.getRemainingFrame() / 70;

		Vector<Float> observation = new Vector<Float>(144);

		// my information
		observation.add(myEnergy); //0,1
		observation.add(myX); //-1,1
		observation.add(myY); //0,1
		observation.add(mySpeedX); //-1,1
		observation.add(mySpeedY); //-1,1
		for (int i = 0; i < 4; i++) {
			if (new Float(i) == myState) {
				observation.add((float) 1.0);
			} else {
				observation.add((float) 0.0);
			}
		}
		for (int i = 0; i < 56; i++) {
			if (new Float(i) == myAction) {
				observation.add((float) 1.0);
			} else {
				observation.add((float) 0.0);
			}
		}
		observation.add(myRemainingFrame);

		// opp information
		observation.add(oppEnergy);
		observation.add(oppX);
		observation.add(oppY);
		observation.add(oppSpeedX);
		observation.add(oppSpeedY);
		for (int i = 0; i < 4; i++) {
			if (new Float(i) == oppState) {
				observation.add((float) 1.0);
			} else {
				observation.add((float) 0.0);
			}
		}
		for (int i = 0; i < 56; i++) {
			if (new Float(i) == oppAction) {
				observation.add((float) 1.0);
			} else {
				observation.add((float) 0.0);
			}
		}
		observation.add(oppRemainingFrame);

		Deque<AttackData> myProjectiles = new LinkedList<>();
		Deque<AttackData> oppProjectiles = new LinkedList<>();
		if (player) {
			myProjectiles = frameData.getProjectilesByP1();
			oppProjectiles = frameData.getProjectilesByP2();
		} else {
			myProjectiles = frameData.getProjectilesByP2();
			oppProjectiles = frameData.getProjectilesByP1();
		}

		float myHitDamage, myHitAreaNowX, myHitAreaNowY;
		if (myProjectiles.size() == 2) {
			myHitDamage = (float) ((float) myProjectiles.getFirst().getHitDamage() / 200.0);
			myHitAreaNowX = (float) (((float) (myProjectiles.getFirst().getCurrentHitArea().getLeft()
					+ myProjectiles.getFirst().getCurrentHitArea().getRight()) / 2
					- (float) (my.getLeft() + my.getRight()) / 2) / 960.0);
			myHitAreaNowY = (float) (((float) (myProjectiles.getFirst().getCurrentHitArea().getTop()
					+ myProjectiles.getFirst().getCurrentHitArea().getBottom()) / 2) / 640.0);
			observation.add(myHitDamage);
			observation.add(myHitAreaNowX);
			observation.add(myHitAreaNowY);
			myHitDamage = (float) ((float) myProjectiles.getLast().getHitDamage() / 200.0);
			myHitAreaNowX = (float) (((float) (myProjectiles.getLast().getCurrentHitArea().getLeft()
					+ myProjectiles.getLast().getCurrentHitArea().getRight()) / 2
					- (float) (my.getLeft() + my.getRight()) / 2) / 960.0);
			myHitAreaNowY = (float) (((float) (myProjectiles.getLast().getCurrentHitArea().getTop()
					+ myProjectiles.getLast().getCurrentHitArea().getBottom()) / 2) / 640.0);
			observation.add(myHitDamage);
			observation.add(myHitAreaNowX);
			observation.add(myHitAreaNowY);
		} else if (myProjectiles.size() == 1) {
			myHitDamage = (float) ((float) myProjectiles.getFirst().getHitDamage() / 200.0);
			myHitAreaNowX = (float) (((float) (myProjectiles.getFirst().getCurrentHitArea().getLeft()
					+ myProjectiles.getFirst().getCurrentHitArea().getRight()) / 2
					- (float) (my.getLeft() + my.getRight()) / 2) / 960.0);
			myHitAreaNowY = (float) (((float) (myProjectiles.getFirst().getCurrentHitArea().getTop()
					+ myProjectiles.getFirst().getCurrentHitArea().getBottom()) / 2) / 640.0);
			observation.add(myHitDamage);
			observation.add(myHitAreaNowX);
			observation.add(myHitAreaNowY);
			for (int i = 0; i < 3; i++) {
				observation.add((float) 0.0);
			}
		} else {
			for (int i = 0; i < 6; i++) {
				observation.add((float) 0.0);
			}
		}

		float oppHitDamage, oppHitAreaNowX, oppHitAreaNowY;
		if (oppProjectiles.size() == 2) {
			oppHitDamage = (float) ((float) oppProjectiles.getFirst().getHitDamage() / 200.0);
			oppHitAreaNowX = (float) (((float) (oppProjectiles.getFirst().getCurrentHitArea().getLeft()
					+ oppProjectiles.getFirst().getCurrentHitArea().getRight()) / 2
					- (float) (my.getLeft() + my.getRight()) / 2) / 960.0);
			oppHitAreaNowY = (float) (((float) (oppProjectiles.getFirst().getCurrentHitArea().getTop()
					+ oppProjectiles.getFirst().getCurrentHitArea().getBottom()) / 2) / 640.0);
			observation.add(oppHitDamage);
			observation.add(oppHitAreaNowX);
			observation.add(oppHitAreaNowY);
			oppHitDamage = (float) ((float) oppProjectiles.getLast().getHitDamage() / 200.0);
			oppHitAreaNowX = (float) (((float) (oppProjectiles.getLast().getCurrentHitArea().getLeft()
					+ oppProjectiles.getLast().getCurrentHitArea().getRight()) / 2
					- (float) (my.getLeft() + my.getRight()) / 2) / 960.0);
			oppHitAreaNowY = (float) (((float) (oppProjectiles.getLast().getCurrentHitArea().getTop()
					+ oppProjectiles.getLast().getCurrentHitArea().getBottom()) / 2) / 640.0);
			observation.add(oppHitDamage);
			observation.add(oppHitAreaNowX);
			observation.add(oppHitAreaNowY);
		} else if (oppProjectiles.size() == 1) {
			oppHitDamage = (float) ((float) oppProjectiles.getFirst().getHitDamage() / 200.0);
			oppHitAreaNowX = (float) (((float) (oppProjectiles.getFirst().getCurrentHitArea().getLeft()
					+ oppProjectiles.getFirst().getCurrentHitArea().getRight()) / 2
					- (float) (my.getLeft() + my.getRight()) / 2) / 960.0);
			oppHitAreaNowY = (float) (((float) (oppProjectiles.getFirst().getCurrentHitArea().getTop()
					+ oppProjectiles.getFirst().getCurrentHitArea().getBottom()) / 2) / 640.0);
			observation.add(oppHitDamage);
			observation.add(oppHitAreaNowX);
			observation.add(oppHitAreaNowY);
			for (int i = 0; i < 3; i++) {
				observation.add((float) 0.0);
			}
		} else {
			for (int i = 0; i < 6; i++) {
				observation.add((float) 0.0);
			}
		}

		NDArray obsArray = np.zeros(observation.size());
		for (int i = 0; i < observation.size(); i++) {
			float value = observation.get(i);
			// clip value
			value = Math.max(Math.min(1.f, value), -1.f);
			obsArray.set(value, i);
		}
		return obsArray;
	}

	public Vector<Boolean> maskEnergyAction(int energy, int[] ownMask, boolean player) {
		Vector<Integer> motionsEnergy;
		if (player == this.player)
			motionsEnergy = this.ownMotionsEnergy;
		else
			motionsEnergy = this.oppMotionsEnergy;
		Vector<Boolean> energyMask = new Vector<Boolean>();
		for (int i = 0; i < actionsSize; i++) {
			if (ownMask[i] == 0) {
				energyMask.add(false);
			} else if (motionsEnergy.get(i) + energy >= 0) {
				energyMask.add(true);
			} else {
				energyMask.add(false);
			}
		}
		return energyMask;
	}

	public FrameData predictCurrentFramedata() {
		// remove old issued commands
		while (this.commandsInDelaysAction.size() > 0 && this.commandsInDelaysCurrentFrame.get(0) < this.delayedFrame) {
			this.commandsInDelaysAction.remove(0);
			this.commandsInDelaysCurrentFrame.remove(0);
		}

		// simulate undelayed framedata with simulator and commands in delays
		FrameData simulateFrameData = this.frameData;
		Action simulateCommand = null;
		int simulateFrame = this.delayedFrame;
		ArrayDeque<Action> actions = null;
		for (int i = 0; i < this.commandsInDelaysAction.size(); i++) {
			int waiting_frame = this.commandsInDelaysCurrentFrame.get(i);
			Action waiting_command = this.commandsInDelaysAction.get(i);
			if (simulateCommand != null) {
				actions = new ArrayDeque<Action>();
				actions.add(simulateCommand);
			} else {
				actions = null;
			}
			simulateFrameData = this.simulator.simulate(simulateFrameData, this.player, actions, null,
					waiting_frame - simulateFrame);
			simulateFrame = waiting_frame;
			simulateCommand = waiting_command;
		}
		if (this.currentFrame > simulateFrame) {
			if (simulateCommand != null) {
				actions = new ArrayDeque<Action>();
				actions.add(simulateCommand);
			} else {
				actions = null;
			}
			simulateFrameData = this.simulator.simulate(simulateFrameData, this.player, actions, null,
					this.currentFrame - simulateFrame);
		}

		// check if simulate framedata is correct, if not just assume not commands in
		// delays
		if (simulateFrameData.getCharacter(this.player).isControl() != this.isControl) {
			simulateFrameData = this.simulator.simulate(this.frameData, this.player, null, null, 15);
		}
		return simulateFrameData;
	}

	@Override
	public void close() {
		// nothing,done
	}

	@Override
	public Key input() {
		System.out.println("this.frameData.getFramesNumber(): " + new Integer(this.frameData.getFramesNumber()).toString() + ". public Key input()");
		return this.inputKey;
	}

	@Override
	public void getInformation(FrameData arg0, boolean arg1) {
		this.startTime = System.nanoTime();
		this.frameData = arg0;
		this.isControl = arg1;
		this.commandCenter.setFrameData(arg0, this.player);
	}

	@Override
	public int initialize(GameData arg0, boolean arg1) {
		this.frameData = new FrameData();
		this.inputKey = new Key();
		this.commandCenter = new CommandCenter();
		this.gameData = arg0;
		this.player = arg1;
		this.simulator = this.gameData.getSimulator();
		this.commandsInDelaysCurrentFrame = new Vector<Integer>();
		this.commandsInDelaysAction = new Vector<Action>();
		this.frozenFrames = 0;
		this.frameObsDeque = new Vector<>();
		this.ownMotions = this.gameData.getMotionData(this.player);
		this.ownMotionsEnergy = new Vector<Integer>();

		AINN = new NN.NoUseframe.AINN(PATH, this.gameData.getCharacterName(this.player),
				"MctsAi".equals(this.gameData.getAiName(!this.player)) ? "2": "1");

		for (int i = 0; i < this.actionsSize; i++) {
			this.ownMotionsEnergy.add(this.ownMotions.get(AINN.ACTIONS[i].ordinal()).getAttackStartAddEnergy());
		}
		this.remainingTimeMilliseconds = 0;
		System.out.println("EmcmAiMaxMinusOne: public int initialize");
		return 0;
	}

	public boolean canProcessing() {
		return !this.frameData.getEmptyFlag() && this.frameData.getRemainingFramesNumber() > 0;
	}

	@Override
	public void processing() {
		//long beforeTime = new Date().getTime();

		try {
			if (!canProcessing())
				return;
			AllTime ++;
			// other round ? Yes: No
			if (this.frameData.getRemainingTimeMilliseconds() > this.remainingTimeMilliseconds) {
				clear();
			}
			this.remainingTimeMilliseconds = this.frameData.getRemainingTimeMilliseconds();
			
			this.delayedFrame = this.frameData.getFramesNumber();
			NDArray frameObs = getFrameObs(this.frameData, this.player);

			if (this.frameObsDeque.size() < 25) {
				int temp = 25 - this.frameObsDeque.size();
				for (int i = 0; i < temp; i++) {
					this.frameObsDeque.add(frameObs);
				}
			} else {
				this.frameObsDeque.remove(0);
				this.frameObsDeque.add(frameObs);
			}
			this.currentFrame = this.delayedFrame + 14;
			
			if (this.frameData.getRemainingTimeMilliseconds() <= 0)
				return;
			this.frozenFrames -= 1;
			if (this.commandCenter.getSkillFlag()) {
				this.inputKey = this.commandCenter.getSkillKey();
				return;
			}
			if (!this.isControl)
				return;
			if (this.frozenFrames > 0) {
				return;
			}
			this.inputKey.empty();
			this.commandCenter.skillCancel();

			FrameData currentFramedata = predictCurrentFramedata();
			
			boolean isOwnAir = currentFramedata.getCharacter(this.player).getState().equals(State.AIR);
			int[] ownMask = (isOwnAir) ? AIR_ACTION_MASK : GROUND_ACTION_MASK;
			this.ownMasks = maskEnergyAction(this.frameData.getCharacter(this.player).getEnergy(), ownMask, this.player);
			this.ownObs = getObs();
			Action action = AINN.ACTIONS[AINN.get_policy(this.ownObs, this.ownMasks)];
			String command = action.name();
			if ("NEUTRAL".equals(command)) {
				frozenFrames = 6;
			} else {
				frozenFrames = 0;
				if ("CROUCH_GUARD".equals(command)) {
					command = "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1";
				} else if ("STAND_GUARD".equals(command)) {
					command = "4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4";
				}
//				System.out.println(": " + command);
				commandCenter.commandCall(command);
				this.commandsInDelaysAction.add(action);
				this.commandsInDelaysCurrentFrame.add(this.currentFrame);
				this.inputKey = this.commandCenter.getSkillKey();
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("EmcmAiMaxMinusOne:ERROR");
		}
		double time = (double)(16500000 - (System.nanoTime() - this.startTime)) / 1000000.0; // ms
//		timeList.add(time);
		if (time < 0)
			delayTime += 1;

//		if (time < 0)
//			System.out.println("EmcmAiMaxMinusOne running time:" + time);
	}
	double delayTime = 0;
	int AllTime = 0;
	public void clear() {
		this.frozenFrames = 0;
		this.commandsInDelaysCurrentFrame = new Vector<Integer>();
		this.commandsInDelaysAction = new Vector<Action>();
		this.frameObsDeque = new Vector<NDArray>();
		this.inputKey.empty();
		this.commandCenter.skillCancel();
		System.gc();
	}

	@Override
	public void roundEnd(int arg0, int arg1, int arg2) {
		System.out.println("EmcmAiMaxMinusOne: public void roundEnd");
		System.out.println("EmcmAiMaxMinusOne delayTime: " + delayTime);
		System.out.println("EmcmAiMaxMinusOne AllTime: " + AllTime);
		delayTime = 0;
		AllTime = 0;
	}

}
