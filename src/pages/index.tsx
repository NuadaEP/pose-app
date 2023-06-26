import { RefObject, useCallback, useEffect, useRef, useState } from "react";
import { Inter } from "next/font/google";

import Webcam from "react-webcam";

import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";

const inter = Inter({ subsets: ["latin"] });

type KeypointName =
  | "nose"
  | "left_eye"
  | "right_eye"
  | "left_ear"
  | "right_ear"
  | "left_shoulder"
  | "right_shoulder"
  | "left_elbow"
  | "right_elbow"
  | "left_wrist"
  | "right_wrist"
  | "left_hip"
  | "right_hip"
  | "left_knee"
  | "right_knee"
  | "left_ankle"
  | "right_ankle";

type Keypoint = {
  y: number;
  x: number;
  score: number;
  name: KeypointName;
};

type StandUp = {
  rightSide: {
    min: number;
    max: number;
  };
  leftSide: {
    min: number;
    max: number;
  };
};

export default function Home() {
  const cameraReference = useRef<Webcam>(null);

  const [isCapturing, setIsCapturing] = useState(false);
  const [pose, setPose] = useState<string>("Not Detected");

  const [rep, setRep] = useState(0);

  const [standUp, setStandUp] = useState<StandUp>({
    leftSide: {
      max: 0,
      min: 0,
    },
    rightSide: {
      max: 0,
      min: 0,
    },
  });

  const comparePositionsWithAccuracy = useCallback(
    (right: number, left: number, accuracy: number): boolean => {
      const rightMax = right * accuracy + right;
      const rightMin = right * accuracy - right;

      const leftMax = left * accuracy + left;
      const leftMin = left * accuracy - left;

      const isLeftAcceptable = left <= rightMax && left >= rightMin;
      const isRightAcceptable = right <= leftMax && right >= leftMin;

      return isLeftAcceptable && isRightAcceptable;
    },
    []
  );

  const estimatePoses = useCallback(async (ref: RefObject<any>) => {
    await tf.ready();

    const detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      {
        modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
        enableTracking: true,
      }
    );

    const poses = await detector.estimatePoses(ref);

    if (poses.length === 0) return undefined;

    return poses[0].keypoints.map(
      (pose): Keypoint => ({
        name: pose.name as KeypointName,
        score: pose.score as number,
        x: pose.x,
        y: pose.y,
      })
    );
  }, []);

  const isLowerLimbsAligned = useCallback(
    (moviment: Keypoint[]) => {
      const [rightAnkle, leftAnkle, rightKnee, leftKnee, rightHip, leftHip] = [
        moviment.find(({ name }) => name === "right_ankle")?.y as number,
        moviment.find(({ name }) => name === "left_ankle")?.y as number,
        moviment.find(({ name }) => name === "right_knee")?.y as number,
        moviment.find(({ name }) => name === "right_knee")?.y as number,
        moviment.find(({ name }) => name === "right_hip")?.y as number,
        moviment.find(({ name }) => name === "left_hip")?.y as number,
      ];

      const isAnkleAcceptable = comparePositionsWithAccuracy(
        rightAnkle,
        leftAnkle,
        0.2
      );

      const isKneeAcceptable = comparePositionsWithAccuracy(
        rightKnee,
        leftKnee,
        0.3
      );

      const isHipAcceptable = comparePositionsWithAccuracy(
        rightHip,
        leftHip,
        0.3
      );

      return isAnkleAcceptable && isKneeAcceptable && isHipAcceptable;
    },
    [comparePositionsWithAccuracy]
  );

  useEffect(() => {
    setTimeout(async () => {
      const standUpAvarege: { right: number[]; left: number[] } = {
        right: [],
        left: [],
      };

      for (let index = 0; index <= 2; index++) {
        const moviment = await estimatePoses(cameraReference.current?.video);

        if (moviment) {
          const lowerLimbs = isLowerLimbsAligned(moviment);

          if (lowerLimbs) {
            const [rightAnkle, leftAnkle, rightHip, leftHip] = [
              moviment.find(({ name }) => name === "right_ankle")?.y as number,
              moviment.find(({ name }) => name === "left_ankle")?.y as number,
              moviment.find(({ name }) => name === "right_hip")?.y as number,
              moviment.find(({ name }) => name === "left_hip")?.y as number,
            ];

            if (index < 2) {
              const isRightStandUp = rightAnkle - rightHip;
              const isLefttStandUp = leftAnkle - leftHip;

              standUpAvarege.right.push(isRightStandUp);
              standUpAvarege.left.push(isLefttStandUp);
            } else {
              const sumRight = standUpAvarege.right.reduce(
                (prev, current) => current + prev,
                0
              );

              const sumLeft = standUpAvarege.left.reduce(
                (prev, current) => current + prev,
                0
              );

              const rightAvarege = sumRight / standUpAvarege.right.length;
              const leftAvarege = sumLeft / standUpAvarege.left.length;
              setStandUp({
                leftSide: {
                  max: leftAvarege + leftAvarege * 0.1,
                  min: leftAvarege - leftAvarege * 0.1,
                },
                rightSide: {
                  max: rightAvarege + rightAvarege * 0.1,
                  min: rightAvarege - rightAvarege * 0.1,
                },
              });

              setPose("DETECTED!");
            }
          }
        }
      }
    }, 2000);
  }, [estimatePoses, isLowerLimbsAligned]);

  const isStadUp = useCallback(
    async (ref: RefObject<any>) => {
      const moviment = await estimatePoses(ref);

      if (!moviment) return false;

      const lowerLimbs = isLowerLimbsAligned(moviment);

      if (lowerLimbs) {
        const [rightAnkle, leftAnkle, rightHip, leftHip] = [
          moviment.find(({ name }) => name === "right_ankle")?.y as number,
          moviment.find(({ name }) => name === "left_ankle")?.y as number,
          moviment.find(({ name }) => name === "right_hip")?.y as number,
          moviment.find(({ name }) => name === "left_hip")?.y as number,
        ];

        const isRightStandUp = rightAnkle - rightHip;
        const isLefttStandUp = leftAnkle - leftHip;

        const isStandup =
          isRightStandUp >= standUp.rightSide.min &&
          isRightStandUp <= standUp.rightSide.max &&
          isLefttStandUp >= standUp.leftSide.min &&
          isLefttStandUp <= standUp.leftSide.max;

        return isStandup;
      }

      return false;
    },
    [estimatePoses, isLowerLimbsAligned, standUp]
  );

  const isSquat = useCallback(
    async (ref: RefObject<any>) => {
      const moviment = await estimatePoses(ref);

      if (!moviment) return false;

      const [rightKnee, leftKnee, rightHip, leftHip] = [
        moviment.find(({ name }) => name === "right_knee")?.y as number,
        moviment.find(({ name }) => name === "right_knee")?.y as number,
        moviment.find(({ name }) => name === "right_hip")?.y as number,
        moviment.find(({ name }) => name === "left_hip")?.y as number,
      ];

      const lowerLimbs = isLowerLimbsAligned(moviment);

      if (lowerLimbs) {
        const accuracy = 0.1;

        const isRightBreakParallel =
          rightHip >= rightKnee - rightKnee * accuracy;
        const isLeftBreakParallel = leftHip >= leftKnee - leftKnee * accuracy;

        const isBreakParallel = isRightBreakParallel && isLeftBreakParallel;

        return isBreakParallel;
      }

      return false;
    },
    [estimatePoses, isLowerLimbsAligned]
  );

  let commandList: Array<"airsquat" | "standup"> = [];

  const startRecord = useCallback(() => {
    setInterval(async () => {
      setIsCapturing(true);

      await tf.ready();

      if (commandList.toString() === ["airsquat", "standup"].toString()) {
        commandList = [];
        setRep((prev) => (prev += 1));
      }

      if (commandList[0] === "standup") commandList = [];

      if (commandList.length === 0) {
        const squat = await isSquat(cameraReference.current?.video);

        if (squat && !commandList.includes("airsquat")) {
          setPose("Squat");
          commandList.push("airsquat");
        }
      } else if (
        commandList.length !== 0 &&
        commandList.toString() === ["airsquat"].toString()
      ) {
        const standup = await isStadUp(cameraReference.current?.video);

        if (standup && !commandList.includes("standup")) {
          setPose("Stand up");
          commandList.push("standup");
        }
      }
    }, 200);
  }, [isSquat, isStadUp]);

  const stopRecord = useCallback(() => {
    setIsCapturing(false);
  }, []);

  const videoConstraints = {
    width: 420,
    height: 420,
    facingMode: "user",
  };

  return (
    <main
      className={`flex min-h-screen flex-col items-center justify-between p-24 ${inter.className}`}
    >
      <h1 style={{ color: "green" }}>
        {pose} AND {rep}
      </h1>

      <Webcam
        ref={cameraReference}
        height={400}
        width={600}
        videoConstraints={videoConstraints}
      />

      {isCapturing ? (
        <button onClick={stopRecord}>Stop Capture</button>
      ) : (
        <button onClick={startRecord} disabled={pose !== "DETECTED!"}>
          {pose === "DETECTED!" ? "Start Capture" : ""}
        </button>
      )}
    </main>
  );
}
