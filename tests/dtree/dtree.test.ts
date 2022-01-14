import * as tf from '@tensorflow/tfjs';

import { DecisionTree } from '../../src/dtree/dtree';
import * as entropy from '../../src/dtree/entropy';
import { Linear } from '../../src/linear';
import * as classificationData from '../../src/data/classification-data';
import * as regressionData from '../../src/data/regression-data';
import { ZeroRule } from '../../src/zeror';

describe('DecisionTree classification test', (): void => {
  const [xTrain, yTrain, xTest, yTest] = classificationData.getIrisData(0.15);

  test('predict with gini', async (): Promise<void> => {
    const model: DecisionTree = new DecisionTree(2, entropy.gini, ZeroRule);
    await model.fit(xTrain, yTrain);
    const result = await model.predict(xTest);
    expect(result.shape[1]).toBe(3);
    const accuracy = (
      await tf.metrics.categoricalAccuracy(yTest, result).mean().buffer()
    ).get(0);
    console.log(`Accuracy = ${accuracy}`);
  });

  test('toString with gini', async (): Promise<void> => {
    const model: DecisionTree = new DecisionTree(2, entropy.gini, ZeroRule);
    await model.fit(xTrain, yTrain);
    const result = model.toString();
    expect(result).toContain('Tensor');
    console.log(result);
  });

  test('splitTree with gini', async (): Promise<void> => {
    const model: DecisionTree = new DecisionTree(2, entropy.gini, ZeroRule);
    const [expectedLeft, expectedRight] = await model.splitTreeSlow(
      xTest,
      yTest
    );
    const [resultLeft, resultRight] = await model.splitTree(xTest, yTest);
    console.log(
      `expectedLeft = ${expectedLeft}, resultLeft = ${resultLeft}\nexpectedRight = ${expectedRight}, resultRight = ${resultRight}\n`
    );
    expect(await resultLeft.array()).toEqual(await expectedLeft.array());
    expect(await resultRight.array()).toEqual(await expectedRight.array());
  });

  test('predict with infgain', async (): Promise<void> => {
    const model: DecisionTree = new DecisionTree(2, entropy.infgain, ZeroRule);
    await model.fit(xTrain, yTrain);
    const result = await model.predict(xTest);
    expect(result.shape[1]).toBe(3);
    const accuracy = (
      await tf.metrics.categoricalAccuracy(yTest, result).mean().buffer()
    ).get(0);
    console.log(`Accuracy = ${accuracy}`);
  });

  test('toString with infgain', async (): Promise<void> => {
    const model: DecisionTree = new DecisionTree(2, entropy.infgain, ZeroRule);
    await model.fit(xTrain, yTrain);
    const result = model.toString();
    expect(result).toContain('Tensor');
    console.log(result);
  });

  test('splitTree with infgain', async (): Promise<void> => {
    const model: DecisionTree = new DecisionTree(2, entropy.infgain, ZeroRule);
    const [expectedLeft, expectedRight] = await model.splitTreeSlow(
      xTest,
      yTest
    );
    const [resultLeft, resultRight] = await model.splitTree(xTest, yTest);
    console.log(
      `expectedLeft = ${expectedLeft}, resultLeft = ${resultLeft}\nexpectedRight = ${expectedRight}, resultRight = ${resultRight}\n`
    );
    expect(await resultLeft.array()).toEqual(await expectedLeft.array());
    expect(await resultRight.array()).toEqual(await expectedRight.array());
  });
});

describe('DecisionTree regression test', (): void => {
  const trueCoefficients = { a: -0.8, b: -0.2, c: 0.9, d: 0.5 };
  const [xTrain, yTrain, xTest, yTest] = regressionData.getPolynomialData(
    100,
    trueCoefficients,
    0.15
  );

  test('predict', async (): Promise<void> => {
    const model: DecisionTree = new DecisionTree(2, entropy.deviation, Linear);
    await model.fit(xTrain, yTrain);
    const result = await model.predict(xTest);
    expect(result.shape[1]).toBe(1);
    const mse = (
      await tf.metrics.meanSquaredError(yTest, result).mean().buffer()
    ).get(0);
    console.log(`Mean Squared Error = ${mse}`);
    const mae = (
      await tf.metrics.meanAbsoluteError(yTest, result).mean().buffer()
    ).get(0);
    console.log(`Mean Absolute Error = ${mae}`);
  });

  test('toString', async (): Promise<void> => {
    const model: DecisionTree = new DecisionTree(2, entropy.deviation, Linear);
    await model.fit(xTrain, yTrain);
    const result = model.toString();
    expect(result).toContain('Tensor');
    console.log(result);
  });

  test('splitTree', async (): Promise<void> => {
    const model: DecisionTree = new DecisionTree(2, entropy.deviation, Linear);
    const [expectedLeft, expectedRight] = await model.splitTreeSlow(
      xTest,
      yTest
    );
    const [resultLeft, resultRight] = await model.splitTree(xTest, yTest);
    console.log(
      `expectedLeft = ${expectedLeft}, resultLeft = ${resultLeft}\nexpectedRight = ${expectedRight}, resultRight = ${resultRight}\n`
    );
    expect(await resultLeft.array()).toEqual(await expectedLeft.array());
    expect(await resultRight.array()).toEqual(await expectedRight.array());
  });
});
