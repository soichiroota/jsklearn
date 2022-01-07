import * as tf from '@tensorflow/tfjs';

import { DecisionStump } from '../../src/dtree/dstump';
import * as entropy from '../../src/dtree/entropy';
import { Linear } from '../../src/linear';
import * as classificationData from '../../src/data/classification-data';
import * as regressionData from '../../src/data/regression-data';

describe('DecisionStump classification test', (): void => {
  test('predict with gini', async (): Promise<void> => {
    const [xTrain, yTrain, xTest, yTest] = classificationData.getIrisData(0.15);
    const model: DecisionStump = new DecisionStump(entropy.gini);
    await model.fit(xTrain, yTrain);
    const result = await model.predict(xTest);
    expect(result.shape[1]).toBe(3);
    const accuracy = (
      await tf.metrics.categoricalAccuracy(yTest, result).mean().buffer()
    ).get(0);
    console.log(`Accuracy = ${accuracy}`);
  });

  test('toString with gini', async (): Promise<void> => {
    const [xTrain, yTrain, xTest, yTest] = classificationData.getIrisData(0.15);
    const model: DecisionStump = new DecisionStump(entropy.gini);
    await model.fit(xTrain, yTrain);
    const result = model.toString();
    expect(result).toContain('Tensor');
    console.log(result);
  });

  test('predict with infgain', async (): Promise<void> => {
    const [xTrain, yTrain, xTest, yTest] = classificationData.getIrisData(0.15);
    const model: DecisionStump = new DecisionStump(entropy.infgain);
    await model.fit(xTrain, yTrain);
    const result = await model.predict(xTest);
    expect(result.shape[1]).toBe(3);
    const accuracy = (
      await tf.metrics.categoricalAccuracy(yTest, result).mean().buffer()
    ).get(0);
    console.log(`Accuracy = ${accuracy}`);
  });

  test('toString with infgain', async (): Promise<void> => {
    const [xTrain, yTrain, xTest, yTest] = classificationData.getIrisData(0.15);
    const model: DecisionStump = new DecisionStump(entropy.infgain);
    await model.fit(xTrain, yTrain);
    const result = model.toString();
    expect(result).toContain('Tensor');
    console.log(result);
  });
});

describe('DecisionStump regression test', (): void => {
  test('predict', async (): Promise<void> => {
    const trueCoefficients = { a: -0.8, b: -0.2, c: 0.9, d: 0.5 };
    const [xTrain, yTrain, xTest, yTest] = regressionData.getPolynomialData(
      100,
      trueCoefficients,
      0.15
    );
    const model: DecisionStump = new DecisionStump(entropy.deviation, Linear);
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
    const trueCoefficients = { a: -0.8, b: -0.2, c: 0.9, d: 0.5 };
    const [xTrain, yTrain, xTest, yTest] = regressionData.getPolynomialData(
      100,
      trueCoefficients,
      0.15
    );
    const model: DecisionStump = new DecisionStump(entropy.deviation, Linear);
    await model.fit(xTrain, yTrain);
    const result = model.toString();
    expect(result).toContain('Tensor');
    console.log(result);
  });
});
