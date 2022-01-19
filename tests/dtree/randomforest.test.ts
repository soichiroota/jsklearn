import * as tf from '@tensorflow/tfjs';

import { PrunedTree } from '../../src/dtree/pruning';
import * as entropy from '../../src/dtree/entropy';
import { Linear } from '../../src/linear';
import * as classificationData from '../../src/data/classification-data';
import * as regressionData from '../../src/data/regression-data';
import { ZeroRule } from '../../src/zeror';
import { Bagging } from '../../src/dtree/bagging';
import { RandomTree } from '../../src/dtree/randomforest';

describe('RandomTree classification test', (): void => {
  const [xTrain, yTrain, xTest, yTest] = classificationData.getIrisData(0.15);

  test('predict', async (): Promise<void> => {
    const model: Bagging = new Bagging(2, 1.0, RandomTree, {
      features: 5,
      maxDepth: 2,
      metric: entropy.gini,
      leaf: ZeroRule,
    });
    await model.fit(xTrain, yTrain);
    const result = await model.predict(xTest);
    expect(result.shape[1]).toBe(3);
    const accuracy = (
      await tf.metrics.categoricalAccuracy(yTest, result).mean().buffer()
    ).get(0);
    console.log(`Accuracy = ${accuracy}`);
  });

  test('toString', async (): Promise<void> => {
    const model: Bagging = new Bagging(2, 1.0, RandomTree, {
      features: 5,
      maxDepth: 2,
      metric: entropy.gini,
      leaf: ZeroRule,
    });
    await model.fit(xTrain, yTrain);
    const result = model.toString();
    expect(result).toContain('Tensor');
    console.log(result);
  });
});

describe('RandomTree regression test', (): void => {
  const trueCoefficients = { a: -0.8, b: -0.2, c: 0.9, d: 0.5 };
  const [xTrain, yTrain, xTest, yTest] = regressionData.getPolynomialData(
    100,
    trueCoefficients,
    0.15
  );

  test('predict', async (): Promise<void> => {
    const model: Bagging = new Bagging(2, 1.0, RandomTree, {
      features: 5,
      maxDepth: 2,
      metric: entropy.deviation,
      leaf: Linear,
    });
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
    const model: Bagging = new Bagging(2, 1.0, RandomTree, {
      features: 5,
      maxDepth: 2,
      metric: entropy.deviation,
      leaf: Linear,
    });
    await model.fit(xTrain, yTrain);
    const result = model.toString();
    expect(result).toContain('Tensor');
    console.log(result);
  });
});
