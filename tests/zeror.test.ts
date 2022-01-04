import * as tf from '@tensorflow/tfjs';

import { ZeroRule } from '../src/zeror';
import * as classificationData from '../src/data/classification-data';
import * as regressionData from '../src/data/regression-data';

describe('ZeroRule classification test', (): void => {
  test('predict', async (): Promise<void> => {
    const [xTrain, yTrain, xTest, yTest] = classificationData.getIrisData(0.15);
    const model: ZeroRule = new ZeroRule();
    model.fit(null, yTrain);
    const result = model.predict(xTest);
    expect(result.shape[1]).toBe(3);
    const accracy = (
      await tf.metrics.categoricalAccuracy(yTest, result).mean().buffer()
    ).get(0);
    console.log(`Accuracy = ${accracy}`);
    const precision = (
      await tf.metrics.precision(yTest, result).mean().buffer()
    ).get(0);
    console.log(`Precision = ${precision}`);
    const recall = (await tf.metrics.recall(yTest, result).mean().buffer()).get(
      0
    );
    console.log(`Recall = ${recall}`);
  });

  test('toString', async (): Promise<void> => {
    const [xTrain, yTrain, xTest, yTest] = classificationData.getIrisData(0.15);
    const model: ZeroRule = new ZeroRule();
    model.fit(null, yTrain);
    const result = model.toString();
    expect(result).toContain(model.r?.toString());
    console.log(result);
  });
});

describe('ZeroRule regression test', (): void => {
  test('predict', async (): Promise<void> => {
    const trueCoefficients = { a: -0.8, b: -0.2, c: 0.9, d: 0.5 };
    const [xTrain, yTrain, xTest, yTest] = regressionData.getPolynomialData(
      100,
      trueCoefficients,
      0.15
    );
    const model: ZeroRule = new ZeroRule();
    model.fit(null, yTrain);
    const result = model.predict(xTest);
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
    const [xTrain, yTrain, xTest, yTest] = classificationData.getIrisData(0.15);
    const model: ZeroRule = new ZeroRule();
    model.fit(null, yTrain);
    const result = model.toString();
    expect(result).toContain(model.r?.toString());
    console.log(result);
  });
});
