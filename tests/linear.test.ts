import * as tf from '@tensorflow/tfjs';

import { Linear } from '../src/linear';
import * as regressionData from '../src/data/regression-data';

describe('Linear regression test', (): void => {
  const trueCoefficients = { a: -0.8, b: -0.2, c: 0.9, d: 0.5 };
  const [xTrain, yTrain, xTest, yTest] = regressionData.getPolynomialData(
    100,
    trueCoefficients,
    0.15
  );
  test('predict', async (): Promise<void> => {
    const model: Linear = new Linear(1);
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
    const model: Linear = new Linear(1);
    await model.fit(xTrain, yTrain);
    const result = model.toString();
    expect(result).toContain('Tensor');
    console.log(result);
  });
});
