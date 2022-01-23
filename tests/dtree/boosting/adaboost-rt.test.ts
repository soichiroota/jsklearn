import * as tf from '@tensorflow/tfjs';

import * as regressionData from '../../../src/data/regression-data';
import { AdaBoostRT } from '../../../src/dtree/boosting/adaboost-rt';

describe('AdaBoostRT regression test', (): void => {
  const trueCoefficients = { a: -0.8, b: -0.2, c: 0.9, d: 0.5 };
  const [xTrain, yTrain, xTest, yTest] = regressionData.getPolynomialData(
    100,
    trueCoefficients,
    0.15
  );

  test('predict', async (): Promise<void> => {
    const model: AdaBoostRT = new AdaBoostRT(0.01, 2, 5);
    await model.fit(xTrain, yTrain);
    const result = await model.predict(xTest);
    expect(result.shape[1]).toBe(1);
    const accuracy = (
      await tf.metrics.categoricalAccuracy(yTest, result).mean().buffer()
    ).get(0);
    console.log(`Accuracy = ${accuracy}`);
  });

  test('toString', async (): Promise<void> => {
    const model: AdaBoostRT = new AdaBoostRT(0.01, 2, 6);
    await model.fit(xTrain, yTrain);
    const result = await model.toString();
    expect(result).toContain('Tensor');
    console.log(result);
  });
});
