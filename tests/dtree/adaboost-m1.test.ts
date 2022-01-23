import * as tf from '@tensorflow/tfjs';

import * as classificationData from '../../src/data/classification-data';
import { AdaBoostM1 } from '../../src/dtree/boosting/adaboost-m1';

describe('AdaBoostM1 classification test', (): void => {
  const [xTrain, yTrain, xTest, yTest] = classificationData.getIrisData(0.15);

  test('predict', async (): Promise<void> => {
    const model: AdaBoostM1 = new AdaBoostM1(2, 2);
    await model.fit(xTrain, yTrain);
    const result = await model.predict(xTest);
    expect(result.shape[1]).toBe(3);
    const accuracy = (
      await tf.metrics.categoricalAccuracy(yTest, result).mean().buffer()
    ).get(0);
    console.log(`Accuracy = ${accuracy}`);
  });

  test('toString', async (): Promise<void> => {
    const model: AdaBoostM1 = new AdaBoostM1(2, 9);
    await model.fit(xTrain, yTrain);
    const result = await model.toString();
    expect(result).toContain('Tensor');
    console.log(result);
  });
});
