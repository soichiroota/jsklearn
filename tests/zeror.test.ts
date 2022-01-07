import * as tf from '@tensorflow/tfjs';

import { ZeroRule } from '../src/zeror';
import * as classificationData from '../src/data/classification-data';
import * as regressionData from '../src/data/regression-data';

describe('ZeroRule classification test', (): void => {
  test('predict', async (): Promise<void> => {
    const [xTrain, yTrain, xTest, yTest] = classificationData.getIrisData(0.15);
    const model: ZeroRule = new ZeroRule();
    await model.fit(null, yTrain);
    const result = await model.predict(xTest);
    expect(result?.shape[1]).toBe(3);
  });

  test('toString', async (): Promise<void> => {
    const [xTrain, yTrain, xTest, yTest] = classificationData.getIrisData(0.15);
    const model: ZeroRule = new ZeroRule();
    await model.fit(null, yTrain);
    const result = model.toString();
    expect(result).toBe(model.r?.toString());
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
    await model.fit(null, yTrain);
    const result = await model.predict(xTest);
    expect(result?.shape[1]).toBe(1);
  });

  test('toString', async (): Promise<void> => {
    const [xTrain, yTrain, xTest, yTest] = classificationData.getIrisData(0.15);
    const model: ZeroRule = new ZeroRule();
    await model.fit(null, yTrain);
    const result = model.toString();
    expect(result).toBe(model.r?.toString());
  });
});
