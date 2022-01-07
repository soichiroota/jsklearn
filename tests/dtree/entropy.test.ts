import { deviation, gini, infgain } from '../../src/dtree/entropy';
import * as classificationData from '../../src/data/classification-data';
import * as regressionData from '../../src/data/regression-data';

describe('Entropy module test', (): void => {
  test('deviation', async (): Promise<void> => {
    const trueCoefficients = { a: -0.8, b: -0.2, c: 0.9, d: 0.5 };
    const [xTrain, yTrain, xTest, yTest] = regressionData.getPolynomialData(
      100,
      trueCoefficients,
      0.15
    );
    const result = await deviation(yTest);
    expect(result).toBeGreaterThan(-Infinity);
    console.log(`Deviation = ${result}`);
  });

  test('gini', async (): Promise<void> => {
    const [xTrain, yTrain, xTest, yTest] = classificationData.getIrisData(0.15);
    const result = await gini(yTest);
    expect(result).toBeGreaterThan(-Infinity);
    console.log(`Gini = ${result}`);
  });

  test('infgain', async (): Promise<void> => {
    const [xTrain, yTrain, xTest, yTest] = classificationData.getIrisData(0.15);
    const result = await infgain(yTest);
    expect(result).toBeGreaterThan(-Infinity);
    console.log(`Infgain = ${result}`);
  });
});
