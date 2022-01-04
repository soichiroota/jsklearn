import * as tf from '@tensorflow/tfjs';

export function generateData(numPoints: number, coeff: any, sigma = 0.04): any {
  return tf.tidy(() => {
    const [a, b, c, d] = [
      tf.scalar(coeff.a),
      tf.scalar(coeff.b),
      tf.scalar(coeff.c),
      tf.scalar(coeff.d),
    ];

    const xs = tf.randomUniform([numPoints], -1, 1);

    // Generate polynomial data
    const three = tf.scalar(3, 'int32');
    const ys = a
      .mul(xs.pow(three))
      .add(b.mul(xs.square()))
      .add(c.mul(xs))
      .add(d)
      // Add random noise to the generated data
      // to make the problem a bit more interesting
      .add(tf.randomNormal([numPoints], 0, sigma));

    // Normalize the y values to the range 0 to 1.
    const ymin = ys.min();
    const ymax = ys.max();
    const yrange = ymax.sub(ymin);
    const ysNormalized = ys.sub(ymin).div(yrange);

    return {
      xs,
      ys: ysNormalized,
    };
  });
}

export function getPolynomialData(
  numPoints: number,
  coeff: any,
  testSplit: number,
  sigma = 0.04
): tf.Tensor<tf.Rank>[] {
  const numTestExamples = Math.round(numPoints * testSplit);
  const numTrainExamples = numPoints - numTestExamples;
  const trainDataset = generateData(numTrainExamples, coeff, sigma);
  const testDataset = generateData(numTestExamples, coeff, sigma);
  return [trainDataset.xs, trainDataset.ys, testDataset.xs, testDataset.ys];
}
