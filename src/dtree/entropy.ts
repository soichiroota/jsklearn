import * as tf from '@tensorflow/tfjs';

export async function deviation(y: tf.Tensor<tf.Rank>): Promise<number> {
  return (await tf.moments(y).variance.sqrt().buffer()).get(0);
}

export async function gini(y: tf.Tensor<tf.Rank>): Promise<number> {
  const m = y.sum(0);
  const size = y.shape[0];
  const e = m.div(size).pow(2);
  return (await tf.sum(e).mul(-1).add(1).buffer()).get(0);
}

export async function infgain(y: tf.Tensor<tf.Rank>): Promise<number> {
  const m = y.sum(0);
  const size = y.shape[0];
  const e = tf
    .log(m.div(size))
    .div(tf.log(2))
    .mul(m)
    .div(size)
    .where(m.notEqual(tf.zeros(m.shape)), tf.zeros(m.shape));
  return (await tf.sum(e).mul(-1).buffer()).get(0);
}
