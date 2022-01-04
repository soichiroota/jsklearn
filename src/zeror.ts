import * as tf from '@tensorflow/tfjs';

export class ZeroRule {
  r: tf.Tensor<tf.Rank> | null;

  constructor() {
    this.r = null;
  }

  fit(
    x: tf.Tensor<tf.Rank> | tf.TensorLike | null,
    y: tf.Tensor<tf.Rank> | tf.TensorLike
  ): ZeroRule {
    this.r = tf.mean(y, 0);
    return this;
  }

  predict(x: tf.Tensor<tf.Rank>): tf.Tensor<tf.Rank> {
    if (this.r) {
      const z = tf.zeros([x.shape[0], this.r.shape[0]]);
      return z.add(this.r);
    }
    throw 'Property r is invalid.';
  }

  toString(): string {
    if (this.r) {
      return this.r.toString();
    }
    return 'null';
  }
}
