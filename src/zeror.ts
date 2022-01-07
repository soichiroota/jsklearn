import * as tf from '@tensorflow/tfjs';

import { BaseEstimator } from './base';

export class ZeroRule extends BaseEstimator {
  r: tf.Tensor<tf.Rank> | null;

  constructor() {
    super();
    this.r = null;
  }

  async fit(
    x: tf.Tensor<tf.Rank> | null,
    y: tf.Tensor<tf.Rank>
  ): Promise<ZeroRule> {
    this.r = tf.mean(y, 0);
    return this;
  }

  async predict(x: tf.Tensor<tf.Rank>): Promise<tf.Tensor<tf.Rank>> {
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
