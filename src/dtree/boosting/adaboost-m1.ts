import * as tf from '@tensorflow/tfjs';

import { BaseEstimator } from '../../base';
import { WeightedDecisionTree, WeightedZeroRule, wGini } from './weighted';

export class AdaBoostM1 extends BaseEstimator {
  boost: number;
  maxDepth: number;
  trees: WeightedDecisionTree[];
  beta: tf.Tensor<tf.Rank> | null;
  nClz: number;

  constructor(boost = 5, maxDepth = 5) {
    super();
    this.boost = boost;
    this.maxDepth = maxDepth;
    this.trees = [];
    this.beta = null;
    this.nClz = 0;
  }

  async fit(x: tf.Tensor<tf.Rank>, y: tf.Tensor<tf.Rank>): Promise<AdaBoostM1> {
    this.trees = [];
    this.beta = tf.zeros([this.boost]);
    this.nClz = (y as tf.Tensor2D).shape[1];
    let weights = tf.ones([x.shape[0]]).div(x.shape[0]);
    for (const i of [...Array(this.boost).keys()]) {
      let tree = new WeightedDecisionTree(
        this.maxDepth,
        wGini,
        WeightedZeroRule
      );
      tree = await tree.fit(x, y, weights);
      const z = await tree.predict(x);
      const falseFilter = z.argMax(1).notEqual(y.argMax(1));
      const err = (
        await weights
          .where(falseFilter, tf.zeros(falseFilter.shape))
          .sum()
          .buffer()
      ).get(0);
      console.log(`iter #${i + 1} -- error=${err}`);
      if (i === 0 && err === 0) {
        this.trees = this.trees.concat([tree]);
        this.beta = this.beta?.slice(0, i + 1);
        break;
      }
      if (err > 0.5 || err === 0) {
        this.beta = this.beta?.slice(0, i);
        break;
      }
      this.trees = this.trees.concat([tree]);
      (await this.beta.buffer()).set(err / (1.0 - err), i);
      weights = weights.where(falseFilter, weights.mul(this.beta.slice(i, 1)));
      weights = weights.div(weights.sum());
    }
    return this;
  }

  async predict(x: tf.Tensor<tf.Rank>): Promise<tf.Tensor<tf.Rank>> {
    if (this.beta === null) throw 'AdaBoost must be fitted.';
    const zArray = await tf
      .zeros([x.shape[0], this.nClz])
      .as2D(x.shape[0], this.nClz)
      .array();
    let w = tf.log(this.beta.pow(-1));
    if ((await w.sum().buffer()).get(0) === 0) {
      w = tf.ones([this.trees.length]).div(this.trees.length);
    }
    const wArray = await w.as1D().array();
    for (const i of [...Array(this.trees.length).keys()]) {
      const p = await this.trees[i].predict(x);
      const cArray = await p.argMax(1).as1D().array();
      for (const j of [...Array(x.shape[0]).keys()]) {
        zArray[j][cArray[j]] = wArray[i];
      }
    }
    return tf.tensor(zArray);
  }

  async toString(): Promise<string> {
    if (this.beta === null) throw 'AdaBoost must be fitted.';
    const s: string[] = [];
    let w = tf.log(this.beta.pow(-1));
    if ((await w.sum().buffer()).get(0) === 0) {
      w = tf.ones([this.trees.length]).div(this.trees.length);
    }
    for (const i of [...Array(this.trees.length).keys()]) {
      s.push(`tree: #${i + 1} -- weight=${w.slice(i, 1)}`);
      s.push(this.trees[i].toString());
    }
    return s.join('\n');
  }
}
