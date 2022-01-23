import * as tf from '@tensorflow/tfjs';

import { BaseEstimator } from '../../base';
import { PrunedTree } from '../pruning';
import * as entropy from '../entropy';
import { Linear } from '../../linear';

export class AdaBoostRT extends BaseEstimator {
  boost: number;
  maxDepth: number;
  trees: PrunedTree[];
  beta: tf.Tensor<tf.Rank> | null;
  threshold: number;

  constructor(threshold = 0.01, boost = 5, maxDepth = 5) {
    super();
    this.boost = boost;
    this.maxDepth = maxDepth;
    this.trees = [];
    this.beta = null;
    this.threshold = threshold;
  }

  accumulate(weight: tf.Tensor<tf.Rank>): tf.Tensor<tf.Rank> {
    let cumulativeWeights = tf.tensor([]);
    for (const i of [...Array(weight.shape[0]).keys()]) {
      const cumulativeWeight = weight
        .slice(0, i + 1)
        .sum()
        .reshape([1, 1]);
      cumulativeWeights = tf.concat([cumulativeWeights, cumulativeWeight]);
    }
    return cumulativeWeights;
  }

  async fit(x: tf.Tensor<tf.Rank>, y: tf.Tensor<tf.Rank>): Promise<AdaBoostRT> {
    const [_x, _y] = [x.clone(), y.clone()];
    this.trees = [];
    this.beta = tf.zeros([this.boost]);
    let weights = tf.ones([x.shape[0]]).div(x.shape[0]);
    const threshold = this.threshold;
    for (const i of [...Array(this.boost).keys()]) {
      let tree = new PrunedTree(
        'critical',
        true,
        0.5,
        0.8,
        this.maxDepth,
        entropy.deviation,
        Linear
      );
      let pWeight = weights.div(weights.sum());
      let idx = tf.tensor([]);
      for (const j of [...Array(_x.shape[0]).keys()]) {
        const cumulativePWeights = this.accumulate(pWeight);
        const p = tf.randomUniform([1]);
        const diff = tf.abs(cumulativePWeights.sub(p));
        const minIdx = diff.argMin().reshape([1]);
        idx = idx.concat(minIdx);
        if (j === _x.shape[0] - 1) break;
        (await pWeight.buffer()).set(0, (await minIdx.buffer()).get(0));
        pWeight = pWeight.div(pWeight.sum());
      }
      const x = _x.gather(idx);
      const y = _y.gather(idx);
      tree = await tree.fit(x, y);
      const z = await tree.predict(x);
      const l = tf.abs(z.sub(y)).reshape([-1]).div(y.mean());
      const filter = l.less(threshold);
      const err = (
        await tf.zeros(weights.shape).where(filter, weights).sum().buffer()
      ).get(0);
      console.log(`iter #${i + 1} -- error=${err}`);
      if (err < Math.pow(10, -10)) {
        this.beta = this.beta?.slice(0, i);
        break;
      }
      this.trees = this.trees.concat([tree]);
      (await this.beta.buffer()).set(err / (1 - err), i);
      weights = weights.where(
        filter,
        weights.mul(this.beta.slice(i, 1).pow(2))
      );
      weights = weights.div(weights.sum());
    }
    return this;
  }

  async predict(x: tf.Tensor<tf.Rank>): Promise<tf.Tensor<tf.Rank>> {
    if (this.beta === null) throw 'AdaBoost must be fitted.';
    let z = tf.zeros([x.shape[0], 1]);
    const w = tf.log(this.beta?.pow(-1));
    for (const i of [...Array(this.trees.length).keys()]) {
      const p = await this.trees[i].predict(x);
      z = z.add(p.mul(w.slice(i, 1)));
    }
    return z.div(w.sum());
  }

  async toString(): Promise<string> {
    if (this.beta === null) return 'null';
    const s: string[] = [];
    const w = tf.log(this.beta?.pow(-1));
    for (const i of [...Array(this.trees.length).keys()]) {
      s.push(`tree: #${i + 1} -- weight=${(await w.buffer()).get(0)}`);
      s.push(this.trees[i].toString());
    }
    return s.join('\n');
  }
}
