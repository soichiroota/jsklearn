import * as tf from '@tensorflow/tfjs';

import { ZeroRule } from '../../zeror';
import { DecisionStump } from '../dstump';
import { Linear } from '../../linear';
import { criticalscore, getscore, PrunedTree } from '../pruning';

export async function wGini(
  y: tf.Tensor<tf.Rank>,
  weight?: tf.Tensor<tf.Rank>
): Promise<number> {
  if (weight === undefined) throw 'weight is required.';
  const i = y.argMax(1);
  const uniqI = tf.unique(i);
  const clz = await uniqI.values.as1D().array();
  let score = 0.0;
  for (const j of [...Array(uniqI.values.shape[0])]) {
    const p = (
      await weight
        .where(i.equal(tf.fill(i.shape, clz[j])), tf.zeros(weight.shape))
        .sum()
        .buffer()
    ).get(0);
    score += Math.pow(p, 2);
  }
  return 1.0 - score;
}

export async function wInfgain(
  y: tf.Tensor<tf.Rank>,
  weight?: tf.Tensor<tf.Rank>
): Promise<number> {
  if (weight === undefined) throw 'weight is required.';
  const i = y.argMax(1);
  const uniqI = tf.unique(i);
  const clz = await uniqI.values.as1D().array();
  let score = 0.0;
  for (const j of [...Array(uniqI.values.shape[0])]) {
    const p = (
      await weight
        .where(i.equal(tf.fill(i.shape, clz[j])), tf.zeros(weight.shape))
        .sum()
        .buffer()
    ).get(0);
    if (p !== 0) {
      score += p * Math.log2(p);
    }
  }
  return -score;
}

export class WeightedZeroRule extends ZeroRule {
  async fit(
    x: tf.Tensor<tf.Rank> | null = null,
    y: tf.Tensor<tf.Rank>,
    weight?: tf.Tensor<tf.Rank>
  ): Promise<WeightedZeroRule> {
    if (weight === undefined) throw `${this}: weight is required.`;
    this.r = tf.dot(weight, y).div(tf.sum(weight, 0));
    return this;
  }
}

export class WeightedDecisionStump extends DecisionStump {
  weight: tf.Tensor<tf.Rank> | null;

  constructor(
    metric = wInfgain,
    leaf:
      | typeof WeightedZeroRule
      | typeof ZeroRule
      | typeof Linear = WeightedZeroRule
  ) {
    super(metric, leaf);
    this.weight = null;
  }

  async makeLoss(
    y1: tf.Tensor<tf.Rank>,
    y2: tf.Tensor<tf.Rank>,
    l: tf.Tensor<tf.Rank>,
    r: tf.Tensor<tf.Rank>
  ): Promise<number> {
    if (y1.shape[0] === 0 || y2.shape[0] === 0) {
      return Infinity;
    }
    const w1 = this.weight?.gather(l).div(tf.sum(this.weight.gather(l)));
    const w2 = this.weight?.gather(r).div(tf.sum(this.weight.gather(r)));
    const total = y1.shape[0] + y2.shape[0];
    const m1 = (await this.metric(y1, w1)) * (y1.shape[0] / total);
    const m2 = (await this.metric(y2, w2)) * (y2.shape[0] / total);
    return m1 + m2;
  }

  async fit(
    x: tf.Tensor<tf.Rank>,
    y: tf.Tensor<tf.Rank>,
    weight?: tf.Tensor<tf.Rank>
  ): Promise<WeightedDecisionStump> {
    if (weight === undefined) throw `${this}: weight is required.`;
    this.weight = weight;
    this.left = new this.leaf();
    this.right = new this.leaf();
    const [left, right] = await this.splitTree(x, y);
    if (left.shape[0] > 0) {
      this.left = await this.left.fit(
        x.gather(left),
        y.gather(left),
        weight.gather(left).div(tf.sum(weight.gather(left)))
      );
    }
    if (right.shape[0] > 0) {
      this.right = await this.right.fit(
        x.gather(right),
        y.gather(right),
        weight.gather(right).div(tf.sum(weight.gather(right)))
      );
    }
    return this;
  }
}

export class WeightedDecisionTree extends PrunedTree {
  weight: tf.Tensor<tf.Rank> | null;

  constructor(
    maxDepth = 5,
    metric = wGini,
    leaf:
      | typeof WeightedZeroRule
      | typeof ZeroRule
      | typeof Linear = WeightedZeroRule,
    depth = 1
  ) {
    super('critical', true, 0.5, 0.8, maxDepth, metric, leaf, depth);
    this.weight = null;
  }

  getNode(): WeightedDecisionTree {
    return new WeightedDecisionTree(
      this.maxDepth,
      this.metric,
      this.leaf,
      this.depth + 1
    );
  }

  async makeLoss(
    y1: tf.Tensor<tf.Rank>,
    y2: tf.Tensor<tf.Rank>,
    l: tf.Tensor<tf.Rank>,
    r: tf.Tensor<tf.Rank>
  ): Promise<number> {
    if (y1.shape[0] === 0 || y2.shape[0] === 0) {
      return Infinity;
    }
    const w1 = this.weight?.gather(l).div(tf.sum(this.weight.gather(l)));
    const w2 = this.weight?.gather(r).div(tf.sum(this.weight.gather(r)));
    const total = y1.shape[0] + y2.shape[0];
    const m1 = (await this.metric(y1, w1)) * (y1.shape[0] / total);
    const m2 = (await this.metric(y2, w2)) * (y2.shape[0] / total);
    return m1 + m2;
  }

  async fit(
    x: tf.Tensor<tf.Rank>,
    y: tf.Tensor<tf.Rank>,
    weight?: tf.Tensor<tf.Rank>
  ): Promise<WeightedDecisionTree> {
    if (weight === undefined) throw `${this}: weight is required.`;
    this.weight = weight;
    if (this.depth === 1 && this.prunfnc !== null) {
      const [xT, yT] = [x.clone(), y.clone()];
    }
    this.left = new this.leaf();
    this.right = new this.leaf();
    const [left, right] = await this.splitTree(x, y);
    if (this.depth < this.maxDepth) {
      if (left.shape[0] > 0) {
        this.left = this.getNode();
      }
      if (right.shape[0] > 0) {
        this.right = this.getNode();
      }
    }
    if (this.depth < this.maxDepth || this.prunfnc !== 'critical') {
      if (left.shape[0] > 0) {
        this.left = await this.left.fit(
          x.gather(left),
          y.gather(left),
          weight.gather(left).div(tf.sum(weight.gather(left)))
        );
      }
      if (right.shape[0] > 0) {
        this.right = await this.right.fit(
          x.gather(right),
          y.gather(right),
          weight.gather(right).div(tf.sum(weight.gather(right)))
        );
      }
    }
    if (this.depth === 1 && this.prunfnc !== null) {
      if (this.prunfnc === 'critical') {
        const score: number[] = [];
        getscore(this, score);
        const i = Math.round(score.length * this.critical);
        const scoreMax = score.sort()[i];
        criticalscore(this, scoreMax);
        this.fitLeaf(x, y, weight);
      }
    }
    return this;
  }

  async fitLeaf(
    x: tf.Tensor<tf.Rank>,
    y: tf.Tensor<tf.Rank>,
    weight?: tf.Tensor<tf.Rank>
  ): Promise<WeightedDecisionTree> {
    if (weight === undefined) throw `${this}: weight is required.`;
    const feat = x.gather([this.featIndex], 1);
    const val = this.featVal;
    const [l, r] = await this.makeSplit(feat, val);
    if (l.shape[0] > 0) {
      if (this.left instanceof PrunedTree) {
        this.left = await this.left.fitLeaf(
          x.gather(l),
          y.gather(l),
          weight.gather(l)
        );
      } else {
        this.left = await this.left.fit(
          x.gather(l),
          y.gather(l),
          weight.gather(l)
        );
      }
    }
    if (r.shape[0] > 0) {
      if (this.right instanceof PrunedTree) {
        this.right = await this.right.fitLeaf(
          x.gather(r),
          y.gather(r),
          weight.gather(r)
        );
      } else {
        this.right = await this.right.fit(
          x.gather(r),
          y.gather(r),
          weight.gather(r)
        );
      }
    }
    return this;
  }
}
