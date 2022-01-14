import * as tf from '@tensorflow/tfjs';

import * as entropy from './entropy';
import { ZeroRule } from '../zeror';
import { Linear } from '../linear';
import { BaseEstimator } from '../base';

export class DecisionStump extends BaseEstimator {
  leaf: any;
  metric: (y: tf.Tensor<tf.Rank>) => Promise<number>;
  left: BaseEstimator | null;
  right: BaseEstimator | null;
  featIndex: number;
  featVal: number;
  score: number;

  constructor(
    metric = entropy.gini,
    leaf: typeof ZeroRule | typeof Linear = ZeroRule
  ) {
    super();
    this.metric = metric;
    this.leaf = leaf;
    this.left = null;
    this.right = null;
    this.featIndex = 0;
    this.featVal = NaN;
    this.score = NaN;
  }

  async makeSplit(
    feat: tf.Tensor<tf.Rank>,
    val: number
  ): Promise<[tf.Tensor<tf.Rank>, tf.Tensor<tf.Rank>]> {
    let left = tf.tensor(new Uint8Array());
    let right = tf.tensor(new Uint8Array());
    const featArray = await feat.flatten().array();
    for (const i of [...Array(feat.shape[0]).keys()]) {
      const v = featArray[i];
      if (v < val) {
        left = tf.concat([left, tf.tensor([i])]);
      } else {
        right = tf.concat([right, tf.tensor([i])]);
      }
    }
    return [left.asType('int32'), right.asType('int32')];
  }

  async makeLoss(
    y1: tf.Tensor<tf.Rank>,
    y2: tf.Tensor<tf.Rank>
  ): Promise<number> {
    if (y1.shape[0] == 0 || y2.shape[0] == 0) {
      return Infinity;
    }
    const total = y1.shape[0] + y2.shape[0];
    const m1 = ((await this.metric(y1)) * y1.shape[0]) / total;
    const m2 = ((await this.metric(y2)) * y2.shape[0]) / total;
    return m1 + m2;
  }

  async splitTree(
    x: tf.Tensor<tf.Rank>,
    y: tf.Tensor<tf.Rank>
  ): Promise<[tf.Tensor<tf.Rank>, tf.Tensor<tf.Rank>]> {
    this.featIndex = 0;
    this.featVal = Infinity;
    let score = Infinity;
    let left = tf.tensor([...Array(x.shape[0]).keys()]).asType('int32');
    let right = tf.tensor(new Uint8Array()).asType('int32');
    const x2d = x as tf.Tensor2D;
    for (const i of [...Array(x.shape[1]).keys()]) {
      const feat = x2d.slice([0, i], [x2d.shape[0], 1]);
      for (const val of await feat.flatten().array()) {
        const [l, r] = await this.makeSplit(feat, val);
        const loss = await this.makeLoss(y.gather(l), y.gather(r));
        if (score > loss) {
          score = loss;
          left = l;
          right = r;
          this.featIndex = i;
          this.featVal = val;
        }
      }
    }
    this.score = score;
    return [left, right];
  }

  async fit(
    x: tf.Tensor<tf.Rank>,
    y: tf.Tensor<tf.Rank>
  ): Promise<DecisionStump> {
    this.left = new this.leaf();
    this.right = new this.leaf();
    const [left, right] = await this.splitTree(x, y);
    if (left.shape[0] > 0) {
      await this.left?.fit(x.gather(left), y.gather(left));
    }
    if (right.shape[0] > 0) {
      await this.right?.fit(x.gather(right), y.gather(right));
    }
    return this;
  }

  async predict(x: tf.Tensor<tf.Rank>): Promise<tf.Tensor<tf.Rank>> {
    if (this.left === null || this.right === null) {
      throw 'Model must be fitted.';
    }
    const feat = x.slice([0, this.featIndex], [x.shape[0], 1]);
    const val = this.featVal;
    const [l, r] = (await this.makeSplit(feat, val)) as [
      tf.Tensor1D,
      tf.Tensor1D
    ];
    let z = null;
    if (l.shape[0] > 0 && r.shape[0] > 0) {
      const left = (await this.left.predict(x.gather(l))) as tf.Tensor2D;
      const right = (await this.right.predict(x.gather(r))) as tf.Tensor2D;
      z = tf.zeros([x.shape[0], left.shape[1]]);
      const zArray = (await z.array()) as number[][];
      const lArray = await l.array();
      const leftArray = await left.array();
      for (const i of [...Array(l.shape[0]).keys()]) {
        zArray[lArray[i]] = leftArray[i];
      }
      const rArray = await r.array();
      const rightArray = await right.array();
      for (const i of [...Array(r.shape[0]).keys()]) {
        zArray[rArray[i]] = rightArray[i];
      }
      z = tf.tensor(zArray);
    } else if (l.shape[0] > 0) {
      z = await this.left.predict(x);
    } else if (r.shape[0] > 0) {
      z = await this.right.predict(x);
    }
    return z as tf.Tensor<tf.Rank>;
  }

  toString(): string {
    return [
      `if feat[${this.featIndex}] <= ${this.featVal} then:`,
      ` ${this.left}`,
      'else:',
      ` ${this.right}`,
    ].join('\n');
  }
}
