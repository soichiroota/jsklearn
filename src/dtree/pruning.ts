import * as tf from '@tensorflow/tfjs';

import { BaseEstimator } from '../base';
import * as entropy from '../dtree/entropy';
import { ZeroRule } from '../zeror';
import { Linear } from '../linear';
import { DecisionTree } from './dtree';

export async function reduceerror(
  node: BaseEstimator,
  x: tf.Tensor<tf.Rank>,
  y: tf.Tensor<tf.Rank>
): Promise<BaseEstimator> {
  if (node instanceof PrunedTree) {
    const feat = x.slice([0, node.featIndex], [x.shape[0], 1]);
    const val = node.featVal;
    const [l, r] = await node.makeSplit(feat, val);
    if (val === Infinity || r.shape[0] === 0) {
      return await reduceerror(node.left, x, y);
    } else if (l.shape[0] === 0) {
      return await reduceerror(node.right, x, y);
    }
    node.left = await reduceerror(node.left, x.gather(l), y.gather(l));
    node.right = await reduceerror(node.right, x.gather(r), y.gather(r));
    const p1 = await node.predict(x);
    const p2 = await node.left.predict(x);
    const p3 = await node.right.predict(x);
    const y2d = y as tf.Tensor2D;
    let d1, d2, d3;
    if (y2d.shape[1] > 1) {
      const ya = y.argMax(1);
      d1 = tf.sum(p1.argMax(1) !== ya);
      d2 = tf.sum(p2.argMax(1) !== ya);
      d3 = tf.sum(p3.argMax(1) !== ya);
    } else {
      d1 = tf.mean(p1.sub(y).pow(2));
      d2 = tf.sum(p2.sub(y).pow(2));
      d3 = tf.sum(p3.sub(y).pow(2));
    }
    if (d2 <= d1 || d3 <= d1) {
      if (d2 < d3) {
        return node.left;
      } else {
        return node.right;
      }
    }
  }
  return node;
}

export function getscore(node: BaseEstimator, score: number[]): void {
  if (node instanceof PrunedTree) {
    if (node.score >= 0 && node.score !== Infinity) {
      score = score.concat([node.score]);
    }
    getscore(node.left, score);
    getscore(node.right, score);
  }
}

export function criticalscore(
  node: BaseEstimator,
  scoreMax: number
): BaseEstimator {
  if (node instanceof PrunedTree) {
    node.left = criticalscore(node.left, scoreMax);
    node.right = criticalscore(node.right, scoreMax);
    if (node.score > scoreMax) {
      const leftisleaf = !(node.left instanceof PrunedTree);
      const rightisleaf = !(node.right instanceof PrunedTree);
      if (leftisleaf && rightisleaf) {
        return node.left;
      } else if (leftisleaf && !rightisleaf) {
        return node.right;
      } else if (!leftisleaf && rightisleaf) {
        return node.left;
      } else if (
        node.left instanceof PrunedTree &&
        node.right instanceof PrunedTree &&
        node.left.score < node.right.score
      ) {
        return node.left;
      } else {
        return node.right;
      }
    }
  }
  return node;
}

export class PrunedTree extends DecisionTree {
  prunfnc: string | null;
  pruntest: boolean;
  splitratio: number;
  critical: number;

  constructor(
    prunfnc: string | null = 'critical',
    pruntest = false,
    splitratio = 0.5,
    critical = 0.8,
    maxDepth = 5,
    metric = entropy.gini,
    leaf: typeof ZeroRule | typeof Linear = ZeroRule,
    depth = 1
  ) {
    super(maxDepth, metric, leaf, depth);
    this.prunfnc = prunfnc;
    this.pruntest = pruntest;
    this.splitratio = splitratio;
    this.critical = critical;
  }

  getNode(): PrunedTree {
    return new PrunedTree(
      this.prunfnc,
      this.pruntest,
      this.splitratio,
      this.critical,
      this.maxDepth,
      this.metric,
      this.leaf,
      this.depth + 1
    );
  }

  async fitLeaf(
    x: tf.Tensor<tf.Rank>,
    y: tf.Tensor<tf.Rank>
  ): Promise<PrunedTree> {
    const feat = x.slice([0, this.featIndex], [x.shape[0], 1]);
    const val = this.featVal;
    const [l, r] = await this.makeSplit(feat, val);
    if (l.shape[0] > 0) {
      if (this.left instanceof PrunedTree) {
        this.left = await this.left.fitLeaf(x.gather(l), y.gather(l));
      } else {
        this.left = await this.left.fit(x.gather(l), y.gather(l));
      }
    }
    if (r.shape[0] > 0) {
      if (this.right instanceof PrunedTree) {
        this.right = await this.right.fitLeaf(x.gather(r), y.gather(r));
      } else {
        this.right = await this.right.fit(x.gather(r), y.gather(r));
      }
    }
    return this;
  }

  getDataForPruning(
    x: tf.Tensor<tf.Rank>,
    y: tf.Tensor<tf.Rank>
  ): [
    tf.Tensor<tf.Rank>,
    tf.Tensor<tf.Rank>,
    tf.Tensor<tf.Rank>,
    tf.Tensor<tf.Rank>
  ] {
    let [xT, yT] = [x.clone(), y.clone()];
    if (this.depth === 1 && this.prunfnc !== null) {
      if (this.pruntest) {
        const nTest = Math.round(x.shape[0] * this.splitratio);
        const nIdx = [...Array(x.shape[0]).keys()];
        tf.util.shuffle(nIdx);
        const tmpx = x.gather(nIdx.slice(nTest));
        const tmpy = y.gather(nIdx.slice(nTest));
        xT = x.gather(nIdx.slice(0, nTest));
        yT = y.gather(nIdx.slice(0, nTest));
        x = tmpx;
        y = tmpy;
      }
    }
    return [xT, yT, x, y];
  }

  async fit(x: tf.Tensor<tf.Rank>, y: tf.Tensor<tf.Rank>): Promise<PrunedTree> {
    const [xT, yT, xNew, yNew] = this.getDataForPruning(x, y);

    this.left = new this.leaf();
    this.right = new this.leaf();
    const [left, right] = await this.splitTree(xNew, yNew);
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
        this.left = await this.left.fit(xNew.gather(left), yNew.gather(left));
      }
      if (right.shape[0] > 0) {
        this.right = await this.right.fit(
          xNew.gather(right),
          yNew.gather(right)
        );
      }
    }

    if (this.depth === 1 && this.prunfnc !== null) {
      if (this.prunfnc === 'reduce') {
        await reduceerror(this, xT, yT);
      } else if (this.prunfnc === 'critical') {
        const score: number[] = [];
        getscore(this, score);
        if (score.length > 0) {
          const i = Math.round(score.length * this.critical);
          const sortedScore = tf.topk(score, score.length, true);
          const scoreMax = (
            await sortedScore.values
              .slice(Math.min(i, score.length - 1), 1)
              .buffer()
          ).get(0);
          criticalscore(this, scoreMax);
        }
        await this.fitLeaf(x, y);
      }
    }

    return this;
  }
}
