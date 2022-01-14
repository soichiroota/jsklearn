import * as tf from '@tensorflow/tfjs';

import { DecisionStump } from './dstump';
import * as entropy from './entropy';
import { ZeroRule } from '../zeror';
import { BaseEstimator } from '../base';
import { argsort } from '../numjs';
import { Linear } from '../linear';

export class DecisionTree extends DecisionStump {
  maxDepth: number;
  depth: number;

  constructor(
    maxDepth = 5,
    metric = entropy.gini,
    leaf: typeof ZeroRule | typeof Linear = ZeroRule,
    depth = 1
  ) {
    super(metric, leaf);
    this.maxDepth = maxDepth;
    this.depth = depth;
  }

  getNode(): DecisionTree {
    return new DecisionTree(
      this.maxDepth,
      this.metric,
      this.leaf,
      this.depth + 1
    );
  }

  async splitTreeFast(
    x: tf.Tensor<tf.Rank>,
    y: tf.Tensor<tf.Rank>
  ): Promise<[tf.Tensor<tf.Rank>, tf.Tensor<tf.Rank>]> {
    this.featIndex = 0;
    this.featVal = Infinity;
    let score = Infinity;
    const x2d = x as tf.Tensor2D;
    const y2d = y as tf.Tensor2D;
    const xindex = tf
      .topk(x.transpose(), x.shape[0])
      .indices.reverse()
      .transpose();
    const ysotArray: tf.Tensor<tf.Rank>[] = [];
    for (const i of [...Array(x2d.shape[1]).keys()]) {
      ysotArray.push(y.gather(xindex.gather(i, 1)));
    }
    const ysot = tf
      .stack(ysotArray, 1)
      .reshape([x2d.shape[0], x2d.shape[1], y2d.shape[1]]);
    const xindexArray = await xindex.as2D(x2d.shape[0], x2d.shape[1]).array();
    const xArray = await x2d.array();
    for (const f of [...Array(x.shape[0]).keys()]) {
      const ly = ysot.slice(0, f);
      const ry = ysot.slice(f);
      const loss = [] as number[];
      for (const yp of [...Array(x2d.shape[1]).keys()]) {
        if (f === 0) {
          loss.push(Infinity);
        } else if (
          xArray[xindexArray[f - 1][yp]][yp] !== xArray[xindexArray[f][yp]][yp]
        ) {
          loss.push(await this.makeLoss(ly.gather(yp, 1), ry.gather(yp, 1)));
        } else {
          loss.push(Infinity);
        }
      }
      const i = (await tf.argMin(tf.tensor1d(loss)).buffer()).get(0);
      if (score > loss[i]) {
        score = loss[i];
        this.featIndex = i;
        this.featVal = xArray[xindexArray[f][i]][i];
      }
    }

    const filter = await x
      .gather(this.featIndex, 1)
      .less(this.featVal)
      .flatten()
      .array();
    const left = [] as number[];
    const right = [] as number[];
    for (const i of [...Array(filter.length).keys()]) {
      if (filter[i]) {
        left.push(i);
      } else {
        right.push(i);
      }
    }
    this.score = score;
    return [tf.tensor(left).asType('int32'), tf.tensor(right).asType('int32')];
  }

  splitTreeSlow(
    x: tf.Tensor<tf.Rank>,
    y: tf.Tensor<tf.Rank>
  ): Promise<[tf.Tensor<tf.Rank>, tf.Tensor<tf.Rank>]> {
    return super.splitTree(x, y);
  }

  splitTree(
    x: tf.Tensor<tf.Rank>,
    y: tf.Tensor<tf.Rank>
  ): Promise<[tf.Tensor<tf.Rank>, tf.Tensor<tf.Rank>]> {
    return this.splitTreeFast(x, y);
  }

  async fit(
    x: tf.Tensor<tf.Rank>,
    y: tf.Tensor<tf.Rank>
  ): Promise<DecisionTree> {
    const [left, right] = await this.splitTree(x, y);
    if (this.depth < this.maxDepth) {
      if (left.shape[0] > 0) {
        this.left = this.getNode();
      }
      if (right.shape[0] > 0) {
        this.right = this.getNode();
      }
    }
    const x2d = x as tf.Tensor2D;
    const y2d = y as tf.Tensor2D;
    if (left.shape[0] > 0) {
      const leftArray = await left.flatten().array();
      const xLeftArray = [] as number[][];
      const yLeftArray = [] as number[][];
      for (const i of leftArray) {
        xLeftArray.push(
          (await x2d.slice([i, 0], [1, x2d.shape[1]]).array())[0]
        );
        yLeftArray.push(
          (await y2d.slice([i, 0], [1, y2d.shape[1]]).array())[0]
        );
      }
      this.left = await this.left.fit(
        tf.tensor(xLeftArray),
        tf.tensor(yLeftArray)
      );
    }
    if (right.shape[0] > 0) {
      const rightArray = await right.flatten().array();
      const xRightArray = [] as number[][];
      const yRightArray = [] as number[][];
      for (const i of rightArray) {
        xRightArray.push(
          (await x2d.slice([i, 0], [1, x2d.shape[1]]).array())[0]
        );
        yRightArray.push(
          (await y2d.slice([i, 0], [1, y2d.shape[1]]).array())[0]
        );
      }
      this.right = await this.right.fit(
        tf.tensor(xRightArray),
        tf.tensor(yRightArray)
      );
    }
    return this;
  }

  printLeaf(node: BaseEstimator | null, d = 0): string {
    if (node instanceof DecisionTree) {
      return [
        ` ${'+'.repeat(d)}if feat[${node.featIndex}] <= ${node.featVal} then:`,
        this.printLeaf(node.left, d + 1),
        ` ${'|'.repeat(d)}else:`,
        this.printLeaf(node.right, d + 1),
      ].join('\n');
    } else {
      return ` ${'|'.repeat(d - 1)} ${node?.toString()}`;
    }
  }

  toString(): string {
    return this.printLeaf(this);
  }
}
