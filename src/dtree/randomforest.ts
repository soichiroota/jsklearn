import * as tf from '@tensorflow/tfjs';

import { PrunedTree } from './pruning';
import * as entropy from './entropy';
import { ZeroRule } from '../zeror';
import { Linear } from '../linear';

export class RandomTree extends PrunedTree {
  features: number;

  constructor(
    features = 5,
    maxDepth = 5,
    metric = entropy.gini,
    leaf: typeof ZeroRule | typeof Linear = ZeroRule,
    depth = 1
  ) {
    super('critical', false, 0.5, 0.8, maxDepth, metric, leaf, depth);
    this.features = features;
  }

  async splitTree(
    x: tf.Tensor<tf.Rank>,
    y: tf.Tensor<tf.Rank>
  ): Promise<[tf.Tensor<tf.Rank>, tf.Tensor<tf.Rank>]> {
    const allIndex = [...Array(x.shape[1]).keys()];
    tf.util.shuffle(allIndex);
    const x2d = x as tf.Tensor2D;
    const nSample = Math.min(this.features, x2d.shape[1]);
    const index = tf.tensor(allIndex).slice(0, nSample).asType('int32');
    const result = this.splitTreeFast(x.gather(index, 1), y);
    this.featIndex = (await index.buffer()).get(this.featIndex);
    return result;
  }

  getNode(): RandomTree {
    return new RandomTree(
      this.features,
      this.maxDepth,
      this.metric,
      this.leaf,
      this.depth + 1
    );
  }
}
