import * as tf from '@tensorflow/tfjs';

import { BaseEstimator } from '../base';
import { DecisionTree } from './dtree';
import { PrunedTree } from './pruning';
import { ZeroRule } from '../zeror';
import * as entropy from './entropy';

export class Bagging extends BaseEstimator {
  nTrees: number;
  ratio: number;
  tree: typeof DecisionTree | typeof PrunedTree;
  treeParams: any;
  trees: BaseEstimator[];

  constructor(
    nTrees = 5,
    ratio = 1.0,
    tree: typeof DecisionTree | typeof PrunedTree = PrunedTree,
    treeParams: any = null
  ) {
    super();
    this.nTrees = nTrees;
    this.ratio = ratio;
    this.tree = tree;
    this.treeParams = treeParams
      ? treeParams
      : {
          maxDepth: 5,
          metric: entropy.gini,
          leaf: ZeroRule,
        };
    this.trees = [];
  }

  async fit(x: tf.Tensor<tf.Rank>, y: tf.Tensor<tf.Rank>): Promise<Bagging> {
    this.trees = [];
    const nSample = Math.round(x.shape[0] * this.ratio);
    for (const _ of [...Array(this.nTrees).keys()]) {
      const allIndex = [...Array(x.shape[0]).keys()];
      tf.util.shuffle(allIndex);
      const index = tf.tensor(allIndex).slice(0, nSample).asType('int32');
      const tree =
        this.tree === PrunedTree
          ? new this.tree(
              this.treeParams.prunfnc,
              this.treeParams.pruntest,
              this.treeParams.splitratio,
              this.treeParams.critical,
              this.treeParams.maxDepth,
              this.treeParams.metric,
              this.treeParams.leaf
            )
          : new this.tree(
              this.treeParams.maxDepth,
              this.treeParams.metric,
              this.treeParams.leaf
            );
      await tree.fit(x.gather(index), y.gather(index));
      this.trees = this.trees.concat([tree]);
    }
    return this;
  }

  async predict(x: tf.Tensor<tf.Rank>): Promise<tf.Tensor<tf.Rank>> {
    const z: tf.Tensor<tf.Rank>[] = [];
    for (const tree of this.trees) {
      z.push(await tree.predict(x));
    }
    return tf.mean(tf.stack(z), 0);
  }

  toString(): string {
    const strings: string[] = [];
    for (const i of [...Array(this.trees.length).keys()]) {
      strings.push(`tree#${i}\n${this.trees[i]}`);
    }
    return strings.join('\n');
  }
}
