import * as tf from '@tensorflow/tfjs';

import { BaseEstimator } from './base';

export class Linear extends BaseEstimator {
  epochs: number;
  earlyStopping: tf.EarlyStopping;
  model: tf.LayersModel | null;
  kernelInitializer: string;
  learningRate: number;
  optimizer: tf.Optimizer | string;
  loss: string;
  batchSize: number;
  metrics: string | string[];

  constructor(
    epochs = 20,
    earlyStopping = null as tf.EarlyStopping | null,
    kernelInitializer = 'Zeros',
    learningRate = 0.01,
    optimizer = 'sgd',
    loss = 'meanSquaredError',
    batchSize = 1,
    metrics = null as string | string[] | null
  ) {
    super();
    this.epochs = epochs;
    this.earlyStopping = earlyStopping
      ? (earlyStopping as tf.EarlyStopping)
      : tf.callbacks.earlyStopping({ monitor: 'loss' });
    this.model = null;
    this.kernelInitializer = kernelInitializer;
    this.learningRate = learningRate;
    this.optimizer = optimizer;
    this.loss = loss;
    this.batchSize = batchSize;
    this.metrics = (metrics ? metrics : []) as string | string[];
  }

  async fit(x: tf.Tensor<tf.Rank>, y: tf.Tensor<tf.Rank>): Promise<Linear> {
    const x2d = x as tf.Tensor2D;
    const input = tf.input({ shape: [x2d.shape[1]] });
    const linearLayer = tf.layers.dense({
      units: 1,
      kernelInitializer: this.kernelInitializer,
      useBias: true,
    });
    const output = linearLayer.apply(input) as
      | tf.SymbolicTensor
      | tf.SymbolicTensor[];
    this.model = tf.model({ inputs: input, outputs: output });
    this.model.compile({
      optimizer: this.optimizer,
      loss: this.loss,
      metrics: this.metrics,
    });
    await this.model.fit(x, y, {
      batchSize: this.batchSize,
      epochs: this.epochs,
      callbacks: this.earlyStopping,
    });
    return this;
  }

  async predict(
    x: tf.Tensor<tf.Rank> | tf.Tensor<tf.Rank>[]
  ): Promise<tf.Tensor<tf.Rank>> {
    if (this.model) {
      return this.model.predict(x) as tf.Tensor<tf.Rank>;
    }
    throw 'Model must be fitted.';
  }

  toString(): string {
    if (this.model) {
      const weights = this.model.getLayer(undefined, 1).getWeights();
      return `Model:\n [${weights.toString()}]`;
    }
    return 'Model:\n null';
  }
}
