import * as tf from '@tensorflow/tfjs';

export abstract class BaseEstimator {
  abstract fit(
    x: tf.Tensor<tf.Rank> | null,
    y: tf.Tensor<tf.Rank>
  ): Promise<BaseEstimator>;

  abstract predict(
    x: tf.Tensor<tf.Rank> | tf.Tensor<tf.Rank>[]
  ): Promise<tf.Tensor<tf.Rank>>;

  abstract toString(): string;
}
