import * as base from './base';
import * as zeror from './zeror';
import * as linear from './linear';
import * as entropy from './dtree/entropy';
import * as dstump from './dtree/dstump';
import * as dtree from './dtree/dtree';
import * as pruning from './dtree/pruning';
import * as bagging from './dtree/bagging';

export default {
  base: base,
  zeror: zeror,
  linear: linear,
  entropy: entropy,
  dstump: dstump,
  dtree: dtree,
  pruning: pruning,
  bagging: bagging,
};
