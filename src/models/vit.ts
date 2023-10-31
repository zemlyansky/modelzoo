import * as tf from '@tensorflow/tfjs';
import { logLayer, positionalEmbedding, classEmbedding, slice, transformer } from '../layers';

export interface VisionTransformerConfig {
  inputResolution: number;
  patchSize: number;
  nEmbd: number;
  nLayer: number;
  nHead: number;
  outputDim: number;
  name?: string;
  debug?: boolean;
  joint?: boolean;
}
export const visionTransformer = VisionTransformer;
export function VisionTransformer(config: VisionTransformerConfig): tf.LayersModel {
  config = Object.assign({ name: 'vit', joint: true }, config);
  const { inputResolution, patchSize, nEmbd, nLayer, nHead, outputDim } = config;
  // Input Layer -> [batch, inputResolution, inputResolution, 3]
  const inputs = tf.input({shape: [inputResolution, inputResolution, 3]});
  let x = inputs as tf.SymbolicTensor;
  if (config.debug) logLayer({name: 'inputs'}).apply(x) as tf.SymbolicTensor;

  // Convolution -> [batch, grid, grid, nEmbd] vs Pytorch [batch, nEmbd, grid, grid]
  x = tf.layers.conv2d({
    name: config.name + '/conv1',
    filters: nEmbd,
    kernelSize: patchSize,
    strides: patchSize,
    useBias: false
  }).apply(x) as tf.SymbolicTensor;
  if (config.debug) x = logLayer({name: config.name + '/post_conv'})
    .apply(x) as tf.SymbolicTensor;

  // Reshape -> [batch, grid * grid, nEmbd]
  const gridSize = Math.floor(inputResolution / patchSize);
  const targetShape = [gridSize * gridSize, nEmbd];
  x = tf.layers.reshape({targetShape: targetShape})
    .apply(x) as tf.SymbolicTensor;
  if (config.debug) x = logLayer({name: config.name + '/post_reshape'})
    .apply(x) as tf.SymbolicTensor;

  // This permutation is not needed as we already in the correct shape
  // x = tf.layers.permute({dims: [2, 1]}).apply(x) as tf.SymbolicTensor;

  // Class embedding + concat -> [batch, grid * grid + 1, nEmbd]
  x = classEmbedding({name: config.name + '/class_embedding'})
    .apply(x) as tf.SymbolicTensor;
  if (config.debug) x = logLayer({name: config.name + '/post_class'})
    .apply(x) as tf.SymbolicTensor;

  // Positional Embedding -> [batch, grid * grid + 1, nEmbd]
  x = positionalEmbedding({name: config.name + '/positional_embedding'})
    .apply(x) as tf.SymbolicTensor;
  if (config.debug) x = logLayer({'name': config.name + '/post_positional'})
    .apply(x) as tf.SymbolicTensor;

  // Layer Normalization
  x = tf.layers.layerNormalization({name: config.name + '/ln_pre', epsilon: 1e-5})
    .apply(x) as tf.SymbolicTensor;
  if (config.debug) x = logLayer({name: config.name + '/pre_transformer'})
    .apply(x) as tf.SymbolicTensor;

  // Transformer
  x = transformer({
    name: config.name + '/transformer',
    nHead: nHead,
    nEmbd: nEmbd,
    nLayer: nLayer,
    activation: 'quickgelu',
    bias: true,
    blockSize: gridSize * gridSize + 1,
    dropout: 0,
    residDrop: 0,
    joint: config.joint
  }).apply(x) as tf.SymbolicTensor;
  if (config.debug) x = logLayer({name: config.name + '/post_transformer'})
    .apply(x) as tf.SymbolicTensor;

  // First token -> [batch, 1, nEmbd]
  x = slice({begin: [0, 0], size: [1, -1]})
    .apply(x) as tf.SymbolicTensor;
  if (config.debug) x = logLayer({name: config.name + '/post_slice'})
    .apply(x) as tf.SymbolicTensor;

  // Reshape -> [batch, nEmbd]
  x = tf.layers.reshape({targetShape: [nEmbd]})
    .apply(x) as tf.SymbolicTensor;
  if (config.debug) x = logLayer({name: config.name + '/post_slice_reshape'})
    .apply(x) as tf.SymbolicTensor;

  // Layer Normalization
  x = tf.layers.layerNormalization({name: config.name + '/ln_post', epsilon: 1e-5})
    .apply(x) as tf.SymbolicTensor;
  if (config.debug) x = logLayer({name: config.name + '/post_ln'})
    .apply(x) as tf.SymbolicTensor;

  // Projection -> [batch, outputDim]
  x = tf.layers.dense({name: config.name + '/proj', units: outputDim, useBias: false})
    .apply(x) as tf.SymbolicTensor;
  if (config.debug) x = logLayer({name: config.name + '/post_proj'})
    .apply(x) as tf.SymbolicTensor;

  return tf.model({ inputs: inputs, outputs: x } as {
    inputs: tf.SymbolicTensor;
    outputs: tf.SymbolicTensor;
  });
}