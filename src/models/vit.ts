import * as tf from '@tensorflow/tfjs';
import { logLayer, positionalEmbedding, classEmbedding, slice, transformer } from '../layers';

interface VisionTransformerOptions {
  inputResolution: number;
  patchSize: number;
  nEmbd: number;
  layers: number;
  nHead: number;
  outputDim: number;
  debug?: boolean;
}
export function VisionTransformer(config: VisionTransformerOptions): tf.LayersModel {
  const { inputResolution, patchSize, nEmbd, layers, nHead, outputDim } = config;
  // Input Layer -> [batch, inputResolution, inputResolution, 3]
  const inputs = tf.input({shape: [inputResolution, inputResolution, 3]});
  let x = inputs as tf.SymbolicTensor;
  if (config.debug) logLayer({name: 'inputs'}).apply(x) as tf.SymbolicTensor;

  // Convolution -> [batch, grid, grid, nEmbd] vs Pytorch [batch, nEmbd, grid, grid]
  x = tf.layers.conv2d({
    name: 'vit/conv1',
    filters: nEmbd,
    kernelSize: patchSize,
    strides: patchSize,
    useBias: false
  }).apply(x) as tf.SymbolicTensor;
  if (config.debug) x = logLayer({name: 'post_conv'})
    .apply(x) as tf.SymbolicTensor;

  // Reshape -> [batch, grid * grid, nEmbd]
  const gridSize = Math.floor(inputResolution / patchSize);
  const targetShape = [gridSize * gridSize, nEmbd];
  x = tf.layers.reshape({targetShape: targetShape})
    .apply(x) as tf.SymbolicTensor;
  if (config.debug) x = logLayer({name: 'post_reshape'})
    .apply(x) as tf.SymbolicTensor;

  // This permutation is not needed as we already in the correct shape
  // x = tf.layers.permute({dims: [2, 1]}).apply(x) as tf.SymbolicTensor;

  // Class embedding + concat -> [batch, grid * grid + 1, nEmbd]
  x = classEmbedding({name: 'vit/class_embedding'})
    .apply(x) as tf.SymbolicTensor;
  if (config.debug) x = logLayer({name: 'post_class'})
    .apply(x) as tf.SymbolicTensor;

  // Positional Embedding -> [batch, grid * grid + 1, nEmbd]
  x = positionalEmbedding({name: 'vit/positional_embedding'})
    .apply(x) as tf.SymbolicTensor;
  if (config.debug) x = logLayer({'name': 'post_positional'})
    .apply(x) as tf.SymbolicTensor;

  // Layer Normalization
  x = tf.layers.layerNormalization({name: 'vit/ln_pre', epsilon: 1e-5})
    .apply(x) as tf.SymbolicTensor;
  if (config.debug) x = logLayer({name: 'pre_transformer'})
    .apply(x) as tf.SymbolicTensor;

  // Transformer
  x = transformer({
    nHead: nHead,
    nEmbd: nEmbd,
    layers: layers,
    activation: 'quickgelu',
    bias: true,
    blockSize: gridSize * gridSize + 1,
    dropout: 0,
    residDrop: 0,
  }).apply(x) as tf.SymbolicTensor;
  if (config.debug) x = logLayer({name: 'post_transformer'})
    .apply(x) as tf.SymbolicTensor;

  // First token -> [batch, 1, nEmbd]
  x = slice([0, 0], [1, -1])
    .apply(x) as tf.SymbolicTensor;
  if (config.debug) x = logLayer({name: 'post_slice'})
    .apply(x) as tf.SymbolicTensor;

  // Reshape -> [batch, nEmbd]
  x = tf.layers.reshape({targetShape: [nEmbd]})
    .apply(x) as tf.SymbolicTensor;
  if (config.debug) x = logLayer({name: 'post_slice_reshape'})
    .apply(x) as tf.SymbolicTensor;

  // Layer Normalization
  x = tf.layers.layerNormalization({name: 'vit/ln_post', epsilon: 1e-5})
    .apply(x) as tf.SymbolicTensor;
  if (config.debug) x = logLayer({name: 'post_ln'})
    .apply(x) as tf.SymbolicTensor;

  // Projection -> [batch, outputDim]
  x = tf.layers.dense({name: 'vit/proj', units: outputDim, useBias: false})
    .apply(x) as tf.SymbolicTensor;
  if (config.debug) x = logLayer({name: 'post_proj'})
    .apply(x) as tf.SymbolicTensor;

  return tf.model({ inputs: inputs, outputs: x } as {
    inputs: tf.SymbolicTensor;
    outputs: tf.SymbolicTensor;
  });
}