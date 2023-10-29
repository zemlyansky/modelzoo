import { positionalEmbedding, eot, transformer, TransformerConfig, BaseConfig } from '../layers';
import { visionTransformer, VisionTransformerConfig } from './vit';
import { getWeight, setWeights, StatsWeights } from '../utils';
import * as tf from '@tensorflow/tfjs';

function resizeBicubic(inputTensor, [newHeight, newWidth]) {
  const [height, width, numChannels] = inputTensor.shape;

  const pixels = inputTensor.arraySync();
  const newPixels = new Array(newHeight).fill(0).map(row => new Array(newWidth).fill(0).map(col => new Array(numChannels).fill(0)));

  const xRatio = width / newWidth;
  const yRatio = height / newHeight;

  for (let y = 0; y < newHeight; y++) {
    for (let x = 0; x < newWidth; x++) {
      const px = x * xRatio;
      const py = y * yRatio;

      for (let c = 0; c < numChannels; c++) {
        newPixels[y][x][c] = cubicInterpolate(pixels, px, py, width, height, c);
      }
    }
  }

  return tf.tensor(newPixels);
}

function cubicInterpolate(pixels, x, y, width, height, channel) {
  const x1 = Math.floor(x);
  const y1 = Math.floor(y);

  // Handle border cases
  const x0 = Math.max(x1 - 1, 0);
  const y0 = Math.max(y1 - 1, 0);
  const x2 = Math.min(x1 + 1, width - 1);
  const y2 = Math.min(y1 + 1, height - 1);
  const x3 = Math.min(x1 + 2, width - 1);
  const y3 = Math.min(y1 + 2, height - 1);

  const p = [x0, x1, x2, x3].map(x => [y0, y1, y2, y3].map(y => pixels[y][x][channel]));

  // Cubic interpolation for each row
  const rows = p.map(row => cubicInterpolateArray(row, x - x1));

  // Cubic interpolation for the column
  return cubicInterpolateArray(rows, y - y1);
}

function cubicInterpolateArray(values, x) {
  const [v0, v1, v2, v3] = values;
  const P = (v3 - v2) - (v0 - v1);
  const Q = (v0 - v1) - P;
  const R = v2 - v0;
  const S = v1;

  return P * Math.pow(x, 3) + Q * Math.pow(x, 2) + R * x + S;
}

function calculateImageSize(height: number, width: number, target: number): [number, number] {
  let newHeight: number;
  let newWidth: number;

  if (height > width) {
    newWidth = target;
    newHeight = Math.floor((height / width) * target);
  } else {
    newHeight = target;
    newWidth = Math.floor((width / height) * target);
  }
  return [newHeight, newWidth];
}

export function transform(inputTensor: tf.Tensor3D | tf.Tensor4D, n_px: number) {
  const s = inputTensor.shape.length;

  // Scale
  let img = tf.div(inputTensor, 255) as tf.Tensor3D | tf.Tensor4D;

  // Resize
  const [height, width] = calculateImageSize(inputTensor.shape[s-3], inputTensor.shape[s-2], n_px);
  img = tf.image.resizeBilinear(img, [height, width]);
  // img = resizeBicubic(img, [height, width]) as tf.Tensor3D | tf.Tensor4D;

  const centerY = Math.floor(img.shape[s-3] / 2);
  const centerX = Math.floor(img.shape[s-2] / 2);
  const halfLength = Math.floor(n_px / 2)

  if (s === 3) {
    img = img.expandDims(0) as tf.Tensor4D;
  }

  img = tf.slice(
    img,
    [0, centerY - halfLength, centerX - halfLength, 0],
    [-1, n_px, n_px, 3]
  );

  // if (s === 3) {
  //   // Convert back to 3D if it was 3D initially
  //   img = img.squeeze();
  // }

  // Normalize
  const mean = tf.tensor([0.48145466, 0.4578275, 0.40821073]); //.reshape([1,1,1,3]);
  const std = tf.tensor([0.26862954, 0.26130258, 0.27577711]); //.reshape([1,1,1,3]);
  img = tf.div(tf.sub(img, mean), std);

  return img;
}


interface CLIPTextConfig extends TransformerConfig {
  vocabSize: number;
}
export function CLIPTextModel (config: CLIPTextConfig): tf.LayersModel {
  config = Object.assign({ name: 'text_model' }, config);
  const input = tf.input({ shape: [config.blockSize] }) as tf.SymbolicTensor;

  // Token embedding
  let x = tf.layers
    .embedding({
      name: config.name + '/token_embedding',
      inputDim: config.vocabSize,
      outputDim: config.nEmbd,
      embeddingsInitializer: 'zeros'
    })
    .apply(input) as tf.SymbolicTensor;

  x = positionalEmbedding({ name: config.name + '/positional_embedding' })
    .apply(x) as tf.SymbolicTensor;

  x = transformer(Object.assign({}, config, { name: config.name + '/transformer' }))
    .apply(x) as tf.SymbolicTensor;

  x = tf.layers.layerNormalization({ name: config.name + '/ln_f', epsilon: 1e-5 })
    .apply(x) as tf.SymbolicTensor;

  x = eot({ name: config.name + '/eot' })
    .apply([x, input]) as tf.SymbolicTensor; // [batch, nEmbd]

  x = tf.layers.dense({ name: config.name + '/text_projection', units: config.nEmbd, useBias: false })
    .apply(x) as tf.SymbolicTensor;

  return tf.model({ inputs: input, outputs: x });
}

/**
 * Custom layer that takes outputs from text and vision models and computes
 * the similarity between them scaling the logits.
 * @returns
 */
export function similarity(config: BaseConfig) { return new Similarity(config) }
export class Similarity extends tf.layers.Layer {
  private logitScale!: tf.LayerVariable;

  constructor(config: BaseConfig) {
    super(config);
  }

  build(): void {
    this.logitScale = this.addWeight(
      'scale',
      [1],
      'float32',
      tf.initializers.constant({ value: Math.log(1 / 0.07) })
    );
  }

  computeOutputShape(inputShape: tf.Shape[]): tf.Shape[] {
    return inputShape;
  }

  call(input: tf.Tensor[], kwargs: any): tf.Tensor[] {
    return tf.tidy(() => {
      // Compute norms
      const visNorm = tf.norm(input[0], 2, -1, true);
      const textNorm = tf.norm(input[1], 2, -1, true);
      // Normalize
      const visFeatures = tf.div(input[0], visNorm);
      const textFeatures = tf.div(input[1], textNorm);
      // Compute logits
      const logitScale = tf.exp(this.logitScale.read());
      const logitsPerImage = tf.mul(
        logitScale,
        tf.matMul(visFeatures, textFeatures, false, true)
      );
      const logitsPerText = tf.transpose(logitsPerImage);
      return [logitsPerImage, logitsPerText];
    });
  }
  static className = 'Similarity';
}
tf.serialization.registerClass(Similarity)

export function CLIPModel (
  config: CLIPConfig,
  textModel: tf.LayersModel,
  visionModel: tf.LayersModel,
  similarity: tf.layers.Layer
): tf.LayersModel {
  const textInput = tf.input({ shape: [config.blockSize] });
  const visionInput = tf.input({ shape: [config.inputResolution, config.inputResolution, 3] });

  const textOutput = textModel.apply(textInput) as tf.SymbolicTensor;
  const visionOutput = visionModel.apply(visionInput) as tf.SymbolicTensor;

  const [logitsPerImage, logitsPerText] = similarity.apply([visionOutput, textOutput]) as tf.SymbolicTensor[];

  return tf.model({ inputs: [textInput, visionInput], outputs: [logitsPerImage, logitsPerText] });
}

interface CLIPConfig {
  inputResolution: number;
  patchSize: number;
  visionWidth: number;
  visionLayers: number;
  nEmbd: number;
  nHead: number;
  nLayer: number;
  blockSize: number;
  vocabSize: number;
  debug?: boolean;
}
export class CLIP {
  config: CLIPConfig;
  visual: tf.LayersModel;
  text: tf.LayersModel;
  similarity: tf.layers.Layer;
  clip: tf.LayersModel;

  constructor(config: CLIPConfig) {
    if (config.visionWidth % 64 !== 0) throw new Error('Vision width must be divisible by 64');
    this.config = config;

    this.visual = visionTransformer({
      name: 'vit',
      inputResolution: config.inputResolution,
      patchSize: config.patchSize,
      nEmbd: config.visionWidth,
      nLayer: config.visionLayers,
      nHead: config.visionWidth / 64,
      outputDim: config.nEmbd,
      debug: config.debug,
      joint: false,
    })

    this.text = CLIPTextModel({
      name: 'text',
      vocabSize: config.vocabSize,
      nEmbd: config.nEmbd,
      nHead: config.nHead,
      nLayer: config.nLayer,
      blockSize: config.blockSize,
      debug: config.debug,
      residDrop: 0,
      dropout: 0,
      joint: false,
      isCausal: true,
      activation: 'quickgelu'
    })

    this.similarity = similarity({
      name: 'similarity'
    });

    this.clip = CLIPModel(config, this.text, this.visual, this.similarity);
  }

  encodeImage(img: tf.Tensor): tf.Tensor {
    let imgPrep: tf.Tensor = transform(img as tf.Tensor3D, this.config.inputResolution);
    const imgFeatures: tf.Tensor = this.visual.apply(imgPrep) as tf.Tensor;
    return imgFeatures;
  }

  encodeText(text: tf.Tensor): tf.Tensor {
    const textFeatures: tf.Tensor = this.text.apply(text) as tf.Tensor;
    return textFeatures;
  }

  apply([text, img]: [tf.Tensor, tf.Tensor]): tf.Tensor[] {
    const [logitsPerImage, logitsPerText] = this.clip.apply([text, img]) as tf.Tensor[];
    return [logitsPerImage, logitsPerText];
  }
  predict([text, img]: [tf.Tensor, tf.Tensor]): tf.Tensor[] {
    return this.apply([text, img]);
  }

  static maps: { [key: string]: [string, string][] } = {
    'clip-vit-base-h5': [
      ['vit/proj', 'visual_projection'],
      ['vit', 'vision_model'],
      ['text/text_projection', 'text_projection'],
      ['text/', 'text_model/'],
      ['transformer', 'encoder'],
      ['conv1', 'embeddings/patch_embedding'],
      ['token_embedding/embeddings', 'embeddings/token_embedding/weight'],
      ['positional_embedding/weight', 'embeddings/position_embedding/embeddings'],
      ['/h_', '/layers_._'],
      ['/ln_f', '/final_layer_norm'],
      ['/ln_pre', '/pre_layrnorm'],
      ['/ln_post', '/post_layernorm'],
      ['/ln_', '/layer_norm'],
      ['attn/c_attn_q', 'attn/q_proj'],
      ['attn/c_attn_k', 'attn/k_proj'],
      ['attn/c_attn_v', 'attn/v_proj'],
      ['attn/c_proj', 'attn/out_proj'],
      ['/attn/', '/self_attn/'],
      ['mlp/c_fc', 'mlp/fc1'],
      ['mlp/c_proj', 'mlp/fc2'],
      ['class_embedding/weight', 'embeddings/class_embedding'],
      ['similarity/scale', 'logit_scale'],
    ]
  }

  static transposes: { [key: string]: [string, number[]][] } = {}

  // weights are dict of name -> tf.Tensor or h5 group
  static buildModel(
    weights: any,
    source: string = 'clip-vit-base-h5',
    cb: (stats: StatsWeights) => void = (stats) => { }
  ): CLIP {
    console.log('Building model from', source)
    const weightMap = source in CLIP.maps ? CLIP.maps[source] : [];
    const weightTranspose = source in CLIP.transposes ? CLIP.transposes[source] : [];
    const visConv = getWeight(weights, 'vit/conv1', weightMap, weightTranspose, 'h5')
    const visPos = getWeight(weights, 'vit/positional_embedding/weight', weightMap, weightTranspose, 'h5')
    const visLayers = getWeight(weights, 'vit/transformer', weightMap, weightTranspose, 'h5')
    const textEmb = getWeight(weights, 'text/token_embedding/embeddings', weightMap, weightTranspose, 'h5')
    const textPos = getWeight(weights, 'text/positional_embedding/weight', weightMap, weightTranspose, 'h5')
    const textLayers = getWeight(weights, 'text/transformer', weightMap, weightTranspose, 'h5')
    const gridSize = Math.round(Math.sqrt(visPos.shape[0] - 1))
    const patchSize = visConv.shape[0]
    const config = {
      'patchSize': patchSize,
      'inputResolution': patchSize * gridSize,
      'gridSize': gridSize,
      'visionWidth': visConv.shape[3],
      'visionLayers': visLayers / 16, // 16 is the number of TF layers in a block
      'blockSize': textPos.shape[0],
      'nEmbd': textPos.shape[1],
      'nHead': textPos.shape[1] / 64,
      'vocabSize': textEmb.shape[0],
      'nLayer': textLayers / 16,
      'debug': false,
      'joint': false
    }
    console.log('Config', config)
    const model = new CLIP(config)
    console.log('Setting weights...')
    // setWeights(model.visual, weights, weightMap, weightTranspose, source.includes('h5') ? 'h5' : 'object', cb)
    // setWeights(model.text, weights, weightMap, weightTranspose, source.includes('h5') ? 'h5' : 'object', cb)
    // setWeights(model.similarity, weights, weightMap, weightTranspose, source.includes('h5') ? 'h5' : 'object', cb)
    setWeights(model.clip, weights, weightMap, weightTranspose, source.includes('h5') ? 'h5' : 'object', cb)
    return model
  }
}