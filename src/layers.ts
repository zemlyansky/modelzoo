import * as tf from '@tensorflow/tfjs';

export interface BaseConfig {
  name?: string
}

/**
 * Logging layer
 *
 * Can be added to a model to log the shape and first/last values of the input
 */

export function logLayer (config: BaseConfig) { return new LogLayer(config) }
export class LogLayer extends tf.layers.Layer {
  private config: BaseConfig;

  constructor(config: BaseConfig) {
    super(config);
    this.config = config;
  }

  computeOutputShape(inputShape: tf.Shape): tf.Shape {
    return inputShape;
  }

  call(input: tf.Tensor | tf.Tensor[], kwargs: any = {}): tf.Tensor {
    return tf.tidy(() => {
      if (Array.isArray(input)) {
        input = input[0];
      }
      this.invokeCallHook(input, kwargs);
      const x = tf.util.flatten(input.arraySync());
      console.log(this.config.name + ">", input.shape, x[0], x[x.length - 1]);
      return input;
    });
  }

  static className: string = "LogLayer";
}
tf.serialization.registerClass(LogLayer)

/**
 * Range layer
 */
export function rangeLayer (config: BaseConfig) { return new Range(config) }
export class Range extends tf.layers.Layer {
  constructor(config: BaseConfig) {
    super(config);
  }

  computeOutputShape(inputShape: tf.Shape): tf.Shape {
    return inputShape;
  }

  call(input: tf.Tensor | tf.Tensor[], kwargs: any): tf.Tensor {
    return tf.tidy(() => {
      if (Array.isArray(input)) {
        input = input[0];
      }
      this.invokeCallHook(input, kwargs);
      const [, T] = input.shape;
      const range = tf.reshape(tf.range(0, T, 1, "int32"), [1, T]); // .tile([B, 1])
      return range;
    });
  }

  static className: string = "Range";
}
tf.serialization.registerClass(Range)

export function eot(config: BaseConfig) { return new EOT(config) }
export class EOT extends tf.layers.Layer {
  // Layers that get x of size: [batch, length, nEmbd] and tokens of size [batch, length]
  // And returns: x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)]
  constructor(config: BaseConfig) {
    super(config);
  }

  computeOutputShape([xShape, tokensShape]) {
    return [xShape[0], xShape[2]]; // [batch, nEmbd]
  }

  call(inputs, kwargs) {
    return tf.tidy(() => {
      let [x, tokens] = inputs;

      if (Array.isArray(x)) {
        x = x[0];
      }
      if (Array.isArray(tokens)) {
        tokens = tokens[0];
      }

      this.invokeCallHook([x, tokens], kwargs);

      // Get the index of the token with the maximum ID for each sequence in the batch
      const maxTokenIndices = tf.argMax(tokens, -1);
      x = logLayer({ name: "eot/pre_gather" }).apply(x) as tf.SymbolicTensor;
      x = tf.gather(x, maxTokenIndices, 1, 1);
      x = logLayer({ name: "eot/post_gather" }).apply(x) as tf.SymbolicTensor;
      // x = x.squeeze([1]); // Not needed if passing that extra `1` to gather
      // x = logLayer({ name: "eot/post_squeeze" }).apply(x) as tf.SymbolicTensor;
      return x;
    });
  }

  static className = "EOT";
}

tf.serialization.registerClass(EOT);

/**
 * Positional embedding layer
 */
export function positionalEmbedding(config: BaseConfig) { return new PositionalEmbedding(config) }
export class PositionalEmbedding extends tf.layers.Layer {
  private positionalEmbedding!: tf.LayerVariable;

  constructor(config: BaseConfig) {
    super(config);
  }

  build(inputShape: number[]): void {
    const nEmbd = inputShape.slice(-1)[0];
    const scale = Math.pow(nEmbd, -0.5);
    this.positionalEmbedding = this.addWeight(
      'weight',
      inputShape.slice(-2),
      'float32',
      tf.initializers.randomNormal({mean: 0, stddev: scale})
    );
  }

  computeOutputShape(inputShape: tf.Shape): tf.Shape {
    return inputShape;
  }

  call(input: tf.Tensor | tf.Tensor[], kwargs: any): tf.Tensor {
    return tf.tidy(() => {
      if (Array.isArray(input)) {
        input = input[0];
      }
      this.invokeCallHook(input, kwargs);
      return tf.add(input, this.positionalEmbedding.read());
    });
  }

  static className = 'PositionalEmbedding';
}
tf.serialization.registerClass(PositionalEmbedding)

/**
 * Class embedding layer
 */
export function classEmbedding(config: BaseConfig) { return new ClassEmbedding(config) }
export class ClassEmbedding extends tf.layers.Layer {
  private classEmbedding!: tf.LayerVariable;

  constructor(config: BaseConfig) {
    super(config);
  }

  build(inputShape: number[]): void {
    // Initialize the class embedding
    const nEmbd = inputShape.slice(-1)[0];
    const scale = Math.pow(nEmbd, -0.5);
    this.classEmbedding = this.addWeight(
      'weight',
      inputShape.slice(-1),
      'float32',
      tf.initializers.randomNormal({mean: 0, stddev: scale})
    );
  }

  computeOutputShape(inputShape: tf.Shape): tf.Shape {
    // Adding 1 to the sequence length dimension
    const newShape = [...inputShape] as number[];
    newShape[1] += 1;
    return newShape;
  }

  call(input: tf.Tensor | tf.Tensor[], kwargs: any): tf.Tensor {
    return tf.tidy(() => {
      if (Array.isArray(input)) {
        input = input[0];
      }
      this.invokeCallHook(input, kwargs);

      const inputShape = input.shape as number[];
      const embShape = [inputShape[0], 1, inputShape[2]];

      const zeros = tf.zeros(embShape, input.dtype);
      const emb = tf.add(zeros, this.classEmbedding.read());

      return tf.concat([emb, input], 1);
    });
  }

  static className = 'ClassEmbedding';
}
tf.serialization.registerClass(ClassEmbedding)

export function slice(begin: number[], size: number[]) { return new Slice(begin, size) }
export class Slice extends tf.layers.Layer {
  private begin: number[];
  private size: number[];

  constructor(begin: number[], size: number[]) {
    super({});
    this.begin = begin;
    this.size = size;
  }

  computeOutputShape(inputShape: tf.Shape): tf.Shape {
    // Compute the output shape based on the slicing operation.
    const outputShape = inputShape.slice();
    for (let axis = 0; axis < outputShape.length; axis++) {
      if (inputShape[axis] != null && this.begin[axis] != null) {
        outputShape[axis] = this.size[axis];
      }
    }
    return outputShape;
  }

  call(input: tf.Tensor | tf.Tensor[], kwargs: any): tf.Tensor {
    return tf.tidy(() => {
      if (Array.isArray(input)) {
        input = input[0];
      }
      this.invokeCallHook(input, kwargs);
      const begin = [0, ...this.begin];
      const size = [-1, ...this.size];
      return tf.slice(input, begin, size);
    });
  }

  static className = 'Slice';
}
tf.serialization.registerClass(Slice)

/**
 * GELU activation function
 *
 * y = x * 0.5 * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
 */
export function gelu() { return new GELU() }
export class GELU extends tf.layers.Layer {
  constructor() {
    super({});
  }

  computeOutputShape(inputShape: tf.Shape): tf.Shape {
    return inputShape;
  }

  call(input: tf.Tensor | tf.Tensor[], kwargs: any): tf.Tensor {
    return tf.tidy(() => {
      // In functional API, input is an array of tensors
      // So we need to get the first element (the actual input)
      // Add a check as here:
      // https://github.com/tensorflow/tfjs-examples/blob/master/custom-layer/custom_layer.js
      if (Array.isArray(input)) {
        input = input[0];
      }
      this.invokeCallHook(input, kwargs);
      const cdf = tf.mul(
        0.5,
        tf.add(
          1,
          tf.tanh(
            tf.mul(
              tf.sqrt(tf.div(2, Math.PI)),
              tf.add(input, tf.mul(0.044715, tf.pow(input, 3))),
            ),
          ),
        ),
      );
      return tf.mul(input, cdf);
    });
  }

  static className: string = "GELU";
}
tf.serialization.registerClass(GELU)

/**
 * QuickGELU activation function
 *
 * y = x * sigmoid(1.702 * x)
 */
export function quickgelu() { return new QuickGELU() }
export class QuickGELU extends tf.layers.Layer {
  constructor() {
    super({})
  }

  computeOutputShape(inputShape: tf.Shape): tf.Shape {
    return inputShape
  }

  call(input: tf.Tensor | tf.Tensor[], kwargs: any): tf.Tensor {
    return tf.tidy(() => {
      if (Array.isArray(input)) {
        input = input[0]
      }
      this.invokeCallHook(input, kwargs)
      return tf.mul(input, tf.sigmoid(tf.mul(1.702, input)))
    })
  }

  static get className() {
    return 'QuickGELU'
  }
}
tf.serialization.registerClass(QuickGELU)

/**
 * Multihead attention layer
 */
export interface MultiheadAttentionConfig {
  blockSize: number;
  nEmbd: number;
  nHead: number;
  dropout: number;
  mask?: tf.Tensor | null | undefined;
  name?: string;
  debug?: boolean;
  isCausal?: boolean;
  bias?: boolean;
  joint?: boolean; // If true, use a single dense layer for q, k, v projections
}
export function multiheadAttention (config: MultiheadAttentionConfig) {
  return new MultiheadAttention(config)
}
export class MultiheadAttention extends tf.layers.Layer {
  private config: MultiheadAttentionConfig;
  private nEmbd: number;
  private nHead: number;
  private dropout: number;
  private bias: boolean | undefined;
  private isCausal: boolean | undefined;
  private mask: tf.Tensor | null | undefined;
  private cAttnKernels!: tf.LayerVariable[];
  private cAttnBiases!: tf.LayerVariable[];
  private cProjKernel!: tf.LayerVariable;
  private cProjBias!: tf.LayerVariable;

  constructor(config: MultiheadAttentionConfig) {
    super(config);
    this.config = Object.assign({
      name: "attn",
      bias: true,
      debug: false,
      isCausal: false,
      joint: true,
    }, config);

    // Config
    this.nEmbd = this.config.nEmbd;
    this.nHead = this.config.nHead;
    this.dropout = this.config.dropout;
    this.bias = this.config.bias;
    this.isCausal = this.config.isCausal;

    if (this.config.isCausal) {
      this.mask = tf.linalg.bandPart(
        tf.ones([this.config.blockSize, this.config.blockSize]),
        -1,
        0,
      );
    } else if (this.config.mask) {
      this.mask = this.config.mask;
    }
  }

  build(_inputShape: tf.Shape): void {
    if (this.config.joint) {
      this.cAttnKernels = [this.addWeight(
        "c_attn/kernel",
        [this.nEmbd, 3 * this.nEmbd],
        "float32",
        tf.initializers.glorotNormal({}),
      )];
      this.cAttnBiases = [this.addWeight(
        "c_attn/bias",
        [3 * this.nEmbd],
        "float32",
        tf.initializers.zeros(),
      )];
    } else {
      this.cAttnKernels = [];
      this.cAttnBiases = [];
      ['q', 'k', 'v'].forEach((name) => {
        this.cAttnKernels.push(this.addWeight(
          `c_attn_${name}/kernel`,
          [this.nEmbd, this.nEmbd],
          "float32",
          tf.initializers.glorotNormal({}),
        ));
        this.cAttnBiases.push(this.addWeight(
          `c_attn_${name}/bias`,
          [this.nEmbd],
          "float32",
          tf.initializers.zeros(),
        ));
      });
    }
    this.cProjKernel = this.addWeight(
      "c_proj/kernel",
      [this.nEmbd, this.nEmbd],
      "float32",
      tf.initializers.glorotNormal({}),
    );
    this.cProjBias = this.addWeight(
      "c_proj/bias",
      [this.nEmbd],
      "float32",
      tf.initializers.zeros(),
    );
  }

  computeOutputShape(inputShape: tf.Shape): tf.Shape {
    // console.log('computeOutputShape', inputShape)
    return inputShape;
    // return [null, this.blockSize, this.nEmbd]
  }

  getConfig() {
    // This is neeed to save and load the model
    // When the model is saved, the config is saved with it
    // When the model is loaded, the config is used to create a new instance of the layer
    const config = super.getConfig();
    return Object.assign({}, config, this.config);
  }

  /**
   * Pure dense function
   * @param x - input tensor
   * @param kernel - kernel tensor
   * @param bias - bias tensor
   * @returns result
   */
  static dense(
    x: tf.Tensor,
    kernel: tf.LayerVariable,
    bias?: tf.LayerVariable | null,
  ): tf.Tensor {
    // Direct application of matMul to x and kernel throws:
    // > Error in gradient for op BatchMatMul.
    // > The gradient of input 'b' has shape '16,48,48',
    // > which does not match the shape of the input '48,48'
    // Two solutions worked:
    // 1. Use tf.layers.dense but reassign kernel and bias
    // 2. Use tf.matMul but expandDims and tile kernel (current)
    // Another option, of course, is to separate attention logic
    // from trainable weights completely and use tf.layers.dense
    // inside a model definition. I was not able to define fully
    // function regular dense layers inside a custom layer.
    // Something related to how weights are loaded with this.kernel
    // and duplicating names
    const k = kernel.read().expandDims(0).tile([x.shape[0], 1, 1]);
    const m = tf.matMul(x, k);
    if (bias) {
      return tf.add(m, bias.read());
    }
    return m;
  }

  /**
   * Pure attention function
   * */
  static attention(
    input: tf.Tensor[],
    nHead: number,
    cAttnKernels: tf.LayerVariable[],
    cAttnBiases: tf.LayerVariable[] | null,
    cProjKernel: tf.LayerVariable,
    cProjBias: tf.LayerVariable | null,
    dropout: number = 0,
    training: boolean = false,
    mask?: tf.Tensor | null | undefined,
    debug?: boolean,
  ): tf.Tensor {
    let q: tf.Tensor, k: tf.Tensor, v: tf.Tensor;
    if ((input.length === 1) && (cAttnKernels.length === 1)) {
      [q, k, v] = tf.split(MultiheadAttention.dense(
        input[0],
        cAttnKernels[0],
        cAttnBiases ? cAttnBiases[0] : null,
      ), 3, -1)
    } else if ((input.length === 3) && (cAttnKernels.length === 3)) {
      [q, k, v] = input.map((x, i) => MultiheadAttention.dense(
        x,
        cAttnKernels[i],
        cAttnBiases ? cAttnBiases[i] : null,
      ));
    } else if ((input.length === 1) && (cAttnKernels.length === 3)) {
      [q, k, v] = cAttnKernels.map((w, i) => MultiheadAttention.dense(
        input[0],
        w,
        cAttnBiases ? cAttnBiases[i] : null,
      ));
    } else if ((input.length === 3) && (cAttnKernels.length === 1)) {
      [q, k, v] = input.map((x, i) => MultiheadAttention.dense(
        x,
        cAttnKernels[0],
        cAttnBiases ? cAttnBiases[0] : null,
      ));
    } else {
      throw new Error("Invalid input");
    }

    const [B, T, C] = k.shape;

    if (debug) {
      new LogLayer({ name: "att_x" }).call(input);
      new LogLayer({ name: "att_q_before" }).call(q);
      new LogLayer({ name: "att_k_before" }).call(k);
      new LogLayer({ name: "att_v_before" }).call(v);
    }

    const splitHeads = (x: tf.Tensor) =>
      tf.transpose(
        tf.reshape(x, [B, T, nHead, C / nHead]),
        [0, 2, 1, 3],
      );

    q = splitHeads(q);
    k = splitHeads(k);
    v = splitHeads(v);

    if (debug) {
      new LogLayer({ name: "att_q_after" }).call(q);
      new LogLayer({ name: "att_k_after" }).call(k);
      new LogLayer({ name: "att_v_after" }).call(v);
    }

    // causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    let att = tf.mul(
      tf.matMul(q, k, false, true),
      tf.div(1, tf.sqrt(tf.cast(k.shape[k.shape.length - 1], "float32"))),
    );

    if (mask) {
      mask = mask.slice([0, 0], [T, T]);
      att = tf.add(att, tf.mul(tf.sub(1, mask), -1e9));
    }

    att = tf.softmax(att, -1);
    att = training ? tf.dropout(att, dropout) : att;
    if (debug) {
      new LogLayer({ name: "> att_softmax" }).call(att);
    }

    let y = tf.matMul(att, v);
    if (debug) {
      new LogLayer({ name: "att_yv" }).call(y);
    }

    y = tf.transpose(y, [0, 2, 1, 3]);
    y = tf.reshape(y, [B, T, C]);
    y = MultiheadAttention.dense(y, cProjKernel, cProjBias)
    y = training ? tf.dropout(y, dropout) : y;
    if (debug) {
      new LogLayer({ name: "att_y" }).call(y);
    }
    return y;
  }

  call(input: tf.Tensor[], kwargs: any): tf.Tensor {
    return tf.tidy(() => {
      if (!Array.isArray(input)) {
        input = [input];
      }
      this.invokeCallHook(input, kwargs);
      return MultiheadAttention.attention(
        input,
        this.nHead,
        this.cAttnKernels,
        this.bias ? this.cAttnBiases : null,
        this.cProjKernel,
        this.bias ? this.cProjBias : null,
        this.dropout,
        kwargs["training"],
        this.mask,
        this.config.debug,
      );
    });
  }

  static className: string = "MultiheadAttention";
}
tf.serialization.registerClass(MultiheadAttention)

/**
 *  MLP layer
 */
export interface MLPConfig {
  blockSize: number;
  nEmbd: number;
  name?: string;
  activation?: string;
  residDrop: number;
}

export const mlp = MLP;
export function MLP(conf: MLPConfig): tf.LayersModel {
  const config = Object.assign({
    name: "mlp",
    activation: "gelu",
  }, conf);
  const inputs = tf.input({ shape: [config.blockSize, config.nEmbd] });
  let x;
  x = tf.layers
    .dense({
      name: config.name + "/c_fc",
      units: 4 * config.nEmbd,
      inputDim: config.nEmbd,
      inputShape: [config.blockSize, config.nEmbd],
    })
    .apply(inputs);
  switch (config.activation) {
    case "gelu":
      x = new GELU().apply(x);
      break;
    case "quickgelu":
      x = new QuickGELU().apply(x);
      break;
  }
  x = tf.layers
    .dense({
      name: config.name + "/c_proj",
      units: config.nEmbd,
      inputDim: 4 * config.nEmbd,
      inputShape: [config.blockSize, 4 * config.nEmbd],
    })
    .apply(x);
  x = tf.layers
    .dropout({
      name: config.name + "/drop",
      rate: config.residDrop,
    })
    .apply(x) as tf.SymbolicTensor;
  return tf.model({ inputs: inputs, outputs: x });
}

/**
 *  Block layer
 */
export interface BlockConfig extends MLPConfig, MultiheadAttentionConfig {}
export const block = Block;
export function Block(conf: BlockConfig): tf.LayersModel {
  const config = Object.assign({
    name: "h",
    activation: "gelu",
  }, conf);
  const inputs = tf.input({ shape: [config.blockSize, config.nEmbd] });
  let x1: tf.SymbolicTensor, x2: tf.SymbolicTensor;
  // Setting epsilon to 1e-5 for LayerNorms to be consistent with PyTorch
  x1 = tf.layers.layerNormalization({ name: config.name + "/ln_1", epsilon: 1e-5 })
    .apply(inputs) as tf.SymbolicTensor;
  if (config.debug) {
    x1 = new LogLayer({ name: config.name + "/ln_1_log" }).apply(
      x1,
    ) as tf.SymbolicTensor;
  }
  x1 = multiheadAttention(Object.assign({}, config, { name: config.name + "/attn" }))
    .apply(x1) as tf.SymbolicTensor;
  x1 = tf.layers.add()
    .apply([inputs, x1]) as tf.SymbolicTensor;
  // MLP
  x2 = tf.layers
    .layerNormalization({ name: config.name + "/ln_2", epsilon: 1e-5 })
    .apply(x1) as tf.SymbolicTensor;
  x2 = mlp(Object.assign({}, config, { name: config.name + "/mlp" }))
    .apply(x2) as tf.SymbolicTensor;
  x2 = tf.layers.add()
    .apply([x1, x2]) as tf.SymbolicTensor;
  return tf.model({
    name: config.name,
    inputs: inputs,
    outputs: x2,
  }) as tf.LayersModel;
}

/*

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

*/

/**
 *  Transformer layer
 */
export interface TransformerConfig extends BlockConfig {
  nLayer: number;
}
export const transformer = Transformer;
export function Transformer(config: TransformerConfig): tf.LayersModel {
  const inputs = tf.input({ shape: [config.blockSize, config.nEmbd] });
  config = Object.assign({
    name: 'transformer',
  }, config);
  let x: tf.SymbolicTensor = inputs;
  for (let i = 0; i < config.nLayer; i++) {
    x = Block(Object.assign({}, config, {
      'name': config.name + '/h_' + i
    })).apply(x) as tf.SymbolicTensor;
  }
  return tf.model({
    name: "transformer",
    inputs: inputs,
    outputs: x,
  }) as tf.LayersModel;
}