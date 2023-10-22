import * as tf from '@tensorflow/tfjs';
import { LogLayer, Range, Block, BlockConfig } from '../layers';
import { AdamW, clipByGlobalNormObj } from "../optimizers";

export async function train(
  model: tf.LayersModel,
  ds: tf.data.Dataset<tf.TensorContainer>,
  config: {
    epochs?: number;
    maxIter?: number;
    batchSize?: number;
    shuffle?: boolean | number | "batch";
    lr?: number;
    weightDecay?: boolean | number;
    callbacks?: any[];
    verbose?: boolean;
  } = {},
) {
  if (undefined === config.batchSize) {
    config.batchSize = 16;
  }

  if (undefined === config.lr) {
    config.lr = 6e-4;
  }

  if (undefined === config.shuffle) {
    config.shuffle = true;
  }

  if (undefined === config.weightDecay) {
    config.weightDecay = false;
  }

  if (undefined === config.callbacks) {
    config.callbacks = [];
  }

  if (config.shuffle === true) {
    ds = ds.shuffle(config.batchSize * 10);
  } else if (config.shuffle === "batch") {
    ds = ds.shuffle(config.batchSize);
  } else if (false !== config.shuffle && !isNaN(config.shuffle)) {
    ds = ds.shuffle(config.shuffle);
  }
  ds = ds.batch(config.batchSize);

  const includeInWeightDecay: string[] = [];
  const excludeFromWeightDecay: string[] = [];

  if (config.weightDecay === true) {
    config.weightDecay = 1e-4;
  }
  let opt = tf.train.adam(config.lr);
  if (config.weightDecay) {
    model["getNamedWeights"]().forEach((v) => {
      if (
        v.name.includes("bias") ||
        v.name.includes("normalization") ||
        v.name.includes("emb")
      ) {
        excludeFromWeightDecay.push(v.name);
      } else {
        includeInWeightDecay.push(v.name);
      }
    });
    opt = new AdamW({
      learningRate: config.lr,
      weightDecayRate: config.weightDecay,
      includeInWeightDecay,
      excludeFromWeightDecay,
    });
  }

  let epoch = 1;
  let iteration = 1;
  let iterator = await ds.iterator();

  // eslint-disable-next-line no-constant-condition
  while (true) {
    let next = await iterator.next();
    if (next.done) {
      epoch++;
      if (config.epochs && epoch > config.epochs) {
        break;
      }
      iterator = await ds.iterator();
      next = await iterator.next();
    }
    const { x, y } = next.value;

    // Keep loss for reporting
    let loss: tf.Tensor<tf.Rank> = null as any;
    const optFunc = (): tf.Scalar => {
      const logits = model.apply(x);
      loss = tf.keep(tf.losses.softmaxCrossEntropy(y, logits));
      return loss.asScalar();
    };
    tf.tidy(() => {
      const { grads } = opt.computeGradients(optFunc);
      const gradsClipped = clipByGlobalNormObj(grads, 1);
      opt.applyGradients(gradsClipped);
    });

    const lossVal = await loss.array();
    if (Array.isArray(config.callbacks)) {
      for (const callback of config.callbacks) {
        await callback(model, lossVal, iteration);
      }
    }

    // Dispose everything
    loss.dispose();
    x.dispose();
    y.dispose();

    // Check if we should stop
    iteration++;
    if (config.maxIter && iteration > config.maxIter) {
      break;
    }

    if (config.verbose) {
      console.log("Mem:", tf.memory());
      console.log(`Epoch: ${epoch}, Step: ${iteration}, Loss: ${lossVal}`);
    }

    await new Promise((resolve) => setTimeout(resolve, 1));
  }
}

interface GPTConfig extends BlockConfig {
  name?: string;
  bias?: boolean;
  debug?: boolean;
  tokEmb?: boolean;
  lmHead?: boolean;
  embdDrop?: number;
  nLayer?: number;
  vocabSize?: number;
  modelType:
    | "gpt2"
    | "gpt2-medium"
    | "gpt2-large"
    | "gpt2-xl"
    | "gpt-mini"
    | "gpt-micro"
    | "gpt-nano";
}

export function GPT(conf: Partial<GPTConfig>): tf.LayersModel {
  const configDefaults = {
    name: "transformer",
    bias: true,
    debug: false,
    tokEmb: true,
    lmHead: true,
  };
  const configModels = {
    gpt2: {
      nLayer: 12,
      nHead: 12,
      nEmbd: 768,
      vocabSize: 50257,
      blockSize: 1024,
    },
    "gpt2-medium": {
      nLayer: 24,
      nHead: 16,
      nEmbd: 1024,
      vocabSize: 50257,
      blockSize: 1024,
    },
    "gpt2-large": {
      nLayer: 36,
      nHead: 20,
      nEmbd: 1280,
      vocabSize: 50257,
      blockSize: 1024,
    },
    "gpt2-xl": {
      nLayer: 48,
      nHead: 25,
      nEmbd: 1600,
      vocabSize: 50257,
      blockSize: 1024,
    },
    "gpt-mini": {
      nLayer: 6,
      nHead: 6,
      nEmbd: 192,
      vocabSize: 50257,
      blockSize: 1024,
    },
    "gpt-micro": {
      nLayer: 4,
      nHead: 4,
      nEmbd: 128,
      vocabSize: 50257,
      blockSize: 1024,
    },
    "gpt-nano": {
      nLayer: 3,
      nHead: 3,
      nEmbd: 48,
      vocabSize: 50257,
      blockSize: 1024,
    },
  };
  // Check if modelType is present in conf
  if (conf.modelType) {
    // If so, check if it's valid
    if (!Object.keys(configModels).includes(conf.modelType)) {
      throw new Error(`Invalid modelType: ${conf.modelType}`);
    }
    // If valid, merge modelConfig with configDefaults
    const modelConfig = configModels[conf.modelType];
    Object.assign(configDefaults, modelConfig);
  }

  const config = Object.assign({}, configDefaults, conf) as Required<GPTConfig>;

  const inputs = tf.input({ shape: [null] });

  const tokEmb = config.tokEmb
    ? tf.layers
        .embedding({
          name: config.name + "/wte",
          inputDim: config.vocabSize as number,
          outputDim: config.nEmbd as number,
          embeddingsInitializer: "zeros",
        })
        .apply(inputs)
    : inputs;

  const range = new Range({}).apply(inputs);
  let posEmb = tf.layers
    .embedding({
      name: config.name + "/wpe",
      inputDim: config.blockSize as number,
      outputDim: config.nEmbd as number,
      embeddingsInitializer: "zeros",
    })
    .apply(range);
  if (config.debug) {
    posEmb = new LogLayer({ name: "posEmb" }).apply(posEmb);
  }

  let x;
  x = tf.layers.add().apply([tokEmb, posEmb] as tf.SymbolicTensor[]);
  x = tf.layers
    .dropout({
      name: "drop",
      rate: config.embdDrop as number,
    })
    .apply(x);
  if (config.debug) {
    x = new LogLayer({ name: "dropadd" }).apply(x);
  }

  for (let i = 0; i < (config.nLayer as number); i++) {
    x = Block(
      Object.assign({}, config, {
        name: config.name + "/h/" + i,
        isCausal: true
      }),
    ).apply(x);
  }
  x = tf.layers
    .layerNormalization({ name: config.name + "/ln_f", epsilon: 1e-5 })
    .apply(x);
  if (config.debug) {
    x = new LogLayer({ name: "fin/ln" }).apply(x);
  }

  if (config.lmHead) {
    x = tf.layers
      .dense({
        name: "lm_head",
        units: config.vocabSize,
        inputDim: config.nEmbd,
        inputShape: [config.blockSize, config.nEmbd],
        useBias: false,
      })
      .apply(x);
  }
  return tf.model({ inputs: inputs, outputs: x } as {
    inputs: tf.SymbolicTensor;
    outputs: tf.SymbolicTensor;
  });
}

// GPTModel

const defaultGenerateConfig: GPTLMHeadModelGenerateConfig = {
  maxNewTokens: 20,
  temperature: 1.0,
  doSample: false,
  topK: 1,
};

function prepareIdx(idx: number[] | tf.Tensor): tf.Tensor {
  tf.tidy(() => {
    // Check if idx is a tensor or an array
    if (idx instanceof tf.Tensor) {
      idx = idx.clone();
    } else {
      idx = tf.tensor(idx);
    }
    // Check data type
    if (idx.dtype !== "int32") {
      idx = idx.toInt();
    }
    // If the shape of idx is 1D, we need to add a dimension
    if (idx.shape.length === 1) {
      idx = idx.expandDims(0);
    }
    tf.keep(idx);
    // keep idx from deletion
  });
  return idx as tf.Tensor;
}

function generateOnce(
  model: tf.LayersModel,
  idx: tf.Tensor,
  config: GPTLMHeadModelGenerateConfig,
) {
  let idxNext: tf.Tensor<tf.Rank> | null = null;
  let timePerToken = performance.now();
  tf.tidy(() => {
    const block_size = model.inputs[0].shape[1] as number;

    const idxCond =
      (idx.shape[1] as number) <= block_size
        ? idx
        : idx.slice([0, -block_size], [-1, -1]);
    // Forward the model to get the logits for the index in the sequence
    const logits = model.predict(idxCond) as tf.Tensor;
    timePerToken = performance.now() - timePerToken;
    // pluck the logits at the final step and scale by desired temperature
    const logitsScaled = logits
      .slice([0, (idx.shape[1] as number) - 1, 0])
      .reshape([logits.shape[0], logits.shape[2] as number])
      .div(tf.scalar(config.temperature));
    // TODO: topK sampling
    // apply softmax to convert logits to (normalized) probabilities
    const probs = logitsScaled.softmax(-1) as tf.Tensor<tf.Rank.R2>;
    // either sample from the distribution or take the most likely element
    if (config.doSample) {
      idxNext = tf.multinomial(probs, 1);
    } else {
      idxNext = probs.argMax(-1);
      idxNext = idxNext.expandDims(1);
    }
    tf.keep(idxNext);
  });
  return {
    idxNext: idxNext as never as tf.Tensor<tf.Rank>,
    timePerToken,
  };
}

export function generateSync(
  model: tf.LayersModel,
  idx: tf.Tensor,
  conf: GPTLMHeadModelGenerateConfig,
  callback?: (data: GPTLMHeadModelGenerateCallbackData) => void,
) {
  const config = Object.assign({}, defaultGenerateConfig, conf);
  idx = prepareIdx(idx);
  for (let step = 0; step < config.maxNewTokens; step++) {
    const { idxNext, timePerToken } = generateOnce(model, idx, config);
    const idxNew = idx.concat(idxNext, 1);
    tf.dispose(idx);
    idx = idxNew;
    const idxNextArr = idxNext.arraySync() as number[];
    tf.dispose(idxNext);
    if (callback) {
      callback({ idxNext: idxNextArr, timePerToken: timePerToken });
    }
  }
  const idxArr = idx.arraySync();
  tf.dispose(idx);
  return idxArr as number[];
}

export async function generate(
  model: tf.LayersModel,
  idx: number[] | tf.Tensor,
  conf: GPTLMHeadModelGenerateConfig,
  callback?: (data: GPTLMHeadModelGenerateCallbackData) => void,
) {
  const config = Object.assign({}, defaultGenerateConfig, conf);
  idx = prepareIdx(idx);
  for (let step = 0; step < config.maxNewTokens; step++) {
    const { idxNext, timePerToken } = generateOnce(model, idx, config);
    const idxNew = idx.concat(idxNext, 1) as tf.Tensor;
    tf.dispose(idx);
    idx = idxNew;
    const idxNextArr = (await idxNext.array()) as number[];
    tf.dispose(idxNext);
    if (callback) {
      callback({ idxNext: idxNextArr, timePerToken: timePerToken });
    }
  }
  const idxArr = await idx.array();
  tf.dispose(idx);
  return idxArr as number[];
}

class GPTModel_ {
  protected config: GPTConfig;
  protected model: tf.LayersModel;

  constructor(config: GPTConfig) {
    this.config = config;
    this.model = GPT(config);
  }

  async load(weights: tf.NamedTensorMap) {
    await this.model.loadWeights(weights);
  }

  async save(modelPath: string) {
    await this.model.save(modelPath);
  }

  apply(inputs: tf.Tensor | tf.Tensor[]) {
    return this.model.apply(inputs);
  }

  predict(inputs: tf.Tensor | tf.Tensor[]) {
    return this.model.predict(inputs);
  }
}
// GPTLMHeadModel

interface GPTLMHeadModelGenerateConfig {
  maxNewTokens: number;
  temperature: number;
  topK: number;
  doSample: boolean;
}

interface GPTLMHeadModelGenerateCallbackData {
  idxNext: number[];
  timePerToken: number;
}

type GPTLMHeadModelGenerateCallback = (
  data: GPTLMHeadModelGenerateCallbackData,
) => void | Promise<void>;


export class GPTLMHeadModel extends GPTModel_ {
  constructor(config: GPTConfig) {
    super(config);
  }

  async train(dataset: any, config: any): Promise<void> {
    await train(this.model, dataset, config);
  }

  async generate(
    idx: tf.Tensor,
    conf: GPTLMHeadModelGenerateConfig,
    callback?: (data: GPTLMHeadModelGenerateCallbackData) => void,
  ): Promise<number[]> {
    return await generate(this.model, idx, conf, callback);
  }

  generateSync(
    idx: tf.Tensor,
    conf: GPTLMHeadModelGenerateConfig,
    callback?: (data: GPTLMHeadModelGenerateCallbackData) => void,
  ): number[] {
    return generateSync(this.model, idx, conf, callback);
  }
}
