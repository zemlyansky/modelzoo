import * as tf from "@tensorflow/tfjs";

export function convertMinGPTConfig(config) {
  const mapping = {
    n_embd: "nEmbd",
    n_head: "nHead",
    n_layer: "nLayer",
    block_size: "blockSize",
    vocab_size: "vocabSize",
    attn_pdrop: "dropout",
    resid_pdrop: "dropout",
    embd_pdrop: "dropout",
    model_type: "modelType",
  };
  const newConfig = {};
  for (const key in config) {
    if (key in mapping) {
      newConfig[mapping[key]] = config[key];
    } else {
      newConfig[key] = config[key];
    }
  }
  return newConfig;
}

export function convertMinGPTWeights(weights) {
  const newWeights: { [key: string]: tf.Tensor } = {};
  for (const wn in weights) {
    const w = weights[wn];
    let wt = tf.tensor(w);
    // Prepare names
    let wnNew = wn.replace(/\./g, "/");
    if (wnNew.includes("ln_")) {
      wnNew = wnNew.replace("weight", "gamma");
      wnNew = wnNew.replace("bias", "beta");
    } else if (wnNew.includes("wte") || wnNew.includes("wpe")) {
      wnNew = wnNew.replace("weight", "embeddings");
    } else {
      wnNew = wnNew.replace("weight", "kernel");
      wnNew = wnNew.replace("bias", "bias");
    }
    if (wnNew.includes("kernel") && wt.shape.length == 2) {
      wt = wt.transpose();
    }
    newWeights[wnNew] = wt;
  }
  return newWeights;
}

/**
 * Recursively go over H5 tree and collect all keys ~ Object.keys(f)
 * @param f - H5 file
 * @param name - current name
 * @param keys - list of keys
 * @returns - list of keys
 */
function loadH5Keys(f: any, name: string = '', keys: string[] = []) {
  const ks: string[] = f.keys()
  ks.forEach(k => {
      const branch = f.get(k)
      const n = name + '/' + k
      if (branch.keys) {
        loadH5Keys(branch, n, keys)
      } else {
        keys.push(n)
      }
  })
  return keys
}

/**
 * Given a weight object (can be regular JS object or H5) get weight `name`
 * Applying transform from mapName and transpose from mapTranspose
 * @param weights - new model weights
 * @param name - tensor name
 * @param mapName - naming map
 * @param mapTranspose - transpose map
 * @param type - weight type ('object' or 'h5')
 * @returns - tf.Tensor
 */
export function getWeight(
  weights: any,
  name: string,
  mapName: [string, string][] | never[],
  mapTranspose: [string, number[]][] | never[],
  type: string = 'object'
) {
  // Update name
  for (let [from, to] of mapName) {
    name = name.replace(new RegExp(from, 'g'), to)
  }

  // Get all matches
  let keys = type === 'h5'
    ? loadH5Keys(weights)
    : Object.keys(weights)
  let keyMatch = keys.filter(x => x.includes(name))
  if (keyMatch.length == 0) {
    console.error(`Weight ${name} not found in weights`)
    return 0
  } else if (keyMatch.length > 1) {
    // Check if there is a perfect match
    const exactMatch = keyMatch.filter(x => x == name)
    if (exactMatch.length == 1) {
      keyMatch = exactMatch
    } else {
      return keyMatch.length
    }
  }
  let key = keyMatch[0]
  // console.log(`Reading tensor: ${name} from ${key}`)

  // Getting actual weights
  let w = type === 'h5' ? weights.get(key).to_array() : weights[key]
  w = Array.isArray(w) ? tf.tensor(w) : w

  // Transposing
  for (let [from, transpose] of mapTranspose) {
    if (name.includes(from) && (transpose.length == w.shape.length)) {
      w = w.transpose(transpose)
    }
  }
  return w
}

/**
 *
 * @param model
 * @param weights
 * @param mapName
 * @param mapTranspose
 * @param type
 */
export interface StatsWeights {
  i: number,
  total: number,
  name: string,
  shape: number[]
}
export async function setWeights(
  model: tf.LayersModel,
  weights: any,
  mapName: [string, string][],
  mapTranspose: [string, number[]][],
  type: string ='object',
  cb: (stats: StatsWeights) => Promise<void> = async () => {}
) {
  const weightsOld = model.getWeights() as unknown as tf.Variable[]; // TODO: report issue?
  const weightsTotal = weightsOld.length
  // weightsOld.forEach((wOld, i) => {
  for (let i = 0; i < weightsTotal; i++) {
    let wOld = weightsOld[i]
    let wOldName = wOld.name;
    let stats: StatsWeights = {
      i: i,
      total: weightsTotal,
      name: wOldName,
      shape: wOld.shape
    }
    await cb(stats)
    let wNew: tf.Tensor = getWeight(weights, wOldName, mapName, mapTranspose, type)
    wOld.assign(wNew)
  }
}