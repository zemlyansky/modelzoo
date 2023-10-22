const { execSync } = require('child_process')
const tf = require('@tensorflow/tfjs-node')
const { models, layers, utils } = require('../dist/cjs')

const {
  createDataset
} = require('../projects/sort/sort')

const tests = require('./gpt_data.json')

test('GELU', () => {
  const gelu = layers.gelu()
  const inputs = tests.gelu.inputs
  const outputsPY = tests.gelu.outputs
  inputs.forEach((input, i) => {
    const outputJS = gelu.apply(tf.tensor(input)).arraySync()
    const outputPY = outputsPY[i]
    // Check MSE
    const mse = tf.losses.meanSquaredError(tf.tensor(outputPY), tf.tensor(outputJS)).arraySync()
    expect(mse).toBeLessThan(1e-6)
    // Check first elements
    const outputJSFlatten = [outputJS].flat(Infinity)
    const outputPYFlatten = [outputPY].flat(Infinity)
    expect(outputJSFlatten[0]).toBeCloseTo(outputPYFlatten[0], 3)
  })

})

test('GELU (functional)', () => {
  // Test in the functional model
  const gelu = layers.gelu()
  const x = tf.input({shape: [1, 2]})
  const y = gelu.apply(x)
  const model = tf.model({ inputs: x, outputs: y })
  const outputJS = model.predict(tf.tensor([[[-10, 10]]]))
  expect(outputJS.shape[2]).toBe(2)
})

test('Self attention (custom layer full)', () => {
  const inputs = tests.att.inputs
  const outputsPY = tests.att.outputs
  const configs = tests.att.configs
  inputs.forEach((input, i) => {
    const config = configs[i]
    const outputPY = outputsPY[i]
    // Create attention and set all weights to 1 (to have deterministic results)
    input = tf.tensor(input)
    const x = tf.input({shape: [input.shape[1], input.shape[2]]})
    const csa = layers.multiheadAttention(Object.assign(config, {
      isCausal: true
    }))
    const y = csa.apply(x)
    const model = tf.model({ inputs: x, outputs: y })
    model.getWeights().forEach(w => { w.assign(tf.ones(w.shape)) })
    const outputJS = model.predict(input).arraySync()
    // Check MSE
    const mse = tf.losses.meanSquaredError(tf.tensor(outputPY), tf.tensor(outputJS)).arraySync()
    expect(mse).toBeLessThan(1e-6)
    // Check first elements
    const outputJSFlatten = [outputJS].flat(Infinity)
    const outputPYFlatten = [outputPY].flat(Infinity)
    expect(outputJSFlatten[0]).toBeCloseTo(outputPYFlatten[0], 3)
  })
})

test('Self attention (gradients, all)', () => {
  const input = tf.tensor(tests.att_grads.inputs[0])
  const gradsPY = tests.att_grads.grads[0]
  const config = tests.att_grads.configs[0]

  const getGrads = (createAttention) => {
    const x = tf.input({shape: [input.shape[1], input.shape[2]]})
    const y = createAttention(Object.assign(config, { isCausal: true })).apply(x)
    const model = tf.model({ inputs: x, outputs: y })
    model.getWeights().forEach(w => { w.assign(tf.ones(w.shape)) })
    const loss = (preds, labels) => tf.losses.meanSquaredError(preds, labels)
    const f = () => {
      const preds = model.predict(input)
      return loss(preds, input)
    }
    return tf.grads(f)(model.getWeights())
  }

  const gradsJS = getGrads(config => layers.multiheadAttention(config))

  gradsJS.forEach((g, i) => {
    const gpy = gradsPY[i]
    const mse = tf.losses.meanSquaredError(g, tf.tensor(gpy)).arraySync()
    expect(mse).toBeLessThan(1e-3)
  })
})

test('Create model', () => {
  const input = tf.tensor(tests.att_grads.inputs[0])
  const config = tests.att_grads.configs[0]
  config.nLayer = 4
  config.vocabSize = 10
  const model = models.GPT(config)
  model.dispose()
})

function sanitizeName(wn) {
  wn = wn.split('_')
  if (wn[wn.length - 1].length == 1) {
    wn.pop()
  }
  return wn.join('_')
}

test('Load MinGPT model (sorting)', async () => {
  const config = utils.convertMinGPTConfig(tests.model_sort.config)
  const weightsNew = utils.convertMinGPTWeights(tests.model_sort.weights)
  const gpt = new models.GPTLMHeadModel(config)
  gpt.model.getWeights().forEach(w => {
    const wn = sanitizeName(w.name)
    expect(weightsNew[wn]).toBeDefined()
    w.assign(weightsNew[wn])
  })
  const logits = gpt.model.predict(tf.tensor(tests.model_sort.inputs))
  const mse = tf.losses.meanSquaredError(logits, tf.tensor(tests.model_sort.logits)).arraySync()
  expect(mse).toBeLessThan(1e-6)

  // Async generation
  let outputs = await gpt.generate(tests.model_sort.inputs, { maxNewTokens: 6 })
  // outputs = await outputs.array()
  outputs = outputs[0].slice(6)
  outputs.forEach((o, i) => {
    expect(o).toBe(tests.model_sort.outputs[0][i])
  })

  // Sync generation
  outputs = gpt.generateSync(tests.model_sort.inputs, { maxNewTokens: 6, temperature: 0.1, doSample: false })
  outputs = outputs[0].slice(6)
  outputs.forEach((o, i) => {
    expect(o).toBe(tests.model_sort.outputs[0][i])
  })
})

// test('Save / load', async () => {
//   // Call subprocess test_save.js and get the output
//   execSync('node test_save.js')
//   const path = process.cwd() + '/temp/test_model'
//
//   const model1Loaded = await tf.loadLayersModel(`file://${path}_1/model.json`)
//   const model2Loaded = await tf.loadLayersModel(`file://${path}_2/model.json`)
//   const weights1Loaded = model1Loaded.getWeights()
//   const weights2Loaded = model2Loaded.getWeights()
//   // Compare weights
//   weights1Loaded.forEach((w1l, i) => {
//     const w2l = weights2Loaded[i]
//     const mse = tf.losses.meanSquaredError(w1l, w2l).arraySync()
//     expect(mse).toBeLessThan(1e-2)
//   })
// })
//

// test('Train', async () => {
//   const config = {
//     nLayer: 3,
//     nHead: 3,
//     nEmbd: 48,
//     vocabSize: 3,
//     blockSize: 11,
//     dropout: 0.1,
//     debug: false
//   }
//   const train_dataset = createDataset('train')
//   const gpt = new models.GPTLMHeadModel(config)
//   // const nTensorsBefore = tf.memory().numTensors
//   await gpt.train(train_dataset, {epochs: 10, verbose: false}) // Expect this API to change
//   const inputs = [2, 2, 2, 1, 0, 1]
//   const inputsSorted = inputs.sort()
//   const idx = await gpt.generate([inputs], { maxNewTokens: 6 })
//   const outputs = idx[0].slice(6)
//   outputs.forEach((o, i) => {
//     expect(o).toBe(inputsSorted[i])
//   })
//   console.log(`Inputs ${inputs} -> Outputs ${outputs} (expected ${inputsSorted})`)
//   return gpt
// }, 30000)