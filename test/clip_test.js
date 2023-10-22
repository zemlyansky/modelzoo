const fs = require('fs')
const tf = require('@tensorflow/tfjs')
const { SimpleTokenizer } = require('../dist/cjs/tokenizers.js')
const { Block } = require('../dist/cjs/layers.js')
const { VisionTransformer } = require('../dist/cjs/models/vit.js')
const { transform } = require('../dist/cjs/models/clip.js')
const tests = require('./clip_data.json')

function setWeights(model, weights, mapName, mapTranspose) {
  // console.log('Setting weights for:', model.name)
  // console.log('From:', Object.keys(weights))
  // console.log('To:', model.getWeights().map(x => x.name))
  for (let w of model.getWeights()) {
    let name = w.name
    for (let [from, to] of mapName) {
      name = name.replace(new RegExp(from, 'g'), to)
    }
    // check if name is in weights
    // console.log(w.name, '>', name)
    // console.log(weights[name].length)

    let wn = tf.tensor(weights[name])

    // Iterate over mapTranspose and if name matches transpose to that shape
    for (let [from, transpose] of mapTranspose) {
      if (name.includes(from) && (transpose.length == wn.shape.length)) {
        // console.log(`Transposing ${w.name} [${wn.shape}] with ${transpose} (match: ${from})`)
        wn = wn.transpose(transpose)
      }
    }

    // console.log(`Setting ${w.name} to ${wn.shape} from ${name}`)
    w.assign(wn)
  }
}

test('Tokenizer', () => {
  const bpe = fs.readFileSync(process.cwd() + '/bpe_simple_vocab_16e6.txt', 'utf8')
  const tokenizer = new SimpleTokenizer(bpe)
  const tokens = tests['cases'].map(x => tokenizer.tokenize(x))
  tokens.forEach((x, i) => {
    expect(x).toEqual(tests['tokenizer'][i])
  })
})

test('ResidualAttentionBlock', () => {
  const config = {
    'name': 'block',
    'blockSize': tests['attn_input_shape'][1],
    'nEmbd': tests['attn_input_shape'][2],
    'nHead': tests['attn_config']['n_head'],
    'dropout': 0.0,
    'bias': true,
    'activation': 'quickgelu'
  }
  const block = Block(config)
  setWeights(
    block,
    tests['attn_weights'],
    [
      ['block/', ''],
      ['attn/c_attn/', 'attn/in_proj_'],
      ['attn/c_proj', 'attn/out_proj'],
      ['/', '.'],
      ['kernel', 'weight'],
      ['gamma', 'weight'],
      ['beta', 'bias'],
    ],
    [
      ['weight', [1, 0]],
    ]
  )
  const inputs = tf.tensor(tests['attn_input'])
  const outputs = block.apply(inputs)

  outputs.arraySync()[0].forEach((x, i) => {
    x.forEach((y, j) => {
      expect(y).toBeCloseTo(tests['attn_output'][0][i][j], 5)
    })
  })
})

test('VisionTransformer', () => {
  const config = {
    'name': 'vit',
    'inputResolution': tests['vit_config']['input_resolution'],
    'patchSize': tests['vit_config']['patch_size'],
    'nEmbd': tests['vit_config']['width'],
    'nHead': tests['vit_config']['heads'],
    'layers': tests['vit_config']['layers'],
    'outputDim': tests['vit_config']['output_dim'],
  }
  const model = VisionTransformer(config)
  setWeights(
    model,
    tests['vit_weights'],
    [
      ['vit/proj/kernel', 'proj'],
      ['vit/', ''],
      ['embedding/weight', 'embedding'],
      ['/', '.'],
      ['kernel', 'weight'],
      ['gamma', 'weight'],
      ['beta', 'bias'],
      ['h_', 'transformer.resblocks.'],
      ['attn.c_attn.', 'attn.in_proj_'],
      ['attn.c_proj', 'attn.out_proj'],
    ],
    [
      ['conv', [2, 3, 1, 0]],
      ['weight', [1, 0]],
    ]
  )
  const inputs = tf.tensor(tests['vit_input'])
  const outputs = model.apply(inputs)

  outputs.arraySync().forEach((x, i) => {
    x.forEach((y, j) => {
      expect(y).toBeCloseTo(tests['vit_output'][i][j], 5)
    })
  })
})

test('Image transform', () => {
  const inputs = tf.tensor(tests['transform_input'])
  const outputs = transform(inputs, tests['transform_resolution']) // .arraySync().flat(Infinity)
  const outputsExpected = tf.tensor(tests['transform_output'])
  expect(outputs.shape).toEqual(outputsExpected.shape)
  const mae = tf.metrics.meanAbsoluteError(outputs.flatten(), outputsExpected.flatten()).arraySync()
  expect(mae).toBeLessThan(0.1)

  // const outputsFlat = outputs.arraySync().flat(Infinity)
  // const outputsExpectedFlat = outputsExpected.arraySync().flat(Infinity)
  // outputsFlat.forEach((x, i) => {
  //   console.log(x, outputsExpectedFlat[i])
  //   // expect(x).toBeCloseTo(outputsExpected[i], 3)
  // })
})
