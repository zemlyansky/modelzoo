const tf = require('@tensorflow/tfjs-node')
const { Block } = require('../dist/cjs/layers.js')
const { setWeights } = require('../dist/cjs/utils')
const tests = require('./clip_data.json')

test('ResidualAttentionBlock', async () => {
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
  await setWeights(
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