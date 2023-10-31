const tf = require('@tensorflow/tfjs-node')
const { VisionTransformer } = require('../dist/cjs/models/vit.js')
const { transform } = require('../dist/cjs/models/clip.js')
const { setWeights } = require('../dist/cjs/utils')
const tests = require('./clip_data.json')

test('VisionTransformer', async () => {
  const config = {
    'name': 'vit',
    'inputResolution': tests['vit_config']['input_resolution'],
    'patchSize': tests['vit_config']['patch_size'],
    'nEmbd': tests['vit_config']['width'],
    'nHead': tests['vit_config']['heads'],
    'nLayer': tests['vit_config']['layers'],
    'outputDim': tests['vit_config']['output_dim'],
  }
  const model = VisionTransformer(config)
  await setWeights(
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
      ['h_', 'resblocks.'],
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
  const mae = tf.metrics.meanAbsoluteError(outputs.flatten(), outputsExpected.flatten()).arraySync()
  expect(mae).toBeLessThan(0.1)
})