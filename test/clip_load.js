const fs = require('fs')
const tf = require('@tensorflow/tfjs')
const { SimpleTokenizer } = require('../dist/cjs/tokenizers.js')
const { Block } = require('../dist/cjs/layers.js')
const { VisionTransformer } = require('../dist/cjs/models/vit.js')
const { transform, CLIP } = require('../dist/cjs/models/clip.js')
const tests = require('./clip_data.json')

test('Tokenizer', () => {
  const bpe = fs.readFileSync(process.cwd() + '/bpe_simple_vocab_16e6.txt', 'utf8')
  const tokenizer = new SimpleTokenizer(bpe)
  const tokens = tests['cases'].map(x => tokenizer.tokenize(x))
  tokens.forEach((x, i) => {
    expect(x).toEqual(tests['tokenizer'][i])
  })
})

test('CLIP', async () => {
  // Check if the ../models/clip.h5 is loaded
  const cwd = process.cwd()
  const fn = cwd + '/models/clip.h5'
  console.log(fn)

  expect(fs.existsSync(fn)).toBe(true)
  const h5wasm = await import("h5wasm");
  await h5wasm.ready
  const f = new h5wasm.File(fn, 'r')
  const keys = loadH5Keys(f)
  expect(keys.length).toBe(398)

  const mapH5 = [
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
  ]
  const visConv = getWeight(f, 'vit/conv1', mapH5, [], 'h5')
  const visPos = getWeight(f, 'vit/positional_embedding/weight', mapH5, [], 'h5')
  const visLayers = getWeight(f, 'vit/transformer', mapH5, [], 'h5')
  const textEmb = getWeight(f, 'text/token_embedding/embeddings', mapH5, [], 'h5')
  const textPos = getWeight(f, 'text/positional_embedding/weight', mapH5, [], 'h5')
  const textLayers = getWeight(f, 'text/transformer', mapH5, [], 'h5')

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
  console.log('Initializing CLIP model with config:', config)
  const model = new CLIP(config)
  console.log('Loading weights...')
  setWeights(model.visual, f, mapH5, [], 'h5')
  setWeights(model.text, f, mapH5, [], 'h5')
  const bpe = fs.readFileSync(process.cwd() + '/bpe_simple_vocab_16e6.txt', 'utf8')
  const tokenizer = new SimpleTokenizer(bpe)
  const tokens = tokenizer.tokenize('a photo of a cat')
  const text = tf.tensor(tokens)
  const emb = model.text.apply(text)
  console.log(emb.arraySync())
  // setWeights(model.visual, f, mapH5, [], 'h5')
  // const clipWeights = iterate(f)

  // const weights = fs.readFileSync('../models/clip.h5')
  // console.log(FS)
})