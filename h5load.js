module.exports = (async () => {
  const cwd = process.cwd()
  const fn = cwd + '/models/clip.h5'
  console.log(fn)
  const h5wasm = await import("h5wasm")
  await h5wasm.ready
  return h5wasm
})()