const path = require('path');

module.exports = {
  mode: 'production',
  entry: './dist/esm/index.js',
  output: {
    filename: 'index.js',
    path: path.resolve(__dirname, 'dist', 'iife'),
    libraryTarget: 'var',
    library: 'modelzoo',
  },
  externals: {
    '@tensorflow/tfjs': 'tf'
  },
  // module: {
  //   rules: [
  //     {
  //       test: /\.js$/,
  //       exclude: /node_modules/,
  //       use: {
  //         loader: 'babel-loader',
  //       },
  //     },
  //   ],
  // },
};

