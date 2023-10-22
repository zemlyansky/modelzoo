import { GELU, gelu } from '../layers';
import * as tf from '@tensorflow/tfjs';

function resizeBicubic(inputTensor, [newHeight, newWidth]) {
  const [height, width, numChannels] = inputTensor.shape;

  const pixels = inputTensor.arraySync();
  const newPixels = new Array(newHeight).fill(0).map(row => new Array(newWidth).fill(0).map(col => new Array(numChannels).fill(0)));

  const xRatio = width / newWidth;
  const yRatio = height / newHeight;

  for (let y = 0; y < newHeight; y++) {
    for (let x = 0; x < newWidth; x++) {
      const px = x * xRatio;
      const py = y * yRatio;

      for (let c = 0; c < numChannels; c++) {
        newPixels[y][x][c] = cubicInterpolate(pixels, px, py, width, height, c);
      }
    }
  }

  return tf.tensor(newPixels);
}

function cubicInterpolate(pixels, x, y, width, height, channel) {
  const x1 = Math.floor(x);
  const y1 = Math.floor(y);

  // Handle border cases
  const x0 = Math.max(x1 - 1, 0);
  const y0 = Math.max(y1 - 1, 0);
  const x2 = Math.min(x1 + 1, width - 1);
  const y2 = Math.min(y1 + 1, height - 1);
  const x3 = Math.min(x1 + 2, width - 1);
  const y3 = Math.min(y1 + 2, height - 1);

  const p = [x0, x1, x2, x3].map(x => [y0, y1, y2, y3].map(y => pixels[y][x][channel]));

  // Cubic interpolation for each row
  const rows = p.map(row => cubicInterpolateArray(row, x - x1));

  // Cubic interpolation for the column
  return cubicInterpolateArray(rows, y - y1);
}

function cubicInterpolateArray(values, x) {
  const [v0, v1, v2, v3] = values;
  const P = (v3 - v2) - (v0 - v1);
  const Q = (v0 - v1) - P;
  const R = v2 - v0;
  const S = v1;

  return P * Math.pow(x, 3) + Q * Math.pow(x, 2) + R * x + S;
}

function calculateImageSize(height: number, width: number, target: number): [number, number] {
  let newHeight: number;
  let newWidth: number;

  if (height > width) {
    newWidth = target;
    newHeight = Math.floor((height / width) * target);
  } else {
    newHeight = target;
    newWidth = Math.floor((width / height) * target);
  }
  return [newHeight, newWidth];
}

export function transform(inputTensor: tf.Tensor3D | tf.Tensor4D, n_px: number) {
  const s = inputTensor.shape.length;

  // Scale
  let img = tf.div(inputTensor, 255) as tf.Tensor3D | tf.Tensor4D;

  // Resize
  const [height, width] = calculateImageSize(inputTensor.shape[s-3], inputTensor.shape[s-2], n_px);
  img = tf.image.resizeBilinear(img, [height, width]);
  // img = resizeBicubic(img, [height, width]) as tf.Tensor3D | tf.Tensor4D;

  const centerY = Math.floor(img.shape[s-3] / 2);
  const centerX = Math.floor(img.shape[s-2] / 2);
  const halfLength = Math.floor(n_px / 2)

  if (s === 3) {
    img = img.expandDims(0) as tf.Tensor4D;
  }

  img = tf.slice(
    img,
    [0, centerY - halfLength, centerX - halfLength, 0],
    [-1, n_px, n_px, 3]
  );

  if (s === 3) {
    // Convert back to 3D if it was 3D initially
    img = img.squeeze();
  }

  // Normalize
  const mean = tf.tensor([0.48145466, 0.4578275, 0.40821073]); //.reshape([1,1,1,3]);
  const std = tf.tensor([0.26862954, 0.26130258, 0.27577711]); //.reshape([1,1,1,3]);
  img = tf.div(tf.sub(img, mean), std);

  return img;
}

export class CLIP {
    constructor() {}
}