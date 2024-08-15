import { bigintFromLimbs, bigintToLimbsRelaxed } from "../util.js";

export { mask51, mask64, c103, c52, c51, c51x3, c2, hiPre, loPre, c52n, c51n };
export {
  numberToBigint64,
  bigint64ToNumber,
  int64ToFloat52,
  float52ToInt64,
  bigintToFloat51Limbs,
  bigintFromFloat51Limbs,
};

let bytes = new Uint8Array(8);
let view = new DataView(bytes.buffer);

// constants
let mask51 = (1n << 51n) - 1n;
let mask64 = (1n << 64n) - 1n;

let c103 = 2 ** 103;
let c52 = 2 ** 52;
let c51 = 2 ** 52;
let c51x3 = 3 * 2 ** 51;
let c2 = c103 + c51x3;

// constants we have to subtract after reinterpreting raw float bytes as int64
let hiPre = numberToBigint64(c103);
let loPre = numberToBigint64(c51x3);
let c52n = numberToBigint64(c52);
let c51n = numberToBigint64(c51);

// conversion between bigints and floats

function numberToBigint64(x: number): bigint {
  view.setFloat64(0, x);
  return view.getBigUint64(0);
}
function bigint64ToNumber(x: bigint): number {
  view.setBigUint64(0, x);
  return view.getFloat64(0);
}

function int64ToFloat52(x: bigint) {
  return bigint64ToNumber(x | c51n) - c51;
}
function float52ToInt64(x: number) {
  return numberToBigint64(x + c51) & mask51;
}

// conversion between bigints and limb vectors

// store a limb vector of float64s
function bigintToFloat51Limbs(x: bigint) {
  let limbs = bigintToLimbsRelaxed(x, 51, 5);
  let floats = new Float64Array(5);
  for (let i = 0; i < 5; i++) {
    floats[i] = int64ToFloat52(limbs[i]);
  }
  return floats;
}
function bigintFromFloat51Limbs(x: Float64Array) {
  let limbs = new BigUint64Array(5);
  for (let i = 0; i < 5; i++) {
    limbs[i] = float52ToInt64(x[i]);
  }
  return bigintFromLimbs(limbs, 51, 5);
}
