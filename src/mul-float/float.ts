/**
 * Explorational code inspired by Niall Emmart's work on using FMA instructions for bigint multiplication.
 *
 * Paper:
 *   "Faster Modular Exponentiation Using Double Precision Floating Point Arithmetic on the GPU."
 *   2018 IEEE 25th Symposium on Computer Arithmetic (ARITH). IEEE Computer Society, 2018.
 *   By Emmart, Zheng and Weems.
 *
 * Reference code:
 * https://github.com/yrrid/submission-wasm-twisted-edwards (see FP51.java and FieldPair.c)
 */

// IEEE floating point manipulation
import { f64, f64x2, func, Module } from "wasmati";
import { randomGenerators } from "../bigint/field-random.js";
import { assertDeepEqual } from "../testing/nested.js";
import { pallasParams } from "../concrete/pasta.params.js";
import { createField, inverse } from "../bigint/field.js";
import { assert, bigintFromLimbs, bigintToLimbsRelaxed } from "../util.js";
import { equivalent, spec, Spec } from "../testing/equivalent.js";
import { Random } from "../testing/random.js";

function numberToBytes(x: number): Uint8Array {
  let xBytes = new Uint8Array(8);
  let f64View = new DataView(xBytes.buffer);
  f64View.setFloat64(0, x, false);
  return xBytes;
}
function bytesToNumber(xBytes: Uint8Array): number {
  let f64View = new DataView(xBytes.buffer);
  return f64View.getFloat64(0, false);
}

type Sign = "pos" | "neg";

type Float = {
  sign: Sign;
  exponent: number;
  mantissa: bigint;
};

function sign(x: Uint8Array): Sign {
  // high bit of byte 0
  return x[0] >> 7 === 1 ? "neg" : "pos";
}
function setSign(x: Uint8Array, sign: Sign) {
  x[0] = sign === "neg" ? x[0] | 0b1000_0000 : x[0] & 0b0111_1111;
}

function exponent(x: Uint8Array): number {
  // low 7 bits of byte 0, high 4 bits of byte 1
  let exp = ((x[0] & 0b0111_1111) << 4) | (x[1] >> 4);
  // offset by 1023
  return exp - 1023;
}
function setExponent(x: Uint8Array, exp: number) {
  // offset by 1023
  exp += 1023;
  x[0] = (x[0] & 0b1000_0000) | (exp >> 4);
  x[1] = (x[1] & 0b0000_1111) | ((exp & 0b1111) << 4);
}

function mantissa(x: Uint8Array): bigint {
  // low 4 bits of byte 1, bytes 2-7
  return (
    ((BigInt(x[1]) & 0b0000_1111n) << 48n) |
    (BigInt(x[2]) << 40n) |
    (BigInt(x[3]) << 32n) |
    (BigInt(x[4]) << 24n) |
    (BigInt(x[5]) << 16n) |
    (BigInt(x[6]) << 8n) |
    BigInt(x[7])
  );
}
function setMantissa(x: Uint8Array, mantissa: bigint) {
  x[1] = (x[1] & 0b1111_0000) | Number(mantissa >> 48n);
  x[2] = Number((mantissa >> 40n) & 0xffn);
  x[3] = Number((mantissa >> 32n) & 0xffn);
  x[4] = Number((mantissa >> 24n) & 0xffn);
  x[5] = Number((mantissa >> 16n) & 0xffn);
  x[6] = Number((mantissa >> 8n) & 0xffn);
  x[7] = Number(mantissa & 0xffn);
}

function bytesToFloat(x: Uint8Array): Float {
  return {
    sign: sign(x),
    exponent: exponent(x),
    mantissa: mantissa(x),
  };
}
function floatToBytes(f: Float): Uint8Array {
  let x = new Uint8Array(8);
  setSign(x, f.sign);
  setExponent(x, f.exponent);
  setMantissa(x, f.mantissa);
  return x;
}

function floatToNumber(f: Float): number {
  let sign = f.sign === "pos" ? 1 : -1;
  let mantissa = Number(f.mantissa) / 2 ** 52;
  return sign * (1 + mantissa) * 2 ** f.exponent;
}
function numberToFloat(x: number): Float {
  let sign: Sign = x < 0 ? "neg" : "pos";
  x = Math.abs(x);
  let exponent = Math.floor(Math.log2(x));
  let mantissa = BigInt(Math.floor((x / 2 ** exponent - 1) * 2 ** 52));
  return { sign, exponent, mantissa };
}

function toFloat(x: number): Float {
  return bytesToFloat(numberToBytes(x));
}

for (let i = 0; i < 1_000; i++) {
  let x = Math.random();

  let xBytes = numberToBytes(x);
  let xFloat = bytesToFloat(xBytes);
  let xRecovered = floatToNumber(xFloat);
  assertDeepEqual(x, xRecovered);

  let xFloat2 = numberToFloat(x);
  let xBytes2 = floatToBytes(xFloat2);
  let xRecovered2 = bytesToNumber(xBytes2);
  assertDeepEqual(x, xRecovered2);
}

// bigint mul using float madd instruction

const maddWasm = func({ in: [f64, f64, f64], out: [f64] }, ([x, y, z]) => {
  f64x2.splat(x);
  f64x2.splat(y);
  f64x2.splat(z);
  f64x2.relaxed_madd();
  f64x2.extract_lane(0);
});
let module = Module({ exports: { madd: maddWasm } });
let { instance } = await module.instantiate();
let madd = instance.exports.madd;

let mBytes = new Uint8Array(8);
let mView = new DataView(mBytes.buffer);

function mantissaFromNumber(x: number): bigint {
  mView.setFloat64(0, x, false);
  mBytes[0] = 0;
  mBytes[1] &= 0b0000_1111;
  return mView.getBigUint64(0, false);
}
function numberToBigint64(x: number): bigint {
  mView.setFloat64(0, x, false);
  return mView.getBigUint64(0, false);
}
function bigint64ToNumber(x: bigint): number {
  mView.setBigUint64(0, x, false);
  return mView.getFloat64(0, false);
}
function floatToBigint64(x: Float): bigint {
  let xBytes = floatToBytes(x);
  let view = new DataView(xBytes.buffer);
  return view.getBigUint64(0, false);
}

let c103 = 2 ** 103;
let c52 = 2 ** 52;
let c51x3 = 3 * 2 ** 51;
let c2 = c103 + c51x3;
let shift51n = 1n << 51n;

// constants we have to subtract after reinterpreting raw float bytes as int64
let hiPre = floatToBigint64({ sign: "pos", exponent: 103, mantissa: 0n });
let loPre = numberToBigint64(c51x3);
let initialPre = floatToBigint64({ sign: "pos", exponent: 52, mantissa: 0n });

// or just mask out everything but the 52 mantissa bits
let mask51 = (1n << 51n) - 1n;
let mask52 = (1n << 52n) - 1n;

console.log("initial", `0x${initialPre.toString(16)}n`); // 0x4330000000000000n
console.log("hiPre", `0x${hiPre.toString(16)}n`); // 0x4660000000000000n
console.log("loPre", `0x${loPre.toString(16)}n`); // 0x4338000000000000n

// random numbers that can be 5 limbs of a number q < 2^255 + 2^253
// e.g., q < 2p where p is one of the Pasta primes or any < 254 bit prime
let rng = randomGenerators((1n << 51n) + (1n << 49n));

for (let i = 0; i < 10_000; i++) {
  let x = rng.randomField();
  let y = rng.randomField();
  let xF = bigint64ToNumber(x | initialPre) - c52;
  let yF = bigint64ToNumber(y | initialPre) - c52;

  let hi = madd(xF, yF, c103);
  let lo = madd(xF, yF, c2 - hi);

  let loRaw = numberToBigint64(lo);
  let hiRaw = numberToBigint64(hi);

  let xyBig = x * y;
  let xyFma = ((hiRaw - hiPre) << 51n) + (loRaw - loPre);
  let xyFma2 = ((hiRaw & mask52) << 51n) + (loRaw & mask52) - shift51n;
  assertDeepEqual(xyBig, xyFma);
  assertDeepEqual(xyBig, xyFma2);
}

// modmul with 5 x 51-bit limbs
let Fp = createField(pallasParams.modulus);
let p = Fp.modulus;
let P = bigintToLimbsRelaxed(p, 51, 5);
let pInv = inverse(-p, 1n << 51n);
let R = Fp.mod(1n << 255n);
let Rinv = Fp.inverse(R);

function split(x: bigint) {
  return [x & mask51, x >> 51n];
}

let field = Spec.field(p, { relaxed: true });
let fieldStrict = Spec.field(p, { relaxed: false });

// simple, inefficient reference implementation of montgomery multiplication

function montmulSimple(x: bigint, y: bigint) {
  let pInv = inverse(-p, 1n << 255n);
  let xy = x * y;
  let q = ((xy & ((1n << 255n) - 1n)) * pInv) & ((1n << 255n) - 1n);
  let z = (xy + q * p) >> 255n;
  return z < p ? z : z - p;
}

function montmulRef(xR: bigint, yR: bigint) {
  let zR2 = Fp.multiply(xR, yR);
  return Fp.multiply(zR2, Rinv);
}

// montmul with the correction step is (exactly) the same as multiplying and dividing by the Montgomery radius
equivalent({ from: [field, field], to: fieldStrict, verbose: true })(
  montmulSimple,
  montmulRef,
  "montmul ref"
);

function montmul(x: bigint, y: bigint) {
  let X = bigintToLimbsRelaxed(x, 51, 5);
  let Y = bigintToLimbsRelaxed(y, 51, 5);

  let Z = new BigUint64Array(6);

  for (let i = 0; i < 5; i++) {
    for (let j = 0; j < 5; j++) {
      let [lo, hi] = split(X[i] * Y[j]);
      Z[j] += lo;
      Z[j + 1] += hi;
    }

    // Z += qi * P, such that Z % 2^51 = 0
    let qi = (Z[0] * pInv) & mask51;

    for (let j = 0; j < 5; j++) {
      let [lo, hi] = split(qi * P[j]);
      Z[j] += lo;
      Z[j + 1] += hi;
    }

    // shift down after propagating carry from first limb
    Z[1] += Z[0] >> 51n;
    for (let j = 0; j < 5; j++) {
      Z[j] = Z[j + 1];
    }
    Z[5] = 0n;
  }
  return bigintFromLimbs(Z, 51, 5);
}

// the bigint and array-based versions of montmul are equivalent
equivalent({ from: [field, field], to: field, verbose: true })(
  montmulSimple,
  montmul,
  "montmul consistent"
);

// given inputs < p, montmul returns a value < 2p
// this means it will need a correction step in many places! (we can't go from 2p to 2p, because the Montgomery radius is too small)
equivalent({ from: [field, field], to: Spec.boolean, verbose: true })(
  (x, y) => montmul(x, y) < 2n * p,
  () => true,
  "montmul < 2p"
);

function int64ToFloat64(x: bigint) {
  return bigint64ToNumber(x | initialPre) - c52;
}
function float51ToInt64(x: number) {
  return numberToBigint64(x + c52) & mask52;
}

let int51 = spec(Random(() => rng.randomField()));
let int51F = spec(Random(() => Number(rng.randomField())));

equivalent({ from: [int51], to: int51, verbose: true })(
  (x) => x,
  (x) => float51ToInt64(int64ToFloat64(x)),
  "int/float roundtrip"
);
equivalent({ from: [int51F], to: int51F, verbose: true })(
  (x) => x,
  (x) => int64ToFloat64(float51ToInt64(x)),
  "float/int roundtrip"
);

// store a limb vector of float64s
function bigintToFloat51Limbs(x: bigint) {
  let limbs = bigintToLimbsRelaxed(x, 51, 5);
  let floats = new Float64Array(5);
  for (let i = 0; i < 5; i++) {
    floats[i] = int64ToFloat64(limbs[i]);
  }
  return floats;
}
function bigintFromFloat51Limbs(x: Float64Array) {
  let limbs = new BigUint64Array(5);
  for (let i = 0; i < 5; i++) {
    limbs[i] = float51ToInt64(x[i]);
  }
  return bigintFromLimbs(limbs, 51, 5);
}

let PF = bigintToFloat51Limbs(p);

function montmulFma(X: Float64Array, Y: Float64Array) {
  let Z = new BigInt64Array(6);

  for (let i = 0; i < 5; i++) {
    let xF = X[i];

    for (let j = 0; j < 5; j++) {
      let yF = Y[j];
      let hi = madd(xF, yF, c103);
      let lo = madd(xF, yF, c2 - hi);
      let loRaw = numberToBigint64(lo);
      let hiRaw = numberToBigint64(hi);
      Z[j] += loRaw - loPre;
      Z[j + 1] += hiRaw - hiPre;
    }

    // Z += qi * P, such that Z % 2^51 = 0
    let qi = bigint64ToNumber(((Z[0] * pInv) & mask51) | initialPre) - c52;
    xF = qi;

    for (let j = 0; j < 5; j++) {
      let yF = PF[j];
      let hi = madd(xF, yF, c103);
      let lo = madd(xF, yF, c2 - hi);
      let loRaw = numberToBigint64(lo);
      let hiRaw = numberToBigint64(hi);
      Z[j] += loRaw - loPre;
      Z[j + 1] += hiRaw - hiPre;
    }

    // shift down after propagating carry from first limb
    Z[1] += Z[0] >> 51n;
    for (let j = 0; j < 5; j++) {
      Z[j] = Z[j + 1];
    }
    Z[5] = 0n;
  }

  // propagate carries to make limbs positive
  // not sure this is needed
  let carry = 0n;
  for (let i = 0; i < 5; i++) {
    let lo = (Z[i] + carry) & mask51;
    carry = Z[i] >> 51n;
    Z[i] = lo;
    assert(Z[i] >= 0n, `negative limb ${i}`);
  }
  assert(carry === 0n, `carry ${carry}`);

  // convert to float64s
  let floats = new Float64Array(5);
  for (let i = 0; i < 5; i++) {
    floats[i] = bigint64ToNumber(Z[i] | initialPre) - c52;
  }

  return floats;
}

function montMulFmaWrapped(x: bigint, y: bigint) {
  let X = bigintToFloat51Limbs(x);
  let Y = bigintToFloat51Limbs(y);
  let Z = montmulFma(X, Y);
  return bigintFromFloat51Limbs(Z);
}

// montmulFma is exactly equivalent to montmul
equivalent({ from: [field, field], to: fieldStrict, verbose: true })(
  montmul,
  montMulFmaWrapped,
  "montmulFma"
);
