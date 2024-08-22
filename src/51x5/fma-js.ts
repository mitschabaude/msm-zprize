/**
 * Explorational JS code inspired by Niall Emmart's work on using FMA instructions for bigint multiplication.
 *
 * Paper:
 *   "Faster Modular Exponentiation Using Double Precision Floating Point Arithmetic on the GPU."
 *   2018 IEEE 25th Symposium on Computer Arithmetic (ARITH). IEEE Computer Society, 2018.
 *   By Emmart, Zheng and Weems.
 *
 * Reference code:
 * https://github.com/yrrid/submission-wasm-twisted-edwards (see FP51.java and FieldPair.c)
 */
import { f64, f64x2, func, Module } from "wasmati";
import { pallasParams } from "../concrete/pasta.params.js";
import { createField, inverse } from "../bigint/field.js";
import { assert, bigintToLimbsRelaxed } from "../util.js";
import {
  bigint64ToNumber,
  bigintFromFloat51Limbs,
  bigintFromInt51Limbs,
  bigintToFloat51Limbs,
  bigintToInt51Limbs,
  c103,
  c2,
  c51,
  c51n,
  c52,
  c52n,
  hiPre,
  loPre,
  mask25,
  mask51,
  mask64,
  numberToBigint64,
} from "./common.js";

export {
  madd,
  montmulSimple,
  montmulRef,
  montmul,
  montmulFma,
  montMulFmaWrapped,
  montMulFmaWrapped2,
  montmulNoFmaWrapped,
  montmulNoFma,
  montmulNoFma2,
};

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
let madd: (x: number, y: number, z: number) => number = instance.exports.madd;

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

function montmul(x: bigint, y: bigint) {
  let X = bigintToInt51Limbs(x);
  let Y = bigintToInt51Limbs(y);

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
  return bigintFromInt51Limbs(Z);
}

let PF = bigintToFloat51Limbs(p);

let zInitial = new BigInt64Array(10);
let loCount = [1n, 2n, 3n, 4n, 5n, 4n, 3n, 2n, 1n, 0n];
let hiCount = [0n, 1n, 2n, 3n, 4n, 5n, 4n, 3n, 2n, 1n];
for (let i = 0; i < 10; i++) {
  zInitial[i] = -((2n * (hiCount[i] * hiPre + loCount[i] * loPre)) & mask64);
}

function montmulFma(X: Float64Array, Y: Float64Array) {
  let Z = new BigInt64Array(5);

  // initialize Z with constants that offset float64 prefixes
  for (let i = 0; i < 5; i++) {
    Z[i] = zInitial[i];
  }

  for (let i = 0; i < 5; i++) {
    let xi = X[i];

    let yj = Y[0];
    let hi1 = madd(xi, yj, c103);
    let lo1 = madd(xi, yj, c2 - hi1);
    Z[0] += numberToBigint64(lo1);

    let qi = bigint64ToNumber(((Z[0] * pInv) & mask51) + c51n) - c51;

    let hi2 = madd(qi, PF[0], c103);
    let lo2 = madd(qi, PF[0], c2 - hi2);
    let carry =
      numberToBigint64(hi1) +
      numberToBigint64(hi2) +
      ((Z[0] + numberToBigint64(lo2)) >> 51n);

    for (let j = 1; j < 5; j++) {
      let yj = Y[j];
      let pj = PF[j];
      let hi1 = madd(xi, yj, c103);
      let hi2 = madd(qi, pj, c103);
      let lo1 = madd(xi, yj, c2 - hi1);
      let lo2 = madd(qi, pj, c2 - hi2);

      Z[j - 1] = Z[j] + carry + numberToBigint64(lo1) + numberToBigint64(lo2);
      carry = numberToBigint64(hi1) + numberToBigint64(hi2);
    }
    Z[4] = zInitial[5 + i] + carry;
  }
  assert(Z[4] >= 0, `negative top limb ${Z[4]}`);

  // propagate carries to make limbs positive
  // not sure this is needed
  let carry = 0n;
  let floats = new Float64Array(5);
  for (let i = 0; i < 5; i++) {
    let lo = (Z[i] + carry) & mask51;
    floats[i] = bigint64ToNumber(lo + c52n) - c52;
    assert(floats[i] >= 0, `negative limb ${i}`);
    carry = Z[i] >> 51n;
  }
  assert(carry === 0n, `carry ${carry}`);
  return floats;
}

function montMulFmaWrapped(x: bigint, y: bigint) {
  let X = bigintToFloat51Limbs(x);
  let Y = bigintToFloat51Limbs(y);
  let Z = montmulFma(X, Y);
  return bigintFromFloat51Limbs(Z);
}

// version that is structured like Niall Emmart's version from https://github.com/yrrid/submission-wasm-twisted-edwards (FieldPair.c)
function montmulFma2(X: Float64Array, Y: Float64Array) {
  let Z = new BigInt64Array(6);
  let LH = new Float64Array(5);

  // initialize Z with constants that offset float64 prefixes
  for (let i = 0; i < 6; i++) {
    Z[i] = zInitial[i];
  }

  for (let i = 0; i < 5; i++) {
    let xi = X[i];

    for (let j = 0; j < 5; j++) LH[j] = madd(xi, Y[j], c103); // hi
    for (let j = 0; j < 5; j++) Z[j + 1] += numberToBigint64(LH[j]);
    for (let j = 0; j < 5; j++) LH[j] = c2 - LH[j]; // lo sub
    for (let j = 0; j < 5; j++) LH[j] = madd(xi, Y[j], LH[j]); // lo
    for (let j = 0; j < 5; j++) Z[j] += numberToBigint64(LH[j]);

    let qi = bigint64ToNumber(((Z[0] * pInv) & mask51) + c51n) - c51;

    for (let j = 0; j < 5; j++) LH[j] = madd(qi, PF[j], c103); // hi
    for (let j = 0; j < 5; j++) Z[j + 1] += numberToBigint64(LH[j]);
    for (let j = 0; j < 5; j++) LH[j] = c2 - LH[j]; // lo sub
    for (let j = 0; j < 5; j++) LH[j] = madd(qi, PF[j], LH[j]); // lo

    Z[0] = Z[0] + numberToBigint64(LH[0]);
    Z[1] = Z[1] + numberToBigint64(LH[1]);
    Z[0] = Z[1] + (Z[0] >> 51n);
    for (let j = 1; j < 4; j++) {
      Z[j] = Z[j + 1] + numberToBigint64(LH[j + 1]);
    }
    Z[4] = Z[5];
    if (i < 4) Z[5] = zInitial[6 + i];
  }
  assert(Z[4] >= 0, `negative top limb ${Z[4]}`);

  // propagate carries to make limbs positive
  let carry = 0n;
  let floats = new Float64Array(5);
  for (let i = 0; i < 5; i++) {
    let lo = (Z[i] + carry) & mask51;
    floats[i] = bigint64ToNumber(lo + c52n) - c52;
    assert(floats[i] >= 0, `negative limb ${i}`);
    carry = Z[i] >> 51n;
  }
  assert(carry === 0n, `carry ${carry}`);
  return floats;
}

function montMulFmaWrapped2(x: bigint, y: bigint) {
  let X = bigintToFloat51Limbs(x);
  let Y = bigintToFloat51Limbs(y);
  let Z = montmulFma2(X, Y);
  return bigintFromFloat51Limbs(Z);
}

const mask26 = (1n << 26n) - 1n;

let PI = bigintToInt51Limbs(p);
let PLo = PI.map((x) => x & mask26);
let PHi = PI.map((x) => x >> 26n);

// version that doesn't require fma instructions / relaxed_simd
function montmulNoFma(X: BigUint64Array, Y: BigUint64Array) {
  let Z = new BigUint64Array(6);
  let Ylo = new BigUint64Array(5);
  let Yhi = new BigUint64Array(5);

  // initialize Ylo, Yhi
  for (let i = 0; i < 5; i++) {
    let yi = Y[i];
    Ylo[i] = yi & mask26;
    Yhi[i] = yi >> 26n;
  }

  for (let i = 0; i < 5; i++) {
    let xi = X[i];
    let xiLo = xi & mask26;
    let xiHi = xi >> 26n;

    for (let j = 0; j < 5; j++) {
      let lo = xiLo * Ylo[j];
      let mid = xiLo * Yhi[j] + xiHi * Ylo[j];
      let hi = xiHi * Yhi[j];
      Z[j] += lo + ((mid & mask26) << 26n);
      Z[j + 1] += ((mid >> 26n) + hi) << 1n;
    }

    let qi = (Z[0] * pInv) & mask51;
    let qiLo = qi & mask26;
    let qiHi = qi >> 26n;

    for (let j = 0; j < 5; j++) {
      let lo = qiLo * PLo[j];
      let mid = qiLo * PHi[j] + qiHi * PLo[j];
      let hi = qiHi * PHi[j];
      Z[j] += lo + ((mid & mask26) << 26n);
      Z[j + 1] += ((mid >> 26n) + hi) << 1n;
    }

    Z[1] += Z[0] >> 51n;
    for (let j = 0; j < 5; j++) {
      Z[j] = Z[j + 1];
    }
    Z[5] = 0n;
  }
  assert(Z[4] >= 0, `negative top limb ${Z[4]}`);

  // propagate carries to make limbs positive
  let carry = 0n;
  for (let i = 0; i < 5; i++) {
    Z[i] += carry;
    let lo = Z[i] & mask51;
    carry = Z[i] >> 51n;
    Z[i] = lo;
  }
  assert(carry === 0n, `carry ${carry}`);

  let Z51 = new BigUint64Array(5);
  for (let i = 0; i < 5; i++) {
    Z51[i] = Z[i];
  }
  return Z51;
}

function montmulNoFmaWrapped(x: bigint, y: bigint) {
  let X = bigintToInt51Limbs(x);
  let Y = bigintToInt51Limbs(y);
  let Z = montmulNoFma(X, Y);
  return bigintFromInt51Limbs(Z);
}

let P10 = new BigUint64Array(10);
for (let i = 0; i < 5; i++) {
  P10[2 * i] = PLo[i];
  P10[2 * i + 1] = PHi[i];
}

// version that doesn't require fma instructions and is close to fast wasm version
function montmulNoFma2(X: BigUint64Array, Y51: BigUint64Array) {
  let Y = new BigUint64Array(10);
  let Z = new BigUint64Array(9);
  let P = P10;
  let pInv25 = inverse(-p, 1n << 25n);
  let pInv26 = inverse(-p, 1n << 26n);

  // initialize Y
  for (let i = 0; i < 5; i++) {
    let yi = Y51[i];
    Y[2 * i] = yi & mask26;
    Y[2 * i + 1] = yi >> 26n;
  }

  for (let i = 0; i < 5; i++) {
    let xi, qi, tmp;

    // LOWER HALF
    xi = X[i] & mask26;

    tmp = Z[0] + xi * Y[0];
    qi = computeQ(tmp, pInv26, mask26);
    Z[1] += (tmp + qi * P[0]) >> 26n;

    for (let j = 1; j < 9; j++) {
      Z[j - 1] = Z[j] + xi * Y[j] + qi * P[j];
    }
    Z[8] = xi * Y[9] + qi * P[9];

    // UPPER HALF
    xi = X[i] >> 26n;

    tmp = Z[0] + xi * Y[0];
    qi = computeQ(tmp, pInv25, mask25);
    Z[1] += (tmp + qi * P[0]) >> 25n;

    for (let j = 1; j < 9; j++) {
      let v = xi * Y[j] + qi * P[j];
      if (j % 2 === 1) v <<= 1n;
      Z[j - 1] = Z[j] + v;
    }
    Z[8] = (xi * Y[9] + qi * P[9]) << 1n;
  }

  let carry = 0n;
  let Z51 = new BigUint64Array(5);

  for (let i = 0; i < 4; i++) {
    let tmp = Z[2 * i] + carry + ((Z[2 * i + 1] & mask25) << 26n);
    Z51[i] = tmp & mask51;
    carry = (tmp >> 51n) + (Z[2 * i + 1] >> 25n);
  }
  Z51[4] = Z[8] + carry;

  return Z51;
}

function computeQ(z: bigint, pInv: bigint, mask: bigint) {
  if (pInv === mask) {
    // p = 1 mod 2^w  <==> -p^(-1) = -1 mod 2^w
    // qi = z * (-1) % 2^w = 2^w - z % 2^w
    return (mask + 1n - z) & mask;
  }
  return ((z & mask) * pInv) & mask;
}
