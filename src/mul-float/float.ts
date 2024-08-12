// IEEE floating point manipulation
import { f64, f64x2, func, Module } from "wasmati";
import { randomGenerators } from "../bigint/field-random.js";
import { assertDeepEqual } from "../testing/nested.js";

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

let c103 = 2 ** 103;
let c51x3 = 3 * 2 ** 51;

// random 51 bit numbers
let R = randomGenerators(1n << 51n);
let rand = () => Number(R.randomField());

for (let i = 0; i < 10_000; i++) {
  let x = rand();
  let y = rand();

  let hi = madd(x, y, c103);
  let loAdd = c103 + c51x3 - hi;
  let lo = madd(x, y, loAdd);
  let loCorr = lo - c51x3;
  let hiM = mantissaFromNumber(hi);

  let xyBig = BigInt(x) * BigInt(y);
  let xyFma = (hiM << 51n) + BigInt(loCorr);
  assertDeepEqual(xyBig, xyFma);
}
