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
import { randomGenerators } from "../bigint/field-random.js";
import { assertDeepEqual } from "../testing/nested.js";
import { pallasParams } from "../concrete/pasta.params.js";
import { equivalent, spec, Spec } from "../testing/equivalent.js";
import { Random } from "../testing/random.js";
import {
  madd,
  montmul,
  montMulFmaWrapped,
  montMulFmaWrapped2,
  montmulRef,
  montmulSimple,
} from "./fma-js.js";
import {
  bigint64ToNumber,
  float52ToInt64,
  int64ToFloat52,
  numberToBigint64,
} from "./common.js";

// bigint mul using float madd instruction

let c103 = 2 ** 103;
let c52 = 2 ** 52;
let c51x3 = 3 * 2 ** 51;
let c2 = c103 + c51x3;
let shift51n = 1n << 51n;

// constants we have to subtract after reinterpreting raw float bytes as int64
let hiPre = numberToBigint64(c103);
let loPre = numberToBigint64(c51x3);
let c52n = numberToBigint64(c52);

// or just mask out everything but the 52 mantissa bits
let mask52 = (1n << 52n) - 1n;

// random numbers that can be 5 limbs of a number q < 2^255 + 2^253
// e.g., q < 2p where p is one of the Pasta primes or any < 254 bit prime
let rng51 = randomGenerators(1n << 51n);
let rng52 = randomGenerators(1n << 52n);
let rng51eps = randomGenerators((1n << 51n) + (1n << 49n)); // also works when both x, y are sampled from this

for (let i = 0; i < 10_000; i++) {
  let x = rng52.randomField(); // note: x can be 52 bits long
  let y = rng51.randomField();
  let xF = bigint64ToNumber(x | c52n) - c52;
  let yF = bigint64ToNumber(y | c52n) - c52;

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
let p = pallasParams.modulus;

let field = Spec.field(p, { relaxed: true });
let fieldStrict = Spec.field(p, { relaxed: false });

// montmul with the correction step is (exactly) the same as multiplying and dividing by the Montgomery radius
equivalent({ from: [field, field], to: fieldStrict, verbose: true })(
  montmulSimple,
  montmulRef,
  "montmul ref"
);

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

let int51 = spec(Random(() => rng51.randomField()));
let int51F = spec(Random(() => Number(rng51.randomField())));

equivalent({ from: [int51], to: int51, verbose: true })(
  (x) => x,
  (x) => float52ToInt64(int64ToFloat52(x)),
  "int/float roundtrip"
);
equivalent({ from: [int51F], to: int51F, verbose: true })(
  (x) => x,
  (x) => int64ToFloat52(float52ToInt64(x)),
  "float/int roundtrip"
);

// montmulFma is exactly equivalent to montmul
equivalent({ from: [field, field], to: fieldStrict, verbose: true })(
  montmul,
  montMulFmaWrapped,
  "montmul fma (js)"
);

// montmulFma2 is exactly equivalent to montmul
equivalent({ from: [field, field], to: fieldStrict, verbose: true })(
  montmul,
  montMulFmaWrapped2,
  "montmul fma 2 (js)"
);
