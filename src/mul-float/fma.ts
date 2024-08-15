/**
 * Wasm code inspired by Niall Emmart's work on using FMA instructions for bigint multiplication.
 *
 * Paper:
 *   "Faster Modular Exponentiation Using Double Precision Floating Point Arithmetic on the GPU."
 *   2018 IEEE 25th Symposium on Computer Arithmetic (ARITH). IEEE Computer Society, 2018.
 *   By Emmart, Zheng and Weems.
 *
 * Reference code:
 * https://github.com/yrrid/submission-wasm-twisted-edwards (see FP51.java and FieldPair.c)
 */
import {
  $,
  call,
  f64,
  f64x2,
  func,
  i32,
  i64,
  i64x2,
  importFunc,
  importMemory,
  local,
  Module,
  v128,
} from "wasmati";
import { assertDeepEqual } from "../testing/nested.js";
import { pallasParams } from "../concrete/pasta.params.js";
import { createField, inverse } from "../bigint/field.js";
import { Random } from "../testing/random.js";
import { memoryHelpers } from "../wasm/memory-helpers.js";
import { createEquivalentWasm, wasmSpec } from "../testing/equivalent-wasm.js";
import { Tuple } from "../types.js";
import {
  bigintToFloat51Limbs,
  float52ToInt64,
  montMulFmaWrapped,
  numberToBigint64,
} from "./fma-js.js";

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

let zInitial = new BigInt64Array(11);
let loCount = [1n, 2n, 3n, 4n, 5n, 4n, 3n, 2n, 1n, 0n, 0n];
let hiCount = [0n, 1n, 2n, 3n, 4n, 5n, 4n, 3n, 2n, 1n, 0n];
for (let i = 0; i < 11; i++) {
  zInitial[i] = -((2n * (hiCount[i] * hiPre + loCount[i] * loPre)) & mask64);
}

let nLocalsV128 = [v128, v128, v128, v128, v128] as const;

function constF64x2(x: number) {
  return v128.const("f64x2", [x, x]);
}
function constI64x2(x: bigint) {
  return v128.const("i64x2", [x, x]);
}

let log = (...args: any) => console.log("wasm", ...args);
let logI64 = importFunc({ in: [i32, i64], out: [] }, log);
let logF64 = importFunc({ in: [i32, f64], out: [] }, log);
let logF64x2_0 = func({ in: [i32, v128], out: [] }, ([i, x]) => {
  local.get(x);
  f64x2.extract_lane(0);
  call(logF64, [i, $]);
});
let logI64x2_0 = func({ in: [i32, v128], out: [] }, ([i, x]) => {
  local.get(x);
  i64x2.extract_lane(0);
  call(logI64, [i, $]);
});

// TODO constants that need to be parameters
let Fp = createField(pallasParams.modulus);
let p = Fp.modulus;
let pInv = inverse(-p, 1n << 51n);
let R = Fp.mod(1n << 255n);
let PF = bigintToFloat51Limbs(p);

let multiplyWasm = func(
  {
    in: [i32, i32, i32], // pointers to z, x, y, where z = x * y
    out: [],
    locals: [
      v128,
      v128,
      v128,
      v128,
      v128,
      v128,
      v128,
      ...nLocalsV128,
      ...nLocalsV128,
    ],
  },
  ([z, x, y], [xi, qi, hi1, hi2, lo1, lo2, carry, ...rest]) => {
    let Y = rest.slice(0, 5);
    let Z = rest.slice(5, 10);

    // load y from memory into locals
    for (let i = 0; i < 5; i++) {
      let xi = v128.load({ offset: i * 16 }, y);
      local.set(Y[i], xi);
    }

    // initialize Z with constants that offset float64 prefixes
    for (let i = 0; i < 5; i++) {
      local.set(Z[i], constI64x2(zInitial[i]));
    }

    for (let i = 0; i < 5; i++) {
      local.set(xi, v128.load({ offset: i * 16 }, x));
      let yj = Y[0];
      let pj = PF[0];

      f64x2.relaxed_madd(xi, yj, constF64x2(c103));
      local.set(hi1);
      f64x2.relaxed_madd(xi, yj, f64x2.sub(constF64x2(c2), hi1));
      local.set(lo1);
      local.set(Z[0], i64x2.add(Z[0], lo1));

      // compute qi
      i64x2.mul(Z[0], constI64x2(pInv));
      v128.and($, constI64x2(mask51));
      i64x2.add($, constI64x2(c51n));
      f64x2.sub($, constF64x2(c51));
      local.set(qi);

      f64x2.relaxed_madd(qi, constF64x2(pj), constF64x2(c103));
      local.set(hi2);
      f64x2.relaxed_madd(qi, constF64x2(pj), f64x2.sub(constF64x2(c2), hi2));
      local.set(lo2);

      // compute carry from Z[0]
      i64x2.add(hi1, hi2);
      i64x2.add(Z[0], lo2);
      i64x2.shr_s($, 51);
      local.set(carry, i64x2.add($, $));

      // inner loop
      for (let j = 1; j < 5; j++) {
        yj = Y[j];
        pj = PF[j];

        f64x2.relaxed_madd(xi, yj, constF64x2(c103));
        local.set(hi1);
        f64x2.relaxed_madd(qi, constF64x2(pj), constF64x2(c103));
        local.set(hi2);
        f64x2.relaxed_madd(xi, yj, f64x2.sub(constF64x2(c2), hi1));
        local.set(lo1);
        f64x2.relaxed_madd(qi, constF64x2(pj), f64x2.sub(constF64x2(c2), hi2));
        local.set(lo2);

        i64x2.add(Z[j], carry);
        i64x2.add($, lo1);
        i64x2.add($, lo2);
        local.set(Z[j - 1]);

        local.set(carry, i64x2.add(hi1, hi2));
      }
      i64x2.add(constI64x2(zInitial[5 + i]), carry);
      local.set(Z[4]);
    }

    // propagate carries (to make limbs positive), convert to f64, store in memory
    local.set(carry, constI64x2(0n));
    for (let i = 0; i < 5; i++) {
      i64x2.add(Z[i], carry);
      v128.and($, constI64x2(mask51));
      i64x2.add($, constI64x2(c52n));
      f64x2.sub($, constF64x2(c52));
      v128.store({ offset: i * 16 }, z, $);

      if (i < 4) local.set(carry, i64x2.shr_s(Z[i], 51));
    }
  }
);

let wasmMemory = importMemory({ min: 1000, max: 1000, shared: true });
let multiplyModule = Module({
  memory: wasmMemory,
  exports: { multiply: multiplyWasm },
});
let { instance: mulInstance } = await multiplyModule.instantiate();
let sizeField = 8 * 5;
let sizeFieldPair = 2 * sizeField;

const Memory = memoryHelpers(p, 51, 5, { memory: wasmMemory.value });
const memory = Memory.memoryBytes;

const Field = {
  multiply: mulInstance.exports.multiply,
  size: sizeFieldPair,

  copy(x: number, y: number) {
    memory.copyWithin(x, y, y + sizeFieldPair);
  },

  writeBigintPair(x: number, x0: bigint, x1: bigint) {
    let view = new DataView(memory.buffer, x, sizeFieldPair);

    for (let offset = 0; offset < sizeFieldPair; offset += 16) {
      // we write each limb of x0 and x1 next to each other, as one v128
      view.setBigInt64(offset, (x0 & mask51) | c51n, true);
      let x0F = view.getFloat64(offset, true);
      view.setFloat64(offset, x0F - c51, true);

      view.setBigInt64(offset + 8, (x1 & mask51) | c51n, true);
      let x1F = view.getFloat64(offset + 8, true);
      view.setFloat64(offset + 8, x1F - c51, true);

      x0 >>= 51n;
      x1 >>= 51n;
    }
  },
  // writes only one half of a pair and leaves the other as is
  writeBigint(x: number, x0: bigint) {
    let view = new DataView(memory.buffer, x, sizeFieldPair);

    for (let offset = 0; offset < sizeFieldPair; offset += 16) {
      view.setBigInt64(offset, (x0 & mask51) | c51n, true);
      let x0F = view.getFloat64(offset, true);
      view.setFloat64(offset, x0F - c51, true);
      x0 >>= 51n;
    }
  },

  readBigintPair(x: number) {
    let view = new DataView(memory.buffer, x, sizeFieldPair);
    let x0 = 0n;
    let x1 = 0n;

    for (let offset = sizeFieldPair - 16; offset >= 0; offset -= 16) {
      let x0F = view.getFloat64(offset, true);
      x0 = (x0 << 51n) | float52ToInt64(x0F);

      let x1F = view.getFloat64(offset + 8, true);
      x1 = (x1 << 51n) | float52ToInt64(x1F);
    }
    return [x0, x1] satisfies Tuple;
  },
  readBigint(x: number) {
    let view = new DataView(memory.buffer, x, sizeFieldPair);
    let x0 = 0n;
    for (let offset = sizeFieldPair - 16; offset >= 0; offset -= 16) {
      let x0F = view.getFloat64(offset, true);
      x0 = (x0 << 51n) | float52ToInt64(x0F);
    }
    return x0;
  },
};

// manual simple test
{
  let x = Memory.local.getPointer(sizeFieldPair);
  let y = Memory.local.getPointer(sizeFieldPair);
  let z = Memory.local.getPointer(sizeFieldPair);

  Field.writeBigintPair(x, 1n, 1n);
  Field.writeBigintPair(y, R, 1n);
  Field.multiply(z, x, y);

  let [z00, z01] = Field.readBigintPair(z);
  let z10 = montMulFmaWrapped(1n, R);
  let z11 = montMulFmaWrapped(1n, 1n);

  assertDeepEqual(z00, z10, "montmul wasm 1");
  assertDeepEqual(z01, z11, "montmul wasm 2");
}

// property tests

let eqivalentWasm = createEquivalentWasm(Memory, { logSuccess: true });
let fieldRng = Random.field(p);

let field = wasmSpec(Memory, fieldRng, {
  size: Field.size,
  there: Field.writeBigint,
  back: Field.readBigint,
});

let fieldPair = wasmSpec(Memory, Random.tuple([fieldRng, fieldRng]), {
  size: Field.size,
  there: (xPtr, [x0, x1]) => Field.writeBigintPair(xPtr, x0, x1),
  back: Field.readBigintPair,
});

eqivalentWasm(
  { from: [field], to: field },
  (x) => x,
  Field.copy,
  "wasm roundtrip"
);

eqivalentWasm(
  { from: [fieldPair], to: fieldPair },
  (x) => x,
  Field.copy,
  "wasm roundtrip pair"
);

// wasm version is exactly equivalent to montmulFma

eqivalentWasm(
  { from: [field, field], to: field },
  montMulFmaWrapped,
  Field.multiply,
  "montmul fma (wasm)"
);

eqivalentWasm(
  { from: [fieldPair, fieldPair], to: fieldPair },
  (x, y): [bigint, bigint] => [
    montMulFmaWrapped(x[0], y[0]),
    montMulFmaWrapped(x[1], y[1]),
  ],
  Field.multiply,
  "montmul fma pairwise"
);
