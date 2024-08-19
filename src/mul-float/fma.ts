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
  local,
  v128,
  Global,
  type Func,
} from "wasmati";
import { inverse } from "../bigint/field.js";
import {
  c103,
  c2,
  c51,
  c51n,
  c52,
  c52n,
  hiPre,
  loPre,
  mask51,
  mask64,
  bigintToFloat51Limbs,
} from "./common.js";
import { constF64x2, constI64x2 } from "./field-base.js";
import { arithmetic, carryLocals } from "./arith.js";

export { Multiply };

type Multiply = {
  multiply: Func<["i32", "i32", "i32"], []>;
};

let zInitial = new BigInt64Array(11);
let loCount = [1n, 2n, 3n, 4n, 5n, 4n, 3n, 2n, 1n, 0n];
let hiCount = [0n, 1n, 2n, 3n, 4n, 5n, 4n, 3n, 2n, 1n];
for (let i = 0; i < 10; i++) {
  zInitial[i] = -((2n * (hiCount[i] * hiPre + loCount[i] * loPre)) & mask64);
}

let nLocalsV128 = [v128, v128, v128, v128, v128] as const;

function Multiply(
  p: bigint,
  pSelectPtr: Global<i32>,
  options?: { reduce?: boolean }
): Multiply {
  let pInv = inverse(-p, 1n << 51n);
  let PF = bigintToFloat51Limbs(p);

  let Arith = arithmetic(p, pSelectPtr);

  // original version that turned out to be slower
  let multiply2 = func(
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
        i32,
        ...nLocalsV128,
        ...nLocalsV128,
      ],
    },
    ([z, x, y], [xi, qi, hi1, hi2, lo1, lo2, carry, idx, ...rest]) => {
      let Y = rest.slice(0, 5);
      let Z = rest.slice(5, 10);

      // load y from memory into locals
      for (let i = 0; i < 5; i++) {
        local.set(Y[i], v128.load({ offset: i * 16 }, y));
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
          f64x2.relaxed_madd(
            qi,
            constF64x2(pj),
            f64x2.sub(constF64x2(c2), hi2)
          );
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

      if (options?.reduce) {
        Arith.reduceLocals(Z, carry, idx);
      }
      // propagate carries (to make limbs positive)
      carryLocals(Z);

      // convert to f64, store in memory
      for (let i = 0; i < 5; i++) {
        i64x2.add(Z[i], constI64x2(c52n));
        f64x2.sub($, constF64x2(c52));
        v128.store({ offset: i * 16 }, z, $);
      }
    }
  );

  let multiply = func(
    {
      in: [i32, i32, i32], // pointers to z, x, y, where z = x * y
      out: [],
      locals: [v128, i32, ...nLocalsV128, ...nLocalsV128, ...nLocalsV128, v128],
    },
    ([z, x, y], [tmp, idx, ...rest]) => {
      let Y = rest.slice(0, 5);
      let LH = rest.slice(5, 10);
      let Z = rest.slice(10, 16);

      // load y from memory into locals
      for (let i = 0; i < 5; i++) {
        local.set(Y[i], v128.load({ offset: i * 16 }, y));
      }

      // initialize Z with constants that offset float64 prefixes
      for (let i = 0; i < 6; i++) {
        local.set(Z[i], constI64x2(zInitial[i]));
      }

      for (let i = 0; i < 5; i++) {
        let xi = tmp;
        local.set(xi, v128.load({ offset: i * 16 }, x));

        for (let j = 0; j < 5; j++)
          local.set(LH[j], f64x2.relaxed_madd(xi, Y[j], constF64x2(c103))); // hi
        for (let j = 0; j < 5; j++)
          local.set(Z[j + 1], i64x2.add(Z[j + 1], LH[j]));
        for (let j = 0; j < 5; j++)
          local.set(LH[j], f64x2.sub(constF64x2(c2), LH[j])); // lo sub
        for (let j = 0; j < 5; j++)
          local.set(LH[j], f64x2.relaxed_madd(xi, Y[j], LH[j])); // lo
        for (let j = 0; j < 5; j++) local.set(Z[j], i64x2.add(Z[j], LH[j]));

        // compute qi
        let qi = tmp;
        i64x2.mul(Z[0], constI64x2(pInv));
        v128.and($, constI64x2(mask51));
        i64x2.add($, constI64x2(c51n));
        f64x2.sub($, constF64x2(c51));
        local.set(qi);

        for (let j = 0; j < 5; j++)
          local.set(
            LH[j],
            f64x2.relaxed_madd(qi, constF64x2(PF[j]), constF64x2(c103))
          );
        for (let j = 0; j < 5; j++)
          local.set(Z[j + 1], i64x2.add(Z[j + 1], LH[j]));
        for (let j = 0; j < 5; j++)
          local.set(LH[j], f64x2.sub(constF64x2(c2), LH[j])); // lo sub
        for (let j = 0; j < 5; j++)
          local.set(LH[j], f64x2.relaxed_madd(qi, constF64x2(PF[j]), LH[j])); // lo

        local.set(Z[0], i64x2.add(Z[0], LH[0]));
        local.set(Z[1], i64x2.add(Z[1], LH[1]));
        local.set(Z[0], i64x2.add(Z[1], i64x2.shr_s(Z[0], 51)));
        local.set(Z[1], i64x2.add(Z[2], LH[2]));
        local.set(Z[2], i64x2.add(Z[3], LH[3]));
        local.set(Z[3], i64x2.add(Z[4], LH[4]));
        local.set(Z[4], Z[5]);
        if (i < 4) local.set(Z[5], constI64x2(zInitial[6 + i]));
      }

      if (options?.reduce) {
        Arith.reduceLocals(Z, tmp, idx);
      }
      // propagate carries (to make limbs positive)
      carryLocals(Z);

      // convert to f64
      for (let i = 0; i < 5; i++) {
        i64x2.add(Z[i], constI64x2(c52n));
        f64x2.sub($, constF64x2(c52));
        local.set(Z[i], $);
      }

      // store in memory
      for (let i = 0; i < 5; i++) {
        v128.store({ offset: i * 16 }, z, Z[i]);
      }
    }
  );

  return { multiply };
}

// debugging helpers, currently unused
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
