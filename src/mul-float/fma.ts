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

export { Multiply };

type Multiply = {
  multiply: Func<["i32", "i32", "i32"], []>;
};

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

function Multiply(p: bigint): Multiply {
  let pInv = inverse(-p, 1n << 51n);
  let PF = bigintToFloat51Limbs(p);

  let multiply = func(
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
