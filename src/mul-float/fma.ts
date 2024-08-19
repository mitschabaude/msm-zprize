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
  bigintToInt51Limbs,
} from "./common.js";
import { constF64x2, constI64x2 } from "./field-base.js";
import { arithmetic, carryLocals } from "./arith.js";
import { assert } from "../util.js";

export { Multiply };

type Multiply = {
  multiply: Func<["i32", "i32", "i32"], []>;
  multiplyNoFma: Func<["i32", "i32", "i32"], []>;
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
  /**
   * Parameters that determine the amount of normalization done on the multiplication output
   */
  {
    reduce = false,
    carry = true,
    convert = true,
  }: {
    /**
     * Whether to optionally subtract p to bring the result back into a range < p + eps with eps << p
     */
    reduce?: boolean;
    /**
     * Whether to propagate carries and make limbs positive
     */
    carry?: boolean;
    /**
     * Whether to convert the result back to a float64
     */
    convert?: boolean;
  } = {}
): Multiply {
  let pInv = inverse(-p, 1n << 51n);
  let PF = bigintToFloat51Limbs(p);

  // if we convert the output to float64, we need to carry as well
  assert(!convert || carry, "must carry if converting to float64");

  let { reduceLocals } = arithmetic(p, pSelectPtr);

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

      if (reduce) reduceLocals(Z, carry, idx);
      if (carry) carryLocals(Z);

      // convert to f64
      if (convert)
        for (let i = 0; i < 5; i++) {
          i64x2.add(Z[i], constI64x2(c52n));
          f64x2.sub($, constF64x2(c52));
          local.set(Z[i], $);
        }

      for (let i = 0; i < 5; i++) {
        v128.store({ offset: i * 16 }, z, Z[i]);
      }
    }
  );

  /**
   * Main 51x5 multiplication routine, using FMA instructions from the "relaxed SIMD" Wasm extension.
   */
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

      if (reduce) reduceLocals(Z, tmp, idx);
      if (carry) carryLocals(Z);

      // convert output to f64
      if (convert)
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

  let mask26 = (1n << 26n) - 1n;
  let PI = bigintToInt51Limbs(p);
  let Plo = PI.map((x) => x & mask26);
  let Phi = PI.map((x) => x >> 26n);

  /**
   * 51x5 multiplication without using FMA instructions, as a fallback for compatibility.
   */
  let multiplyNoFma = func(
    {
      in: [i32, i32, i32], // pointers to z, x, y, where z = x * y
      out: [],
      locals: [
        v128,
        i32,
        v128,
        v128,
        ...nLocalsV128,
        ...nLocalsV128,
        ...nLocalsV128,
        v128,
      ],
    },
    ([z, x, y], [tmp, idx, xiLo, xiHi, ...rest]) => {
      let Ylo = rest.slice(0, 5);
      let Yhi = rest.slice(5, 10);
      let Z = rest.slice(10, 16);

      // load y from memory into locals
      for (let i = 0; i < 5; i++) {
        local.set(tmp, v128.load({ offset: i * 16 }, y));
        local.set(Ylo[i], v128.and(tmp, constI64x2(mask26)));
        local.set(Yhi[i], i64x2.shr_s(tmp, 26));
      }

      for (let i = 0; i < 5; i++) {
        local.set(tmp, v128.load({ offset: i * 16 }, x));
        local.set(xiLo, v128.and(tmp, constI64x2(mask26)));
        local.set(xiHi, i64x2.shr_s(tmp, 26));

        for (let j = 0; j < 5; j++) {
          let mid = tmp;
          i64x2.mul(xiLo, Yhi[j]);
          i64x2.mul(xiHi, Ylo[j]);
          local.set(mid, i64x2.add());

          // Z[j] += lo + ((mid & mask26) << 26n);
          v128.and(mid, constI64x2(mask26));
          i64x2.shl($, 26);
          i64x2.mul(xiLo, Ylo[j]);
          i64x2.add();
          local.set(Z[j], i64x2.add($, Z[j]));

          // Z[j + 1] += 2n * ((mid >> 26n) + hi);
          i64x2.shr_s(mid, 26);
          i64x2.mul(xiHi, Yhi[j]);
          i64x2.add();
          i64x2.shl($, 1);
          local.set(Z[j + 1], i64x2.add($, Z[j + 1]));
        }

        // compute qi
        let qi = tmp;
        let qiLo = xiLo;
        let qiHi = xiHi;
        i64x2.mul(Z[0], constI64x2(pInv));
        v128.and($, constI64x2(mask51));
        local.set(qi);
        local.set(qiLo, v128.and(qi, constI64x2(mask26)));
        local.set(qiHi, i64x2.shr_s(qi, 26));

        for (let j = 0; j < 5; j++) {
          let mid = tmp;
          i64x2.mul(qiLo, constI64x2(Phi[j]));
          i64x2.mul(qiHi, constI64x2(Plo[j]));
          local.set(mid, i64x2.add());

          // Z[j] += lo + ((mid & mask26) << 26n);
          v128.and(mid, constI64x2(mask26));
          i64x2.shl($, 26);
          i64x2.mul(qiLo, constI64x2(Plo[j]));
          i64x2.add();
          local.set(Z[j], i64x2.add($, Z[j]));

          // Z[j + 1] += 2n * ((mid >> 26n) + hi);
          i64x2.shr_s(mid, 26);
          i64x2.mul(qiHi, constI64x2(Phi[j]));
          i64x2.add();
          i64x2.shl($, 1);
          local.set(Z[j + 1], i64x2.add($, Z[j + 1]));
        }

        local.set(Z[1], i64x2.add(Z[1], i64x2.shr_s(Z[0], 51)));
        for (let j = 0; j < 5; j++) {
          local.set(Z[j], Z[j + 1]);
        }
        local.set(Z[5], constI64x2(0n));
      }

      // carryLocals(Z);
      if (reduce) reduceLocals(Z, tmp, idx);
      if (carry) carryLocals(Z);

      // store in memory
      for (let i = 0; i < 5; i++) {
        v128.store({ offset: i * 16 }, z, Z[i]);
      }
    }
  );

  return { multiply, multiplyNoFma };
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
