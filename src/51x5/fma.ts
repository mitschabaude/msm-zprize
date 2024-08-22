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
  Local,
  i8x16,
  Type,
  Input,
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
import { arithmetic, carryLocals, carryLocalsSingle } from "./arith.js";
import { assert } from "../util.js";
import { forLoop, forLoop4 } from "../wasm/wasm-util.js";

export { Multiply };

type Multiply = {
  multiply: Func<["i32", "i32", "i32"], []>;
  multiplyNoFma: Func<["i32", "i32", "i32"], []>;
  multiplySingle: Func<["i32", "i32", "i32"], []>;
};

let zInitial = new BigInt64Array(11);
let loCount = [1n, 2n, 3n, 4n, 5n, 4n, 3n, 2n, 1n, 0n];
let hiCount = [0n, 1n, 2n, 3n, 4n, 5n, 4n, 3n, 2n, 1n];
for (let i = 0; i < 10; i++) {
  zInitial[i] = -((2n * (hiCount[i] * hiPre + loCount[i] * loPre)) & mask64);
}

let shift26 = 1n << 26n;
let mask26 = (1n << 26n) - 1n;
let shift25 = 1n << 25n;
let mask25 = (1n << 25n) - 1n;

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

  let { reduceLocals, reduceLocalsSingle } = arithmetic(p, pSelectPtr);

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

  let nLocalsI64 = [i64, i64, i64, i64, i64] as const;
  let threeV128 = [v128, v128, v128] as const;

  /**
   * 51x5 multiplication of two _single_, packed field elements
   *
   * we use 64x2 operations on two elements of a 5- or 6-limb vector
   *
   * the layout of limbs as three v128s is as follows:
   * (0 3), (1 4), (2 5)
   * i.e. the first v128 contains the limbs with indices 0 and 3
   *
   * this layout makes it fairly cheap to do shifts by 1 limb, like when assigning Z[j+1] += LH[j]:
   * Z[1,4] = Z[1,4] + LH[0,3]
   * Z[2,5] = Z[2,5] + LH[1,4]
   * Z[0,3] = Z[0,3] + swap(LH[2,5])
   * where swap() swaps the to 64-bit halves of a v128, and LH[5] = 0
   *
   * for 5-limb vectors, the last limb is not used and is assigned 0, i.e.
   * (0 3), (1 4), (2 *)
   *
   * the layout in memory expected from x, y is (0, 1, 2, 3, 4) i.e. 5 consecutive 64-bit limbs
   */
  let multiplySingleSlow = func(
    {
      in: [i32, i32, i32], // pointers to z, x, y, where z = x * y
      out: [],
      locals: [v128, ...threeV128, ...threeV128, ...threeV128, ...nLocalsI64],
    },
    ([z, x, y], [l128, ...rest]) => {
      let Y = rest.slice(0, 3) as Local<v128>[];
      let LH = rest.slice(3, 6) as Local<v128>[];
      let Z = rest.slice(6, 9) as Local<v128>[];
      let Z5 = rest.slice(9, 14) as Local<i64>[];

      let layout = [
        [0, 3],
        [1, 4],
        [2, 5],
      ];

      let [P0, P1, P2, P3, P4] = PF;
      let P = [
        [P0, P3],
        [P1, P4],
        [P2, 0],
      ] as const;

      // load y from memory into locals
      layout.forEach(([j, k], i) => {
        if (k === 5) {
          v128.load64_zero({ offset: j * 8 }, y);
          local.set(Y[i]);
          return;
        }
        local.get(y);
        local.get(Y[i]);
        v128.load64_lane({ offset: j * 8 }, 0);
        local.set(Y[i]);
        local.get(y);
        local.get(Y[i]);
        v128.load64_lane({ offset: k * 8 }, 0);
        local.set(Y[i]);
      });

      // initialize Z with constants that offset float64 prefixes
      layout.forEach(([j, k], i) => {
        local.set(Z[i], constI64x2(zInitial[j], zInitial[k]));
      });

      for (let i = 0; i < 5; i++) {
        let xi = l128;
        local.set(xi, v128.load64_splat({ offset: i * 8 }, x));

        // hi; note LH[5] = xi*0 + 0 = 0
        local.set(LH[0], f64x2.relaxed_madd(xi, Y[0], constF64x2(c103)));
        local.set(LH[1], f64x2.relaxed_madd(xi, Y[1], constF64x2(c103)));
        local.set(LH[2], f64x2.relaxed_madd(xi, Y[2], constF64x2(c103, 0)));
        // Z[j+1] += hi, and Z[0] += LH[5] = 0
        local.set(Z[1], i64x2.add(Z[1], LH[0]));
        local.set(Z[2], i64x2.add(Z[2], LH[1]));
        local.set(Z[0], i64x2.add(Z[0], swap64x2(LH[2])));
        // lo sub; maintains LH[5] = 0
        local.set(LH[0], f64x2.sub(constF64x2(c2), LH[0]));
        local.set(LH[1], f64x2.sub(constF64x2(c2), LH[1]));
        local.set(LH[2], f64x2.sub(constF64x2(c2, 0), LH[2]));
        // lo; LH[5] = xi*0 + 0 = 0
        for (let j = 0; j < 3; j++)
          local.set(LH[j], f64x2.relaxed_madd(xi, Y[j], LH[j]));
        // Z[j] += lo and Z[5] += LH[5] = 0
        for (let j = 0; j < 3; j++) local.set(Z[j], i64x2.add(Z[j], LH[j]));

        // compute qi
        let qi = l128;
        local.get(Z[0]);
        i64.mul(i64x2.extract_lane(0), pInv);
        i64.and($, mask51);
        i64.add($, c51n);
        i64x2.splat();
        f64x2.sub($, constF64x2(c51));
        local.set(qi);

        // hi; note LH[5] = qi*0 + 0 = 0
        local.set(
          LH[0],
          f64x2.relaxed_madd(qi, constF64x2(...P[0]), constF64x2(c103))
        );
        local.set(
          LH[1],
          f64x2.relaxed_madd(qi, constF64x2(...P[1]), constF64x2(c103))
        );
        local.set(
          LH[2],
          f64x2.relaxed_madd(qi, constF64x2(...P[2]), constF64x2(c103, 0))
        );
        // Z[j+1] += hi, and Z[0] += LH[5] = 0
        local.set(Z[1], i64x2.add(Z[1], LH[0]));
        local.set(Z[2], i64x2.add(Z[2], LH[1]));
        local.set(Z[0], i64x2.add(Z[0], swap64x2(LH[2])));
        // lo sub; maintains LH[5] = 0
        local.set(LH[0], f64x2.sub(constF64x2(c2), LH[0]));
        local.set(LH[1], f64x2.sub(constF64x2(c2), LH[1]));
        local.set(LH[2], f64x2.sub(constF64x2(c2, 0), LH[2]));
        // lo; LH[5] = qi*0 + 0 = 0
        for (let j = 0; j < 3; j++)
          local.set(
            LH[j],
            f64x2.relaxed_madd(qi, constF64x2(P[j][0], P[j][1]), LH[j])
          );
        // Z[j] += lo and Z[5] += LH[5] = 0
        for (let j = 0; j < 3; j++) local.set(Z[j], i64x2.add(Z[j], LH[j]));

        // Z[1,4] += [carry, 0]
        constI64x2(0n);
        local.get(Z[0]);
        i64x2.extract_lane(0);
        i64.shr_s($, 51n);
        i64x2.replace_lane(0);
        local.set(Z[1], i64x2.add($, Z[1]));

        constI64x2(zInitial[6 + i]);
        local.get(Z[0]);
        i64x2.extract_lane(1);
        i64x2.replace_lane(0);
        local.set(Z[0]); // Z[3, *]

        // shift down
        local.set(Z[0], Z[1]); // Z[0, 3] = Z[1, 4]
        local.set(Z[1], Z[2]); // Z[1, 4] = Z[2, 5]
        local.set(Z[2], Z[0]); // Z[2, 5] = Z[3, *]
      }

      // read into Z5
      layout.forEach(([j, k], i) => {
        local.get(Z[i]);
        i64x2.extract_lane(0);
        local.set(Z5[j]);
        if (k === 5) return;
        local.get(Z[i]);
        i64x2.extract_lane(1);
        local.set(Z5[k]);
      });

      if (reduce) reduceLocalsSingle(Z5);
      if (carry) carryLocalsSingle(Z5);

      // convert output to f64
      if (convert)
        for (let i = 0; i < 5; i++) {
          i64.add(Z5[i], c52n);
          i64x2.splat();
          f64x2.sub($, constF64x2(c52));
          i64x2.extract_lane(0);
          local.set(Z5[i], $);
        }

      // store in memory
      for (let i = 0; i < 5; i++) {
        i64.store({ offset: i * 8 }, z, Z5[i]);
      }
    }
  );

  let PI = bigintToInt51Limbs(p);
  let Plo = PI.map((x) => x & mask26);
  let Phi = PI.map((x) => x >> 26n);

  let localsI64 = (n: number) => Array<Type<i64>>(n).fill(i64);
  let P = [...Plo].flatMap((lo, i) => [lo, Phi[i]]);

  let pInv26 = inverse(-p, 1n << 26n);
  let pInv25 = inverse(-p, 1n << 25n);

  function computeQ(z: Input<i64>, pInv: bigint, mask: bigint) {
    if (pInv === mask) {
      // p = 1 mod 2^w  <==> -p^(-1) = -1 mod 2^w
      // qi = z * (-1) % 2^w = (2^w - z) % 2^w
      return i64.and(i64.sub(mask + 1n, i64.and(z, mask)), mask);
    }
    return i64.and(i64.mul(i64.and(z, mask), pInv), mask);
  }

  let multiplySingle = (limbGap: number, limbOffset: number) =>
    func(
      {
        in: [i32, i32, i32],
        locals: [i64, i64, i64, i64, ...localsI64(10), ...localsI64(9)],
        out: [],
      },
      ([xy, x, y], [tmp, qi, xix2, xi, ...rest]) => {
        let Y = rest.slice(0, 10);
        let Z = rest.slice(10, 19);

        // load y from memory into locals
        for (let i = 0; i < 5; i++) {
          local.set(xi, i64.load({ offset: i * limbGap + limbOffset }, y));
          local.set(Y[2 * i], i64.and(xi, mask26));
          local.set(Y[2 * i + 1], i64.shr_s(xi, 26n));
        }

        for (let i = 0; i < 5; i++) {
          local.set(xix2, i64.load({ offset: i * limbGap + limbOffset }, x));

          // LOWER HALF
          local.set(xi, i64.and(xix2, mask26));

          local.set(tmp, i64.add(i64.mul(xi, Y[0]), Z[0]));
          local.set(qi, computeQ(tmp, pInv26, mask26));
          local.get(tmp);
          addMul(qi, P[0]);
          i64.shr_s($, 26n);

          for (let j = 1; j < 9; j++) {
            i64.mul(xi, Y[j]);
            if (j === 1) i64.add();
            addMul(qi, P[j]);
            local.get(Z[j]);
            i64.add();
            local.set(Z[j - 1]);
          }
          i64.mul(xi, Y[9]);
          addMul(qi, P[9]);
          local.set(Z[8]);

          // UPPER HALF
          local.set(xi, i64.shr_s(xix2, 26n));

          local.set(tmp, i64.add(i64.mul(xi, Y[0]), Z[0]));
          local.set(qi, computeQ(tmp, pInv25, mask25));
          local.get(tmp);
          addMul(qi, P[0]);
          i64.shr_s($, 25n);
          local.set(Z[1], i64.add($, Z[1]));

          for (let j = 1; j < 9; j++) {
            i64.mul(xi, Y[j]);
            addMul(qi, P[j]);
            if (j % 2 === 1) i64.shl($, 1n);
            local.get(Z[j]);
            i64.add();
            local.set(Z[j - 1]);
          }
          i64.mul(xi, Y[9]);
          addMul(qi, P[9]);
          i64.shl($, 1n);
          local.set(Z[8]);
        }

        // final pass of collecting carries, store output in memory
        for (let i = 0; i < 4; i++) {
          local.get(Z[2 * i]);
          if (i > 0) i64.add(); // add carry
          i64.and(Z[2 * i + 1], mask25);
          i64.shl($, 26n);
          local.set(tmp, i64.add());

          // store 51 bits at a time
          i64.and(tmp, mask51);
          i64.store({ offset: i * limbGap + limbOffset }, xy, $);

          // put carry on the stack
          i64.shr_s(tmp, 51n);
          i64.shr_s(Z[2 * i + 1], 25n);
          i64.add();
        }
        // final iteration simpler because X[9] is not used
        local.get(Z[8]);
        i64.add(); // add carry
        i64.store({ offset: 4 * limbGap + limbOffset }, xy, $);
      }
    );

  let multiplyNoFma0 = multiplySingle(16, 0);
  let multiplyNoFma1 = multiplySingle(16, 8);

  let multiplyNoFma = func(
    {
      in: [i32, i32, i32],
      out: [],
    },
    ([xy, x, y]) => {
      call(multiplyNoFma0, [xy, x, y]);
      call(multiplyNoFma1, [xy, x, y]);
    }
  );

  function computeQx2(z: Input<v128>, pInv: bigint, mask: bigint) {
    if (pInv === mask) {
      // p = 1 mod 2^w  <==> -p^(-1) = -1 mod 2^w
      // qi = z * (-1) % 2^w = (2^w - z) % 2^w
      return v128.and(
        i64x2.sub(constI64x2(mask + 1n), v128.and(z, constI64x2(mask))),
        constI64x2(mask)
      );
    }
    return v128.and(
      i64x2.mul(v128.and(z, constI64x2(mask)), constI64x2(pInv)),
      constI64x2(mask)
    );
  }

  let localsV128 = (n: number) => Array<Type<v128>>(n).fill(v128);

  let multiplyNoFmaSimd = func(
    {
      in: [i32, i32, i32],
      locals: [v128, v128, v128, v128, ...localsV128(10), ...localsV128(9)],
      out: [],
    },
    ([xy, x, y], [tmp, qi, xix2, xi, ...rest]) => {
      let Y = rest.slice(0, 10);
      let Z = rest.slice(10, 19);

      // load y from memory into locals
      for (let i = 0; i < 5; i++) {
        local.set(xi, v128.load({ offset: i * 16 }, y));
        local.set(Y[2 * i], v128.and(xi, constI64x2(mask26)));
        local.set(Y[2 * i + 1], i64x2.shr_s(xi, 26));
      }

      for (let i = 0; i < 5; i++) {
        local.set(xix2, v128.load({ offset: i * 16 }, x));

        // LOWER HALF
        local.set(xi, v128.and(xix2, constI64x2(mask26)));

        local.set(tmp, i64x2.add(i64x2.mul(xi, Y[0]), Z[0]));
        local.set(qi, computeQx2(tmp, pInv26, mask26));
        local.get(tmp);
        addMulx2(qi, P[0]);
        i64x2.shr_s($, 26);

        for (let j = 1; j < 9; j++) {
          i64x2.mul(xi, Y[j]);
          if (j === 1) i64x2.add();
          addMulx2(qi, P[j]);
          local.get(Z[j]);
          i64x2.add();
          local.set(Z[j - 1]);
        }
        i64x2.mul(xi, Y[9]);
        addMulx2(qi, P[9]);
        local.set(Z[8]);

        // UPPER HALF
        local.set(xi, i64x2.shr_s(xix2, 26));

        local.set(tmp, i64x2.add(i64x2.mul(xi, Y[0]), Z[0]));
        local.set(qi, computeQx2(tmp, pInv25, mask25));
        local.get(tmp);
        addMulx2(qi, P[0]);
        i64x2.shr_s($, 25);
        local.set(Z[1], i64x2.add($, Z[1]));

        for (let j = 1; j < 9; j++) {
          i64x2.mul(xi, Y[j]);
          addMulx2(qi, P[j]);
          if (j % 2 === 1) i64x2.shl($, 1);
          local.get(Z[j]);
          i64x2.add();
          local.set(Z[j - 1]);
        }
        i64x2.mul(xi, Y[9]);
        addMulx2(qi, P[9]);
        i64x2.shl($, 1);
        local.set(Z[8]);
      }

      // final pass of collecting carries, store output in memory
      for (let i = 0; i < 4; i++) {
        local.get(Z[2 * i]);
        if (i > 0) i64x2.add(); // add carry
        v128.and(Z[2 * i + 1], constI64x2(mask25));
        i64x2.shl($, 26);
        local.set(tmp, i64x2.add());

        // store 51 bits at a time
        v128.and(tmp, constI64x2(mask51));
        v128.store({ offset: i * 16 }, xy, $);

        // put carry on the stack
        i64x2.shr_s(tmp, 51);
        i64x2.shr_s(Z[2 * i + 1], 25);
        i64x2.add();
      }
      // final iteration simpler because X[9] is not used
      local.get(Z[8]);
      i64x2.add(); // add carry
      v128.store({ offset: 4 * 16 }, xy, $);
    }
  );

  /**
   * 51x5 multiplication without using FMA instructions, as a fallback for compatibility.
   */
  let multiplyNoFmaSlow = func(
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

      if (reduce) reduceLocals(Z, tmp, idx);
      if (carry) carryLocals(Z);

      // store in memory
      for (let i = 0; i < 5; i++) {
        v128.store({ offset: i * 16 }, z, Z[i]);
      }
    }
  );

  return { multiply, multiplyNoFma, multiplySingle: multiplySingle(8, 0) };
}

function swap64x2(z: Local<v128>) {
  return i8x16.swizzle(
    z,
    v128.const("i8x16", [8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7])
  );
}

function addMul(l: Local<i64>, c: bigint) {
  if (c === 0n) return;
  if (c === 1n) {
    i64.add($, l);
    return;
  }
  i64.mul(l, c);
  i64.add();
}

function addMulx2(l: Local<v128>, c: bigint) {
  if (c === 0n) return;
  if (c === 1n) {
    i64x2.add($, l);
    return;
  }
  i64x2.mul(l, constI64x2(c));
  i64x2.add();
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
