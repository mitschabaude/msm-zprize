import type * as W from "wasmati"; // for type names
import { Module, func, i32, memory, call, if_, local } from "wasmati";
import { FieldWithArithmetic } from "./wasm/field-arithmetic.js";
import { fieldInverse } from "./wasm/inverse.js";
import { multiplyMontgomery } from "./wasm/multiply-montgomery.js";
import { ImplicitMemory, forLoop1 } from "./wasm/wasm-util.js";
import { mod, montgomeryParams } from "./field-util.js";
import { curveOps } from "./wasm/curve.js";
import { memoryHelpers } from "./wasm/helpers.js";
import { fromPackedBytes, toPackedBytes } from "./wasm/field-helpers.js";
import { UnwrapPromise } from "./types.js";
import { assert } from "./util.js";

export { createMsmField, MsmField };

type MsmField = UnwrapPromise<ReturnType<typeof createMsmField>>;

async function createMsmField(p: bigint, beta: bigint, w: number) {
  let { K, R, lengthP: N, n, nPackedBytes } = montgomeryParams(p, w);

  let implicitMemory = new ImplicitMemory(memory({ min: 1 << 16 }));

  let Field_ = FieldWithArithmetic(p, w);
  let { multiply, square, leftShift } = multiplyMontgomery(p, w, {
    countMultiplications: false,
  });
  const Field = Object.assign(Field_, { multiply, square, leftShift });

  let { inverse, makeOdd, batchInverse } = fieldInverse(implicitMemory, Field);
  let { addAffine, endomorphism } = curveOps(
    implicitMemory,
    Field,
    inverse,
    beta
  );

  let {
    isEqual,
    isGreater,
    isZero,
    add,
    subtract,
    subtractPositive,
    reduce,
    copy,
  } = Field;

  let mg1 = implicitMemory.data(Field.bigintToData(mod(1n * R, p)));

  /**
   * z = x^n mod p
   *
   * z, x are passed in montgomery form, n as a plain field element
   */
  const power = func(
    { in: [i32, i32, i32, i32], locals: [i32, i32], out: [] },
    ([x, z, xIn, n], [j, ni]) => {
      call(copy, [z, mg1]);
      call(copy, [x, xIn]);
      Field.forEach((i) => {
        local.set(ni, Field.loadLimb32(n, i));
        forLoop1(j, 0, w, () => {
          i32.and(ni, i32.shl(1, j));
          if_(null, () => {
            call(multiply, [z, z, x]);
          });
          call(square, [x, x]);
        });
      });
    }
  );

  let module = Module({
    exports: {
      ...implicitMemory.getExports(),
      // curve ops
      addAffine,
      endomorphism,
      // multiplication
      multiply,
      square,
      leftShift,
      power,
      // inverse
      inverse,
      makeOdd,
      batchInverse,
      // arithmetic
      add,
      subtract,
      subtractPositive,
      reduce,
      copy,
      // helpers
      isEqual,
      isGreater,
      isZero,
      fromPackedBytes: fromPackedBytes(w, n),
      toPackedBytes: toPackedBytes(w, n, nPackedBytes),
    },
  });

  let wasm = (await module.instantiate()).instance.exports;
  let helpers = memoryHelpers(p, w, wasm);

  // precomputed constants for tonelli-shanks
  let t = p - 1n;
  let S = 0;
  while ((t & 1n) === 0n) {
    t >>= 1n;
    S++;
  }
  let t0 = (p - 1n) >> 1n;
  let t1 = (t - 1n) / 2n;

  // put some constants in wasm memory

  let constantsBigint = {
    zero: 0n,
    one: 1n,
    p,
    R: mod(R, p),
    R2: mod(R * R, p),
    R2corr: mod(1n << BigInt(4 * K - 2 * N + 1), p),
    // common numbers in montgomery representation
    mg1: mod(1n * R, p),
    mg2: mod(2n * R, p),
    mg4: mod(4n * R, p),
    mg8: mod(8n * R, p),
    // for sqrt
    t1,
  };
  let constantsKeys = Object.keys(constantsBigint);
  let constantsPointers = helpers.getStablePointers(constantsKeys.length);

  let constants = Object.fromEntries(
    constantsKeys.map((key, i) => {
      let pointer = constantsPointers[i];
      helpers.writeBigint(
        pointer,
        constantsBigint[key as keyof typeof constantsBigint]
      );
      return [key, pointer];
    })
  ) as Record<keyof typeof constantsBigint, number>;

  function fromMontgomery(x: number) {
    wasm.multiply(x, x, constants.one);
    wasm.reduce(x);
  }
  function toMontgomery(x: number) {
    wasm.multiply(x, x, constants.R2);
  }

  let memoryBytes = new Uint8Array(wasm.memory.buffer);

  // higher level finite field algorithms

  function pow(
    [scratch, n0]: number[],
    z: number,
    x: number,
    n: bigint | number
  ) {
    helpers.writeBigint(n0, BigInt(n));
    wasm.power(scratch, z, x, n0);
  }

  // find z = non square
  // start with z = 2
  let [z, zp, ...scratch] = helpers.getPointers(5);
  wasm.copy(z, constants.mg2);

  while (true) {
    // Euler's criterion, test z^(p-1)/2 = 1
    pow(scratch, zp, z, t0);
    wasm.reduce(zp);
    let isSquare = wasm.isEqual(zp, constants.mg1);
    if (!isSquare) break;
    // z++
    wasm.add(z, z, constants.mg1);
  }

  // roots of unity w = z^t, w^2, ..., w^(2^(S-1)) = -1
  let roots = helpers.getStablePointers(S);
  pow(scratch, roots[0], z, t);
  for (let i = 1; i < S; i++) {
    wasm.square(roots[i], roots[i - 1]);
  }

  /**
   * square root, sqrtx^2 === x mod p
   *
   * returns boolean that indicates whether the square root exists
   *
   * can use the same pointer for sqrtx and x
   *
   * Algorithm: https://en.wikipedia.org/wiki/Tonelli-Shanks_algorithm#The_algorithm
   *
   * note: atm, the exponentiation x^(t-1)/2 takes about 2/3 of the time here (and seems hard to improve)
   * probably possible optimize the second part of the algorithm with more caching
   */
  function sqrt([u, s, scratch]: number[], sqrtx: number, x: number) {
    if (wasm.isZero(x)) {
      wasm.copy(sqrtx, constants.zero);
      return true;
    }
    let i = S;
    // t1 is (t-1)/2, where t is the odd factor in p-1
    wasm.power(scratch, u, x, constants.t1); // u = x^((t-1)/2)
    wasm.multiply(sqrtx, u, x); // sqrtx = x^((t+1)/2) = u * x
    wasm.multiply(u, u, sqrtx); // u = x^t = x^((t-1)/2) * x^((t+1)/2) = u * sqrtx

    while (true) {
      // if u === 1, we're done
      if (wasm.isEqual(u, constants.mg1)) return true;

      // use repeated squaring to find the least i', 0 < i' < i, such that u^(2^i') = 1
      let i_ = 1;
      wasm.square(s, u);
      while (!wasm.isEqual(s, constants.mg1)) {
        wasm.square(s, s);
        i_++;
      }
      if (i_ === i) return false; // no solution
      assert(i_ < i); // by construction
      i = i_;
      wasm.multiply(sqrtx, sqrtx, roots[S - i - 1]); // sqrtx *= b = w^(2^(S - i - 1))
      wasm.multiply(u, u, roots[S - i]); // u *= b^2
    }
  }

  return {
    p,
    w,
    t,
    ...wasm,
    /**
     * affine EC addition, G3 = G1 + G2
     *
     * assuming d = 1/(x2 - x1) is given, and inputs aren't zero, and x1 !== x2
     * (edge cases are handled one level higher, before batching)
     *
     * this supports addition with assignment where G3 === G1 (but not G3 === G2)
     * @param scratch
     * @param G3 (x3, y3)
     * @param G1 (x1, y1)
     * @param G2 (x2, y2)
     * @param d 1/(x2 - x1)
     */
    addAffine: wasm.addAffine,
    /**
     * montgomery inverse, a 2^K -> a^(-1) 2^K (mod p)
     *
     * needs 3 fields of scratch space
     */
    inverse: wasm.inverse,
    ...helpers,
    constants,
    roots,
    memoryBytes,
    toMontgomery,
    fromMontgomery,
    sqrt,
    toBigint(x: number) {
      fromMontgomery(x);
      let x0 = helpers.readBigint(x);
      toMontgomery(x);
      return x0;
    },
  };
}
