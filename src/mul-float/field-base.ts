// import type * as W from "wasmati";
import {
  $,
  i32,
  i64,
  local,
  Local,
  StackVar,
  Input,
  v128,
  i64x2,
  f64,
} from "wasmati";
import { assert } from "../util.js";
import { bigintToFloat51Limbs, bigintToInt51Limbs, mask51 } from "./common.js";

export { createFieldBase, FieldBase, constF64x2, constI64x2 };

function constF64x2(x: number) {
  return v128.const("f64x2", [x, x]);
}
function constI64x2(x: bigint) {
  return v128.const("i64x2", [x, x]);
}

// inline methods to operate on a field element stored as n * w-bit limbs

type FieldBase = {
  modulus: bigint;
  size: number;

  forEach(cb: (i: number) => void): void;
  forEachReversed(cb: (i: number) => void): void;

  loadLimb(ptr: Local<i32>, i: number): StackVar<v128>;
  storeLimb(ptr: Local<i32>, i: number, value: Input<v128>): void;

  load(X: Local<v128>[], x: Local<i32>): void;
  store(x: Local<i32>, X: Input<v128>[]): void;

  i64x2: {
    P: BigUint64Array;
    P2: BigUint64Array;
    p(i: number): StackVar<v128>;
    p2(i: number): StackVar<v128>;

    loadLane(x: Local<i32>, i: number, lane: 0 | 1): StackVar<i64>;
    loadLimbFirst(x: Local<i32>, i: number): StackVar<i64>;
    loadLimbSecond(x: Local<i32>, i: number): StackVar<i64>;

    const(x: bigint): StackVar<v128>;
    carry(carry: StackVar<v128>, tmp: Local<v128>): void;
  };
  f64x2: {
    P: Float64Array;
    P2: Float64Array;
    p(i: number): StackVar<v128>;
    p2(i: number): StackVar<v128>;

    loadLane(x: Local<i32>, i: number, lane: 0 | 1): StackVar<f64>;

    const(x: number): StackVar<v128>;
  };
};

const n = 5;

function createFieldBase(p: bigint): FieldBase {
  const size = 8 * n; // size in bytes

  function loadLimb(x: Local<i32>, i: number) {
    assert(i >= 0, "positive index");
    return v128.load({ offset: 16 * i }, x);
  }
  function storeLimb(x: Local<i32>, i: number, xi: Input<v128>) {
    assert(i >= 0, "positive index");
    v128.store({ offset: 16 * i }, x, xi);
  }

  function forEach(callback: (i: number) => void) {
    for (let i = 0; i < n; i++) {
      callback(i);
    }
  }
  function forEachReversed(callback: (i: number) => void) {
    for (let i = n - 1; i >= 0; i--) {
      callback(i);
    }
  }

  function load(X: Local<v128>[], x: Local<i32>) {
    for (let i = 0; i < n; i++) {
      local.set(X[i], v128.load({ offset: 16 * i }, x));
    }
  }

  function store(x: Local<i32>, X: Input<v128>[]) {
    for (let i = 0; i < n; i++) {
      v128.store({ offset: 16 * i }, x, X[i]);
    }
  }

  function carryI(input: StackVar<v128>, tmp: Local<v128>) {
    // put carry on the stack
    local.tee(tmp, input);
    i64x2.shr_s($, 51);
    // mod 2^w the current result
    v128.and(tmp, constI64x2(mask51));
  }

  let PF = bigintToFloat51Limbs(p);
  let PF2 = bigintToFloat51Limbs(2n * p);

  let PI = bigintToInt51Limbs(p);
  let PI2 = bigintToInt51Limbs(2n * p);

  return {
    modulus: p,
    size,

    loadLimb,
    storeLimb,
    load,
    store,

    forEach,
    forEachReversed,

    i64x2: {
      P: PI,
      P2: PI2,
      p: (i: number) => constI64x2(PI[i]),
      p2: (i: number) => constI64x2(PI2[i]),

      loadLane(x, i, lane) {
        return i64.load({ offset: 16 * i + 8 * lane }, x);
      },
      loadLimbFirst(x, i) {
        return i64.load({ offset: 16 * i }, x);
      },
      loadLimbSecond(x, i) {
        return i64.load({ offset: 16 * i + 8 }, x);
      },
      const(x) {
        return constI64x2(x);
      },
      carry: carryI,
    },

    f64x2: {
      P: PF,
      P2: PF2,
      p: (i: number) => constF64x2(PF[i]),
      p2: (i: number) => constF64x2(PF2[i]),

      loadLane(x, i, lane) {
        return f64.load({ offset: 16 * i + 8 * lane }, x);
      },

      const(x) {
        return constF64x2(x);
      },
    },
  };
}
