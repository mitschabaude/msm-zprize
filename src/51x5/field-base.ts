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

export {
  Field,
  constF64x2,
  constI64x2,
  loadLimb,
  storeLimb,
  forEach,
  forEachReversed,
  load,
  store,
  I64x2,
  F64x2,
  i64x2Constants,
  f64x2Constants,
  bigintPairToData,
};

function constF64x2(x: number, y?: number) {
  return v128.const("f64x2", [x, y ?? x]);
}
function constI64x2(x: bigint, y?: bigint) {
  return v128.const("i64x2", [x, y ?? x]);
}

// inline methods to operate on a field element stored as n * w-bit limbs

const n = 5;
const sizePair = 16 * n;

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

function bigintPairToData(x0: bigint, x1: bigint) {
  let bytes = new Uint8Array(sizePair);
  let view = new DataView(bytes.buffer);

  for (let offset = 0; offset < sizePair; offset += 16) {
    view.setBigInt64(offset, x0 & mask51, true);
    view.setBigInt64(offset + 8, x1 & mask51, true);
    x0 >>= 51n;
    x1 >>= 51n;
  }
  return [...bytes];
}

const I64x2 = {
  loadLane(x: Local<i32>, i: number, lane: 0 | 1) {
    return i64.load({ offset: 16 * i + 8 * lane }, x);
  },
  storeLane(x: Local<i32>, i: number, lane: 0 | 1, xi: Input<i64>) {
    i64.store({ offset: 16 * i + 8 * lane }, x, xi);
  },

  const(x: bigint) {
    return constI64x2(x);
  },
};

const F64x2 = {
  loadLane(x: Local<i32>, i: number, lane: 0 | 1) {
    return f64.load({ offset: 16 * i + 8 * lane }, x);
  },
  storeLane(x: Local<i32>, i: number, lane: 0 | 1, xi: Input<f64>) {
    f64.store({ offset: 16 * i + 8 * lane }, x, xi);
  },
};

function i64x2Constants(p: bigint) {
  let P = bigintToInt51Limbs(p);
  let P2 = bigintToInt51Limbs(2n * p);
  return {
    P,
    P2,
    p: (i: number) => constI64x2(P[i]),
    p2: (i: number) => constI64x2(P2[i]),
  };
}

function f64x2Constants(p: bigint) {
  let P = bigintToFloat51Limbs(p);
  let P2 = bigintToFloat51Limbs(2n * p);
  return {
    P,
    P2,
    p: (i: number) => constF64x2(P[i]),
    p2: (i: number) => constF64x2(P2[i]),
  };
}

const Field = {
  size: 16 * n,
  loadLimb,
  storeLimb,
  forEach,
  forEachReversed,
  load,
  store,
  i64x2: I64x2,
  f64x2: F64x2,
  i64x2Constants,
  f64x2Constants,
};
