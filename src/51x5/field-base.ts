import {
  i32,
  i64,
  local,
  Local,
  Input,
  v128,
  f64,
  StackVar,
  memory,
  func,
  Func,
} from "wasmati";
import { assert } from "../util.js";
import { bigintToInt51Limbs, mask51 } from "./common.js";

export {
  Constants,
  FieldPair,
  createField,
  FieldBase,
  constF64x2,
  constI64x2,
  forEach,
  forEachReversed,
  bigintPairToData,
  bigintToData,
};

function constF64x2(x: number, y?: number): StackVar<v128> {
  return v128.const("f64x2", [x, y ?? x]);
}
function constI64x2(x: bigint, y?: bigint): StackVar<v128> {
  return v128.const("i64x2", [x, y ?? x]);
}

const n = 5;
const sizePair = 16 * n;
const sizeSingle = 8 * n;

function copyInline(x: Local<i32>, y: Local<i32>) {
  local.get(x);
  local.get(y);
  i32.const(sizePair);
  memory.copy();
}

const FieldPair = {
  size: 16 * n,

  loadLimb(x: Local<i32>, i: number) {
    assert(i >= 0, "positive index");
    return v128.load({ offset: 16 * i }, x);
  },
  storeLimb(x: Local<i32>, i: number, xi: Input<v128>) {
    assert(i >= 0, "positive index");
    v128.store({ offset: 16 * i }, x, xi);
  },
  load(X: Local<v128>[], x: Local<i32>) {
    for (let i = 0; i < n; i++) {
      local.set(X[i], v128.load({ offset: 16 * i }, x));
    }
  },
  store(x: Local<i32>, X: Input<v128>[]) {
    for (let i = 0; i < n; i++) {
      v128.store({ offset: 16 * i }, x, X[i]);
    }
  },
  copyInline,
  copy: func({ in: [i32, i32], out: [] }, ([x, y]) => {
    copyInline(x, y);
  }),
  forEach,
  forEachReversed,
  i64: {
    loadLane(x: Local<i32>, i: number, lane: 0 | 1) {
      return i64.load({ offset: 16 * i + 8 * lane }, x);
    },
    storeLane(x: Local<i32>, i: number, lane: 0 | 1, xi: Input<i64>) {
      i64.store({ offset: 16 * i + 8 * lane }, x, xi);
    },
  },
  f64: {
    loadLane(x: Local<i32>, i: number, lane: 0 | 1) {
      return f64.load({ offset: 16 * i + 8 * lane }, x);
    },
    storeLane(x: Local<i32>, i: number, lane: 0 | 1, xi: Input<f64>) {
      f64.store({ offset: 16 * i + 8 * lane }, x, xi);
    },
  },
};

type FieldLayout = "single" | ["lane", 0 | 1];

function layout(type: FieldLayout) {
  switch (type) {
    case "single":
      return { limbGap: 8, limbOffset: 0, size: 8 * n, lane: 0 };
    default:
      return {
        limbGap: 16,
        limbOffset: 8 * type[1],
        size: 16 * n,
        lane: type[1],
      };
  }
}

type FieldBase = ReturnType<typeof createField>;

function createField(type: FieldLayout, p: bigint) {
  let { limbGap, limbOffset, size } = layout(type);

  function copyInline(x: Local<i32>, y: Local<i32>) {
    local.get(x);
    local.get(y);
    i32.const(size);
    memory.copy();
  }

  return {
    size,
    n,
    p,

    i64: {
      size,
      n,
      p,
      ...Constants(p).i64,

      loadLimb(x: Local<i32>, i: number) {
        assert(i >= 0, "positive index");
        return i64.load({ offset: limbGap * i + limbOffset }, x);
      },
      storeLimb(x: Local<i32>, i: number, xi: Input<i64>) {
        assert(i >= 0, "positive index");
        i64.store({ offset: limbGap * i + limbOffset }, x, xi);
      },
      load(X: Local<i64>[], x: Local<i32>) {
        for (let i = 0; i < n; i++) {
          local.set(X[i], i64.load({ offset: limbGap * i + limbOffset }, x));
        }
      },
      store(x: Local<i32>, X: Input<i64>[]) {
        for (let i = 0; i < n; i++) {
          i64.store({ offset: limbGap * i + limbOffset }, x, X[i]);
        }
      },
      forEach,
      forEachReversed,
      copyInline,
    },
  };
}

function Constants(p: bigint) {
  return {
    i64: {
      P: [...bigintToInt51Limbs(p)],
      P2: [...bigintToInt51Limbs(2n * p)],
      Zero: [...bigintToInt51Limbs(0n)],
      One: [...bigintToInt51Limbs(1n)],
    },
  };
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

function bigintToData(x0: bigint) {
  let bytes = new Uint8Array(sizeSingle);
  let view = new DataView(bytes.buffer);

  for (let offset = 0; offset < sizeSingle; offset += 8) {
    view.setBigInt64(offset, x0 & mask51, true);
    x0 >>= 51n;
  }
  return [...bytes];
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
