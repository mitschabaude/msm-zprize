/**
 * Basic arithmetic for 51x5 fields
 */
// import type * as W from "wasmati";
import {
  $,
  block,
  br_if,
  drop,
  func,
  i32,
  i64,
  v128,
  if_,
  local,
  memory,
  return_,
  Local,
  StackVar,
  i64x2,
  f64x2,
  type Func,
  type Input,
} from "wasmati";
import { constI64x2, createFieldBase, FieldBase } from "./field-base.js";

export { arithmetic, fieldHelpers, FieldWithArithmetic };

type FieldWithArithmetic = ReturnType<typeof FieldWithArithmetic>;

function FieldWithArithmetic(p: bigint) {
  const Field = createFieldBase(p);
  const arithmetic_ = arithmetic(Field);
  const helper_ = fieldHelpers(Field);
  return { ...Field, ...arithmetic_, ...helper_ };
}

// TODO all of this doesn't work

function arithmetic(Field: FieldBase) {
  /**
   * reduce in place from < 2*p to < p, i.e.
   * if (x > p) x -= p
   */
  const reduceI = func(
    { in: [i32, i32], locals: [v128], out: [] },
    ([out, x], [tmp]) => {
      // first loop: x - p
      Field.forEach((i) => {
        // (carry, out[i]) = x[i] - p[i] + carry;
        Field.loadLimb(x, i);
        if (i > 0) i64x2.add();
        i64x2.sub($, Field.i64x2.p(i));
        Field.i64x2.carry($, tmp);
        Field.storeLimb(out, i, $);
      });
      // check if we underflowed by checking carry === 0 (in that case, we didn't and can return)
      i64x2.eq($, constI64x2(0n));
      if_(null, () => return_());
      // second loop
      // if we're here, y > x and out = x - y + R, while we want x - p + p
      // so do (out += 2p) and ignore the known overflow of R
      Field.forEach((i) => {
        // (carry, out[i]) = (2*p)[i] + out[i] + carry;
        Field.i64x2.p(i);
        if (i > 0) f64x2.add();
        Field.loadLimb(out, i);
        f64x2.add();
        Field.i64x2.carry($, tmp);
        Field.storeLimb(out, i, $);
      });
      drop();
    }
  );

  const additionFNoCarry = func(
    { in: [i32, i32, i32], out: [] },
    ([out, x, y]) => {
      Field.forEach((i) => {
        let xi = Field.loadLimb(x, i);
        let yi = Field.loadLimb(y, i);
        f64x2.add(xi, yi);
        Field.storeLimb(out, i, $);
      });
    }
  );

  const subtractionI = (doReduce: boolean) =>
    func(
      { in: [i32, i32, i32], locals: [v128], out: [] },
      ([out, x, y], [tmp]) => {
        // first loop: x - y
        Field.forEach((i) => {
          // (carry, out[i]) = x[i] - y[i] + carry;
          Field.loadLimb(x, i);
          if (i > 0) i64x2.add();
          Field.loadLimb(y, i);
          i64x2.sub();
          Field.i64x2.carry($, tmp);
          Field.storeLimb(out, i, $);
        });
        if (!doReduce) return drop();
        // check if we underflowed by checking carry === 0 (in that case, we didn't and can return)
        i64x2.eq($, constI64x2(0n));
        if_(null, () => return_());
        // second loop
        // if we're here, y > x and out = x - y + R, while we want x - y + 2p
        // so do (out += 2p) and ignore the known overflow of R
        Field.forEach((i) => {
          // (carry, out[i]) = (2*p)[i] + out[i] + carry;
          Field.i64x2.p2(i);
          if (i > 0) f64x2.add();
          Field.loadLimb(out, i);
          f64x2.add();
          Field.i64x2.carry($, tmp);
          Field.storeLimb(out, i, $);
        });
        drop();
      }
    );

  const subtractI = subtractionI(true);
  const subtractINoReduce = subtractionI(false);

  return {
    i64x2: {
      reduce: reduceI,
      subtract: subtractI,
      subtractNoReduce: subtractINoReduce,
    },
    f64x2: {
      addNoCarry: additionFNoCarry,
    },
  };
}

/**
 * various helpers for finite field arithmetic:
 * isEqual, isZero, isGreater, copy
 */
function fieldHelpers(Field: FieldBase) {
  // x === y
  function isEqual(lane: 0 | 1) {
    return func({ in: [i32, i32], out: [i32] }, ([x, y]) => {
      Field.forEach((i) => {
        // if (x[i] !== y[i]) return false;
        Field.i64x2.loadLane(x, i, lane);
        Field.i64x2.loadLane(y, i, lane);
        i64.ne();
        if_(null, () => {
          i32.const(0);
          return_();
        });
      });
      i32.const(1);
    });
  }

  // x === 0
  function isZero(lane: 0 | 1) {
    return func({ in: [i32], out: [i32] }, ([x]) => {
      Field.forEach((i) => {
        // if (x[i] !== 0) return false;
        Field.i64x2.loadLane(x, i, lane);
        i64.ne($, 0n);
        if_(null, () => {
          i32.const(0);
          return_();
        });
      });
      i32.const(1);
    });
  }

  // x > y
  function isGreater(lane: 0 | 1) {
    return func(
      { in: [i32, i32], locals: [i64, i64], out: [i32] },
      ([x, y], [xi, yi]) => {
        block(null, () => {
          Field.forEachReversed((i) => {
            // if (x[i] > y[i]) return true;
            Field.i64x2.loadLane(x, i, lane);
            local.tee(xi);
            Field.i64x2.loadLane(y, i, lane);
            local.tee(yi);
            i64.gt_s();
            if_(null, () => {
              i32.const(1);
              return_();
            });
            // if (x[i] !== y[i]) break;
            i64.ne(xi, yi);
            br_if(0);
          });
        });
        // return false;
        i32.const(0);
      }
    );
  }

  // copy contents of y into x
  // this should just be inlined if possible
  function copyInline(x: Local<i32>, y: Local<i32>) {
    local.get(x);
    local.get(y);
    i32.const(Field.size);
    memory.copy();
  }
  const copy = func({ in: [i32, i32], out: [] }, ([x, y]) => {
    copyInline(x, y);
  });

  return { isEqual, isZero, isGreater, copy, copyInline };
}
