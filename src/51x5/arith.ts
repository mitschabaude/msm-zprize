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
  i64x2,
  f64x2,
  Global,
  type Func,
  type Input,
  i32x4,
} from "wasmati";
import { Constants, FieldPair, constI64x2 } from "./field-base.js";
import { mask51 } from "./common.js";

export { arithmetic, fieldHelpers, carryLocals, carryLocalsSingle };

function carryLocals(Z: Local<v128>[]) {
  local.set(Z[1], i64x2.add(Z[1], i64x2.shr_s(Z[0], 51)));
  local.set(Z[2], i64x2.add(Z[2], i64x2.shr_s(Z[1], 51)));
  local.set(Z[3], i64x2.add(Z[3], i64x2.shr_s(Z[2], 51)));
  local.set(Z[4], i64x2.add(Z[4], i64x2.shr_s(Z[3], 51)));

  local.set(Z[0], v128.and(Z[0], constI64x2(mask51)));
  local.set(Z[1], v128.and(Z[1], constI64x2(mask51)));
  local.set(Z[2], v128.and(Z[2], constI64x2(mask51)));
  local.set(Z[3], v128.and(Z[3], constI64x2(mask51)));
}

function carryLocalsSingle(Z: Local<i64>[]) {
  local.set(Z[1], i64.add(Z[1], i64.shr_s(Z[0], 51n)));
  local.set(Z[2], i64.add(Z[2], i64.shr_s(Z[1], 51n)));
  local.set(Z[3], i64.add(Z[3], i64.shr_s(Z[2], 51n)));
  local.set(Z[4], i64.add(Z[4], i64.shr_s(Z[3], 51n)));

  local.set(Z[0], i64.and(Z[0], mask51));
  local.set(Z[1], i64.and(Z[1], mask51));
  local.set(Z[2], i64.and(Z[2], mask51));
  local.set(Z[3], i64.and(Z[3], mask51));
}

function arithmetic(p: bigint, pSelectPtr: Global<i32>) {
  let constants = Constants(p);
  let PI = constants.i64.P;

  /**
   * Reduce lane in i64 arithmetic, assuming all limbs are positive
   *
   * This only reduces inputs < 2p to < (p4 + 1) * 2^(4 * 51), not < p,
   * but that's enough for multiplication to map back to < 2p
   *
   * Also, this does not perform a carry!
   */
  function reduceLaneLocals(lane: 0 | 1, X: Local<v128>[]) {
    block(null, ($outer) => {
      // return if x4 <= p4
      // if not, x4 > p4 implies x > p
      i64x2.extract_lane(lane, X[4]);
      i64.le_s($, PI[4]);
      br_if($outer);

      // if we're here, x > p but, by assumption, x < 2p, so do x - p
      FieldPair.forEach((i) => {
        v128.const("i64x2", lane === 0 ? [PI[i], 0n] : [0n, PI[i]]);
        local.set(X[i], i64x2.sub(X[i], $));
      });
    });
  }

  /**
   * Like `reduceLaneLocals`, but reduces both lanes at the same time, by
   * selecting a pointer that points to the right combination of p and 0 to subtract
   *
   * Not sure if this is more efficient, because we add memory reads
   */
  function reduceLocals(X: Local<v128>[], tmp: Local<v128>, pOr0: Local<i32>) {
    // subtract 0 if x4 <= p4, and p if x4 > p4
    local.tee(tmp, i64x2.le_s(X[4], constI64x2(PI[4])));
    i32x4.extract_lane(0);
    i32.mul($, 2);
    i32x4.extract_lane(2, tmp);
    // now there are two int32s on the stack which are either -1 (if x4 <= p4) or 0
    // use them to compute:
    // (x=-1, y=-1) -> 0, (-1, 0) -> 1, (0, -1) -> 2, (0, 0) -> 3
    // solution: 2(x + 1) + (y + 1) = 2x + y + 3
    i32.add();
    i32.add($, 3);
    i32.load8_u({}, i32.add($, pSelectPtr));
    local.set(pOr0);

    // if we're here, x > p but, by assumption, x < 2p, so do x - p
    FieldPair.forEach((i) => {
      FieldPair.loadLimb(pOr0, i);
      local.set(X[i], i64x2.sub(X[i], $));
    });
  }

  function reduceLocalsSingle(X: Local<i64>[]) {
    block(null, ($outer) => {
      // return if x4 <= p4
      // if not, x4 > p4 implies x > p
      i64.le_s(X[4], PI[4]);
      br_if($outer);

      // if we're here, x > p but, by assumption, x < 2p, so do x - p
      FieldPair.forEach((i) => {
        local.set(X[i], i64.sub(X[i], PI[i]));
      });
    });
  }

  function fullyReduceLane(lane: 0 | 1, x: Local<i32>, xi: Local<i64>) {
    block(null, ($outer) => {
      // check if x < p
      block(null, ($inner) => {
        FieldPair.forEachReversed((i) => {
          // if (x[i] < p[i]) return
          local.tee(xi, FieldPair.i64.loadLane(x, i, lane));
          i64.lt_u($, PI[i]);
          br_if($outer);
          // if (x[i] !== p[i]) break;
          i64.ne(xi, PI[i]);
          br_if($inner);
        });
      });

      // if we're here, x >= p but we assume x < 2p, so do x - p
      FieldPair.forEach((i) => {
        // (carry, x[i]) = x[i] - p[i] + carry;
        FieldPair.i64.loadLane(x, i, lane);
        if (i > 0) i64.add(); // add the carry
        local.tee(xi, i64.sub($, PI[i]));
        i64.shr_s($, 51n); // carry, left on the stack
        FieldPair.i64.storeLane(x, i, lane, i64.and(xi, mask51));
      });
      drop();
    });
  }

  /**
   * Reduce lane in i64 arithmetic, assuming all limbs are positive
   */
  function fullyReduceLaneLocals(
    lane: 0 | 1,
    X: Local<v128>[],
    xi: Local<i64>,
    tmp: Local<v128>
  ) {
    block(null, ($outer) => {
      // check if x < p
      block(null, ($inner) => {
        FieldPair.forEachReversed((i) => {
          // if (x[i] < p[i]) return
          local.get(X[i]);
          i64x2.extract_lane(lane);
          local.tee(xi);
          i64.lt_u($, PI[i]);
          br_if($outer);
          // if (x[i] !== p[i]) break;
          i64.ne(xi, PI[i]);
          br_if($inner);
        });
      });

      // if we're here, x >= p but we assume x < 2p, so do x - p
      local.set(tmp, constI64x2(0n));
      FieldPair.forEach((i) => {
        // (carry, x[i]) = x[i] - p[i] + carry;
        local.get(X[i]);
        if (i > 0) i64x2.add(); // add the carry
        v128.const("i64x2", lane === 0 ? [PI[i], 0n] : [0n, PI[i]]);
        i64x2.sub();
        if (i < 4) {
          local.tee(tmp);
          i64x2.shr_s($, 51); // carry, left on the stack
          local.set(X[i], v128.and(tmp, constI64x2(mask51)));
        } else {
          local.set(X[i], v128.and($, constI64x2(mask51)));
        }
      });
    });
  }

  /**
   * reduce in place from < 2*p to < p, i.e.
   * if (x > p) x -= p
   */
  const fullyReduce = func(
    { in: [i32], locals: [i64], out: [] },
    ([x], [xi]) => {
      fullyReduceLane(0, x, xi);
      fullyReduceLane(1, x, xi);
    }
  );

  return {
    fullyReduce,
    fullyReduceLane,
    fullyReduceLaneLocals,
    reduceLaneLocals,
    reduceLocals,
    reduceLocalsSingle,
  };
}

/**
 * various helpers for finite field arithmetic:
 * isEqual, isZero, isGreater, copy
 */
function fieldHelpers(p: bigint) {
  // x === y
  function isEqual(lane: 0 | 1) {
    return func({ in: [i32, i32], out: [i32] }, ([x, y]) => {
      FieldPair.forEach((i) => {
        // if (x[i] !== y[i]) return false;
        FieldPair.i64.loadLane(x, i, lane);
        FieldPair.i64.loadLane(y, i, lane);
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
      FieldPair.forEach((i) => {
        // if (x[i] !== 0) return false;
        FieldPair.i64.loadLane(x, i, lane);
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
          FieldPair.forEachReversed((i) => {
            // if (x[i] > y[i]) return true;
            FieldPair.i64.loadLane(x, i, lane);
            local.tee(xi);
            FieldPair.i64.loadLane(y, i, lane);
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
    i32.const(FieldPair.size);
    memory.copy();
  }
  const copy = func({ in: [i32, i32], out: [] }, ([x, y]) => {
    copyInline(x, y);
  });

  return { isEqual, isZero, isGreater, copy, copyInline };
}
