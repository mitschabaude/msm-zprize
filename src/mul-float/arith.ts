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
  type Func,
  type Input,
  br,
} from "wasmati";
import {
  Field,
  I64x2,
  constI64x2,
  f64x2Constants,
  i64x2Constants,
} from "./field-base.js";
import { mask51 } from "./common.js";

export { arithmetic, fieldHelpers, FieldWithArithmetic };

type FieldWithArithmetic = ReturnType<typeof FieldWithArithmetic>;

function FieldWithArithmetic(p: bigint) {
  const arithmetic_ = arithmetic(p);
  const helper_ = fieldHelpers(p);
  return { ...arithmetic_, ...helper_ };
}

// TODO most of this doesn't work

function arithmetic(p: bigint) {
  let constants = {
    i64x2: i64x2Constants(p),
    f64x2: f64x2Constants(p),
  };
  let PI = constants.i64x2.P;
  let PF = constants.f64x2.P;

  function reduceLaneI(lane: 0 | 1, x: Local<i32>, xi: Local<i64>) {
    block(null, ($outer) => {
      // check if x < p
      block(null, ($inner) => {
        Field.forEachReversed((i) => {
          // if (x[i] < p[i]) return
          local.tee(xi, I64x2.loadLane(x, i, lane));
          i64.lt_u($, PI[i]);
          br_if($outer);
          // if (x[i] !== p[i]) break;
          i64.ne(xi, PI[i]);
          br_if($inner);
        });
      });

      // if we're here, x >= p but we assume x < 2p, so do x - p
      Field.forEach((i) => {
        // (carry, x[i]) = x[i] - p[i] + carry;
        I64x2.loadLane(x, i, lane);
        if (i > 0) i64.add(); // add the carry
        local.tee(xi, i64.sub($, PI[i]));
        i64.shr_s($, 51n); // carry, left on the stack
        I64x2.storeLane(x, i, lane, i64.and(xi, mask51));
      });
      drop();
    });
  }

  /**
   * Reduce lane in i64 arithmetic, assuming all limbs are positive
   */
  function reduceLaneLocals(
    lane: 0 | 1,
    X: Local<v128>[],
    xi: Local<i64>,
    tmp: Local<v128>
  ) {
    block(null, ($outer) => {
      // check if x < p
      block(null, ($inner) => {
        Field.forEachReversed((i) => {
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
      Field.forEach((i) => {
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
  const reduceI = func({ in: [i32], locals: [i64], out: [] }, ([x], [xi]) => {
    reduceLaneI(0, x, xi);
    reduceLaneI(1, x, xi);
  });

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

  return {
    i64x2: {
      reduce: reduceI,
      reduceLane: reduceLaneI,
      reduceLaneLocals: reduceLaneLocals,
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
function fieldHelpers(p: bigint) {
  // x === y
  function isEqual(lane: 0 | 1) {
    return func({ in: [i32, i32], out: [i32] }, ([x, y]) => {
      Field.forEach((i) => {
        // if (x[i] !== y[i]) return false;
        I64x2.loadLane(x, i, lane);
        I64x2.loadLane(y, i, lane);
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
        I64x2.loadLane(x, i, lane);
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
            I64x2.loadLane(x, i, lane);
            local.tee(xi);
            I64x2.loadLane(y, i, lane);
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
