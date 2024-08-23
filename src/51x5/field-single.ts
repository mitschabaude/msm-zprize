import {
  $,
  block,
  br_if,
  func,
  i32,
  i64,
  if_,
  local,
  return_,
  Local,
  type Func,
  type Input,
  StackVar,
} from "wasmati";
import { FieldBase } from "./field-base.js";
import { mask51 } from "./common.js";

export { fieldWithMethods, fieldMethods };

function fieldWithMethods(Field: FieldBase) {
  let methods = fieldMethods(Field);
  return { ...Field, ...methods };
}

/**
 * various helpers for finite field arithmetic:
 * isEqual, isZero, isGreater, copy
 */
function fieldMethods(Field: FieldBase) {
  // x === y
  const isEqual = func({ in: [i32, i32], out: [i32] }, ([x, y]) => {
    Field.forEach((i) => {
      // if (x[i] !== y[i]) return false;
      Field.loadLimb(x, i);
      Field.loadLimb(y, i);
      i64.ne();
      if_(null, () => {
        i32.const(0);
        return_();
      });
    });
    i32.const(1);
  });

  // x === 0
  const isZero = func({ in: [i32], out: [i32] }, ([x]) => {
    Field.forEach((i) => {
      // if (x[i] !== 0) return false;
      Field.loadLimb(x, i);
      i64.ne($, 0n);
      if_(null, () => {
        i32.const(0);
        return_();
      });
    });
    i32.const(1);
  });

  // x > y
  const isGreater = func(
    { in: [i32, i32], locals: [i64, i64], out: [i32] },
    ([x, y], [xi, yi]) => {
      block(null, () => {
        Field.forEachReversed((i) => {
          // if (x[i] > y[i]) return true;
          Field.loadLimb(x, i);
          local.tee(xi);
          Field.loadLimb(y, i);
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

  function carry(input: StackVar<i64>, tmp: Local<i64>) {
    // put carry on the stack
    local.tee(tmp, input);
    i64.shr_s($, 51n);
    // mod 2^51 the current result
    i64.and(tmp, mask51);
  }

  const addRaw = func({ in: [i32, i32, i32], out: [] }, ([z, x, y]) => {
    for (let i = 0; i < 5; i++) {
      Field.loadLimb(x, i);
      Field.loadLimb(y, i);
      i64.add();
      Field.storeLimb(z, i, $);
    }
  });
  const addCarry = func(
    { in: [i32, i32, i32], locals: [i64], out: [] },
    ([z, x, y], [tmp]) => {
      for (let i = 0; i < 5; i++) {
        // (carry, z[i]) = x[i] + y[i] + carry;
        let xi = Field.loadLimb(x, i);
        let yi = Field.loadLimb(y, i);
        i64.add(xi, yi);
        if (i > 0) i64.add(); // add carry
        if (i < 4) carry($, tmp); // push carry on the stack
        Field.storeLimb(z, i, $);
      }
    }
  );

  const subRaw = func({ in: [i32, i32, i32], out: [] }, ([out, x, y]) => {
    for (let i = 0; i < 5; i++) {
      Field.loadLimb(x, i);
      Field.loadLimb(y, i);
      i64.sub();
      Field.storeLimb(out, i, $);
    }
  });
  const subCarry = func(
    { in: [i32, i32, i32], locals: [i64], out: [] },
    ([out, x, y], [tmp]) => {
      for (let i = 0; i < 5; i++) {
        // (carry, out[i]) = x[i] - y[i] + carry;
        Field.loadLimb(x, i);
        Field.loadLimb(y, i);
        i64.sub();
        if (i > 0) i64.add(); // add carry
        if (i < 4) carry($, tmp);
        Field.storeLimb(out, i, $);
      }
    }
  );

  /**
   * if (x >= p) x -= p
   */
  const fullyReduce = func(
    { in: [i32], locals: [i64], out: [] },
    ([x], [tmp]) => {
      // check if x < p
      block(null, () => {
        Field.forEachReversed((i) => {
          // if (x[i] < p[i]) return
          Field.loadLimb(x, i);
          local.tee(tmp);
          i64.lt_u($, Field.P[i]);
          br_if(1);
          // if (x[i] !== p[i]) break;
          i64.ne(tmp, Field.P[i]);
          br_if(0);
        });
      });
      // if we're here, t >= p but we assume t < 2p, so do t - p
      Field.forEach((i) => {
        // (carry, x[i]) = x[i] - p[i] + carry;
        Field.loadLimb(x, i);
        if (i > 0) i64.add(); // add the carry
        i64.sub($, Field.P[i]);
        if (i < 4) carry($, tmp);
        Field.storeLimb(x, i, $);
      });
    }
  );

  let copy = func({ in: [i32, i32], out: [] }, ([x, y]) => {
    Field.copyInline(x, y);
  });

  return {
    addRaw,
    addCarry,
    subRaw,
    subCarry,
    fullyReduce,
    isEqual,
    isZero,
    isGreater,
    copy,
  };
}
