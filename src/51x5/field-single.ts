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
  drop,
  call,
} from "wasmati";
import { FieldBase } from "./field-base.js";
import { mask51 } from "./common.js";
import { log64 } from "./debug.js";

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

  /**
   * Let pU = (p[4] + 1) << 204
   *
   * Note that p < pU < p + 2^204
   *
   * We can efficiently test for x < pU by checking if
   * x[4] <= p[4]
   *
   * `reduceInline` does:
   * x < pU ? x : x - p
   *
   * Controls size explosion after addition as follows:
   * [0, 2p) -> [0, p + 2^204)
   * [0, 2*(p + 2^204)) -> [0, p + 2^205)
   * [0, 2*(p + 2^205)) -> [0, p + 2^206)
   * ...
   *
   * As long as the number of consecutive additions (or "positive subtractions") with
   * subsequent reduction stays below a few dozen, a multiplication will always bring
   * both inputs back into [0, pU)
   */
  function reduceInline(x: Local<i32>, carry_?: Local<i64>) {
    block(null, ($return) => {
      // return if x4 <= p4
      // if not, x4 > p4 implies x > p
      i64.le_s(Field.loadLimb(x, 4), Field.P[4]);
      br_if($return);

      if (carry_ === undefined) {
        Field.forEach((i) => {
          Field.storeLimb(x, i, i64.sub(Field.loadLimb(x, i), Field.P[i]));
        });
        return;
      }
      Field.forEach((i) => {
        Field.loadLimb(x, i);
        if (i > 0) i64.add(); // add the carry
        i64.sub($, Field.P[i]);
        if (i < 4) carry($, carry_);
        Field.storeLimb(x, i, $);
      });
    });
  }

  const reduce = func({ in: [i32], locals: [i64], out: [] }, ([x], [carry]) => {
    reduceInline(x, carry);
  });

  const addRaw = func({ in: [i32, i32, i32], out: [] }, ([z, x, y]) => {
    for (let i = 0; i < 5; i++) {
      Field.loadLimb(x, i);
      Field.loadLimb(y, i);
      Field.storeLimb(z, i, i64.add());
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
        if (i < 4) carry($, tmp);
        Field.storeLimb(z, i, $);
      }
    }
  );
  const add = func(
    { in: [i32, i32, i32], locals: [i64], out: [] },
    ([z, x, y], [tmp]) => {
      for (let i = 0; i < 5; i++) {
        Field.loadLimb(x, i);
        Field.loadLimb(y, i);
        Field.storeLimb(z, i, i64.add());
      }
      reduceInline(z);
      // carry result
      for (let i = 0; i < 5; i++) {
        Field.loadLimb(z, i);
        if (i > 0) i64.add(); // add carry
        if (i < 4) carry($, tmp);
        Field.storeLimb(z, i, $);
      }
    }
  );

  const subRaw = func({ in: [i32, i32, i32], out: [] }, ([z, x, y]) => {
    for (let i = 0; i < 5; i++) {
      Field.loadLimb(x, i);
      Field.loadLimb(y, i);
      i64.sub();
      Field.storeLimb(z, i, $);
    }
  });
  const subCarry = func(
    { in: [i32, i32, i32], locals: [i64], out: [] },
    ([z, x, y], [tmp]) => {
      for (let i = 0; i < 5; i++) {
        // (carry, out[i]) = x[i] - y[i] + carry;
        Field.loadLimb(x, i);
        Field.loadLimb(y, i);
        i64.sub();
        if (i > 0) i64.add(); // add carry
        if (i < 4) carry($, tmp);
        Field.storeLimb(z, i, $);
      }
    }
  );
  /**
   * Subtracting x, y in [0, pU) means the result is in [-pU, pU)
   * We conditionally add p, which brings the result to [p-pU, pU)
   * Note that it can still be slightly negative -- in which case we do the same again
   */
  const sub = func(
    { in: [i32, i32, i32], locals: [i64], out: [] },
    ([z, x, y], [tmp]) => {
      for (let i = 0; i < 5; i++) {
        // (carry, z[i]) = x[i] - y[i] + carry;
        Field.loadLimb(x, i);
        Field.loadLimb(y, i);
        i64.sub();
        if (i > 0) i64.add(); // add carry
        carry($, tmp); // we leave carry even in the last iteration
        Field.storeLimb(z, i, $);
      }
      // if we underflowed, carry = -1, otherwise carry = 0
      i64.eqz();
      br_if(0); // conditional return
      // + p
      Field.forEach((i) => {
        Field.loadLimb(z, i);
        if (i > 0) i64.add(); // add the carry
        i64.add($, Field.P[i]);
        carry($, tmp);
        Field.storeLimb(z, i, $);
      });
      // since the carry was negative before, we really computed z = x - y + 2^255
      // now with the addition of p, we have x - y + p + 2^255
      // the carry of that is 0 only if x - y + p < 0
      // in that case we add p again, and ignoring the carry is just what gives us x - y + 2p
      i64.ne($, 0n);
      br_if(0);
      Field.forEach((i) => {
        Field.loadLimb(z, i);
        if (i > 0) i64.add(); // add the carry
        i64.add($, Field.P[i]);
        if (i < 4) carry($, tmp);
        else i64.and($, mask51);
        Field.storeLimb(z, i, $);
      });
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
    add,
    addRaw,
    addCarry,
    sub,
    subRaw,
    subCarry,
    reduce,
    fullyReduce,
    isEqual,
    isZero,
    isGreater,
    copy,
  };
}
