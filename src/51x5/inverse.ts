import {
  $,
  block,
  br,
  br_if,
  call,
  func,
  i32,
  i64,
  if_,
  Local,
  local,
  loop,
  memory,
  return_,
  unreachable,
  Func,
} from "wasmati";
import { mod } from "../bigint/field-util.js";
import { ImplicitMemory } from "../wasm/wasm-util.js";
import { log2 } from "../util.js";
import { bigintToData, createField } from "./field-base.js";
import { mask51 } from "./common.js";
import { fieldWithMethods } from "./field-single.js";
import { multiplySingle } from "./fma.js";

// TODO: untested, WIP

export { Inverse };

type Inverse = {
  makeOdd: Func<[i32, i32], [i32]>;
  inverse: Func<[i32, i32, i32], []>;
};

function Inverse(p: bigint, implicitMemory: ImplicitMemory): Inverse {
  const Field = fieldWithMethods(createField(p, "single"));
  const n = 5;
  const multiply = multiplySingle(p, "single");

  /**
   * a core building block for montgomery inversion
   *
   * takes u, s < p. sets k=0. while u is even, update u /= 2 and s *= 2 and increment k++
   * at the end, u <- u/2^k, s <- s*2^k and the new u is odd
   * returns k
   * (the implementation shifts u >> k and s << k at once if k < w, and shifts by whole words until k < w)
   *
   * in the inversion algorithm it's guaranteed that s << k will remain < p,
   * so everything holds modulo p
   */
  const makeOdd = func(
    { in: [i32, i32], locals: [i64, i32, i64, i64], out: [i32] },
    ([u, s], [k, k0, l, tmp]) => {
      // k = count_trailing_zeros(u[0])
      let ui = Field.loadLimb(u, 0);
      local.tee(k, i64.ctz(ui));
      i64.eqz();
      // if (k === 0) return; (the most common case)
      if_(null, () => {
        i32.const(0);
        return_();
      });
      // while k === 64 (i.e., u[0] === 0), shift by whole words
      // (note: u is not supposed to be 0, so u[0] = 0 implies that u is divisible by 2^w)
      block(null, (block) => {
        loop(null, (loop) => {
          i64.ne(k, 64n);
          br_if(block);

          // copy u[1],...,u[n-1] --> u[0],...,u[n-2]
          local.get(u);
          i32.add(u, 4);
          i32.const((n - 1) * 4);
          memory.copy();

          // u[n-1] = 0
          Field.storeLimb(u, n - 1, 0n);

          // copy s[0],...,s[n-2] --> s[1],...,s[n-1]
          i32.add(s, 4);
          local.get(s);
          i32.const((n - 1) * 4);
          memory.copy();

          // s[0] = 0
          Field.storeLimb(s, 0, 0n);

          local.set(k0, i32.add(k0, 51));
          local.set(k, i64.ctz(Field.loadLimb(u, 0)));
          br(loop);
        });
      });

      // here we know that k \in 0,...,w-1
      // l = w - k
      local.set(l, i64.sub(51n, k));

      // u >> k

      // for (let i = 0; i < n-1; i++) {
      //   u[i] = (u[i] >> k) | ((u[i + 1] << l) & wordMax);
      // }
      // u[n-1] = u[n-1] >> k;
      local.set(tmp, Field.loadLimb(u, 0));
      Field.forEach((i) => {
        i64.shr_u(tmp, k);
        if (i < n - 1) {
          local.tee(tmp, Field.loadLimb(u, i + 1));
          i64.shl($, l);
          i64.and($, mask51);
          i64.or();
        }
        Field.storeLimb(u, i, $);
      });

      // s << k

      // for (let i = n-1; i >= 0; i--) {
      //   s[i+1] = (s[i] >> l) | ((s[i+1] << k) & wordMax);
      // }
      // s[0] = (s[0] << k) & wordMax;
      local.set(tmp, Field.loadLimb(s, n - 1));
      for (let i = n - 2; i >= 0; i--) {
        i64.shl(tmp, k);
        i64.and($, mask51);
        local.tee(tmp, Field.loadLimb(s, i));
        i64.shr_u($, l);
        i64.or();
        Field.storeLimb(s, i + 1, $);
      }
      i64.shl(tmp, k);
      i64.and($, mask51);
      Field.storeLimb(s, 0, $);

      // return k
      i32.add(k0, i32.wrap_i64(k));
    }
  );

  // kaliski "almost inverse" algorithm
  // this is modified from the algorithms in papers in that it
  // * returns k-1 instead of k
  // * returns r < p unconditionally
  // * allows to batch left- / right-shifts
  const almostInverse = func(
    { in: [i32, i32, i32], locals: [i32, i32, i32], out: [i32] },
    ([u, r, a], [v, s, k]) => {
      // setup locals
      local.set(v, i32.add(u, Field.size));
      local.set(s, i32.add(v, Field.size));

      // u = p, v = a, r = 0, s = 1
      Field.store(u, Field.P);
      Field.copyInline(v, a);
      Field.store(r, Field.Zero);
      Field.store(s, Field.One);

      // main algorithm
      call(makeOdd, [u, s]);
      call(makeOdd, [v, r]);
      local.set(k, i32.add());

      block(null, (block) => {
        loop(null, (loop) => {
          call(Field.isGreater, [u, v]);
          if_(
            null,
            () => {
              call(Field.subCarry, [u, u, v]);
              call(Field.addCarry, [r, r, s]);
              call(makeOdd, [u, s]);
              local.set(k, i32.add($, k));
            },
            () => {
              call(Field.subCarry, [v, v, u]);
              call(Field.addCarry, [s, s, r]);
              call(Field.isZero, [v]);
              br_if(block);
              call(makeOdd, [v, r]);
              local.set(k, i32.add($, k));
            }
          );
          br(loop);
        });
      });
      local.get(k);
    }
  );

  // constants we store as global pointers
  let K = n * 51;
  let N = log2(p);
  let R2corr = mod(1n << BigInt(4 * K - 2 * N + 1), p);
  let r2corrGlobal = implicitMemory.data(bigintToData(R2corr));
  let pGlobal = implicitMemory.data(bigintToData(p));

  /**
   * montgomery multiplication r 2^l 2^(-K)
   */
  function mulPow2(scratch: Local<i32>, r: Local<i32>, l: Local<i32>) {
    // store 2^l in scratch
    Field.setZero(scratch);
    i64.store(
      {},
      // limb index
      i32.add(scratch, i32.div_u(l, 51)),
      // bit vector
      i64.shl(1n, i64.extend_i32_u(i32.rem_u(l, 51)))
    );
    // multiply r by 2^l
    call(multiply, [r, r, scratch]);
  }

  /**
   * montgomery inverse, a 2^K -> a^(-1) 2^K (mod p)
   */
  const inverse = func(
    { in: [i32, i32, i32], locals: [i32], out: [] },
    ([scratch, r, a], [k]) => {
      call(Field.fullyReduce, [a]);

      // error if input is zero
      call(Field.isZero, [a]);
      if_(null, () => unreachable());

      call(almostInverse, [scratch, r, a]);
      local.set(k);
      // don't have to reduce r here, because it's already < p
      call(Field.subCarry, [r, pGlobal, r]);
      // multiply by 2^(2N - k), where N = bit length of p
      // we use k+1 here because that's the value the theory is about:
      // N <= k+1 <= 2N, so that 0 <= 2N-(k+1) <= N, so that
      // 1 <= 2^(2N-(k+1)) <= 2^N < 2p
      // (in practice, k seems to be normally distributed around ~1.4N and never reach either N or 2N)
      local.set(k, i32.sub(2 * N - 1, k));
      mulPow2(scratch, r, k); // * 2^(2N - (k+1)) * 2^(-K)
      call(multiply, [r, r, r2corrGlobal]); // * 2^(4K - 2N + 1) * 2^(-K)
      // = * 2^(2K - k)
      // ^^^ transforms (a 2^K)^(-1) 2^k = a^(-1) 2^(-K+k)
      //     to a^(-1) 2^(-K+k + 2K -k) = a^(-1) 2^K = the montgomery representation of a^(-1)
    }
  );

  return { makeOdd, inverse };
}
