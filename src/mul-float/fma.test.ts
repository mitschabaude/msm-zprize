/**
 * Wasm code inspired by Niall Emmart's work on using FMA instructions for bigint multiplication.
 *
 * Paper:
 *   "Faster Modular Exponentiation Using Double Precision Floating Point Arithmetic on the GPU."
 *   2018 IEEE 25th Symposium on Computer Arithmetic (ARITH). IEEE Computer Society, 2018.
 *   By Emmart, Zheng and Weems.
 *
 * Reference code:
 * https://github.com/yrrid/submission-wasm-twisted-edwards (see FP51.java and FieldPair.c)
 */
import { assertDeepEqual } from "../testing/nested.js";
import { pallasParams } from "../concrete/pasta.params.js";
import { Random } from "../testing/random.js";
import { createEquivalentWasm, wasmSpec } from "../testing/equivalent-wasm.js";
import { montMulFmaWrapped } from "./fma-js.js";
import { Field, Memory } from "./field.js";

let p = pallasParams.modulus;
let R = (1n << 255n) % p;

// manual simple test
{
  let x = Memory.local.getPointer(Field.size);
  let y = Memory.local.getPointer(Field.size);
  let z = Memory.local.getPointer(Field.size);

  Field.writeBigintPair(x, 1n, 1n);
  Field.writeBigintPair(y, R, 1n);
  Field.multiply(z, x, y);

  let [z00, z01] = Field.readBigintPair(z);
  let z10 = montMulFmaWrapped(1n, R);
  let z11 = montMulFmaWrapped(1n, 1n);

  assertDeepEqual(z00, z10, "montmul wasm 1");
  assertDeepEqual(z01, z11, "montmul wasm 2");
}

// property tests

let eqivalentWasm = createEquivalentWasm(Memory, { logSuccess: true });
let fieldRng = Random.field(p);

let field = wasmSpec(Memory, fieldRng, {
  size: Field.size,
  there: Field.writeBigint,
  back: Field.readBigint,
});

let fieldPair = wasmSpec(Memory, Random.tuple([fieldRng, fieldRng]), {
  size: Field.size,
  there: (xPtr, [x0, x1]) => Field.writeBigintPair(xPtr, x0, x1),
  back: Field.readBigintPair,
});

eqivalentWasm(
  { from: [field], to: field },
  (x) => x,
  Field.copy,
  "wasm roundtrip"
);

eqivalentWasm(
  { from: [fieldPair], to: fieldPair },
  (x) => x,
  Field.copy,
  "wasm roundtrip pair"
);

// wasm version is exactly equivalent to montmulFma

eqivalentWasm(
  { from: [field, field], to: field },
  montMulFmaWrapped,
  Field.multiply,
  "montmul fma (wasm)"
);

eqivalentWasm(
  { from: [fieldPair, fieldPair], to: fieldPair },
  (x, y): [bigint, bigint] => [
    montMulFmaWrapped(x[0], y[0]),
    montMulFmaWrapped(x[1], y[1]),
  ],
  Field.multiply,
  "montmul fma pairwise"
);
