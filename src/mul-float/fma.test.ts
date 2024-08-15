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
import { Field } from "./field.js";
import { writeWat } from "../wasm/wat-helpers.js";

const doWrite = true;

let p = pallasParams.modulus;
let R = (1n << 255n) % p;

const Fp = await Field.create(p);
let Local = Fp.Memory.local;

if (doWrite && Fp.moduleBytes) {
  await writeWat(
    import.meta.url.slice(7).replace(".ts", ".wat").replace(".js", ".wat"),
    Fp.moduleBytes
  );
}

// manual simple test
{
  let x = Local.getPointer(Fp.size);
  let y = Local.getPointer(Fp.size);
  let z = Local.getPointer(Fp.size);

  Fp.writePair(x, 1n, 1n);
  Fp.writePair(y, R, 1n);
  Fp.Wasm.multiply(z, x, y);

  let [z00, z01] = Fp.readPair(z);
  let z10 = montMulFmaWrapped(1n, R);
  let z11 = montMulFmaWrapped(1n, 1n);

  assertDeepEqual(z00, z10, "montmul wasm 1");
  assertDeepEqual(z01, z11, "montmul wasm 2");

  // can read/write individual fields from pair
  assertDeepEqual(Fp.read(z), z10, "read");
  assertDeepEqual(Fp.readSecond(z), z11, "read 2nd");

  Fp.write(z, 2n);
  Fp.writeSecond(x, 3n);
  assertDeepEqual(Fp.read(z), 2n, "write");
  assertDeepEqual(Fp.readSecond(x), 3n, "write");
}

// property tests

let eqivalentWasm = createEquivalentWasm(Fp.Memory, { logSuccess: true });
let fieldRng = Random.field(p);

let field = wasmSpec(Fp.Memory, fieldRng, {
  size: Fp.size,
  there: (xPtr, x) => Fp.write(xPtr, x),
  back: (x) => Fp.read(x),
});

let fieldPair = wasmSpec(Fp.Memory, Random.tuple([fieldRng, fieldRng]), {
  size: Fp.size,
  there: (xPtr, [x0, x1]) => Fp.writePair(xPtr, x0, x1),
  back: (x) => Fp.readPair(x),
});

eqivalentWasm(
  { from: [field], to: field },
  (x) => x,
  (out, x) => Fp.copy(out, x),
  "wasm roundtrip"
);

eqivalentWasm(
  { from: [fieldPair], to: fieldPair },
  (x) => x,
  (out, x) => Fp.copy(out, x),
  "wasm roundtrip pair"
);

// wasm version is exactly equivalent to montmulFma

eqivalentWasm(
  { from: [field, field], to: field },
  montMulFmaWrapped,
  Fp.Wasm.multiply,
  "mul fma"
);

eqivalentWasm(
  { from: [fieldPair, fieldPair], to: fieldPair },
  (x, y): [bigint, bigint] => [
    montMulFmaWrapped(x[0], y[0]),
    montMulFmaWrapped(x[1], y[1]),
  ],
  Fp.Wasm.multiply,
  "mul fma pairwise"
);

// when reducing, wasm version is exactly equivalent to montMulFmaWrapped
// followed by a conditional reduction that's slightly weaker than reducing to < p

function reduce(x: bigint) {
  return x >> 204n <= p >> 204n ? x : x - p;
}
function montmulReduce(x: bigint, y: bigint) {
  return reduce(montMulFmaWrapped(x, y));
}

eqivalentWasm(
  { from: [field, field], to: field },
  montmulReduce,
  Fp.Wasm.multiplyReduce,
  "mul reduce"
);

eqivalentWasm(
  { from: [fieldPair, fieldPair], to: fieldPair },
  (x, y): [bigint, bigint] => [
    montmulReduce(x[0], y[0]),
    montmulReduce(x[1], y[1]),
  ],
  Fp.Wasm.multiplyReduce,
  "mul reduce pairwise"
);

// it's still equivalent when we do 1000 squarings in a row

eqivalentWasm(
  { from: [field], to: field },
  (x) => {
    for (let i = 0; i < 1000; i++) {
      x = montmulReduce(x, x);
    }
    return x;
  },
  (z, x) => {
    for (let i = 0; i < 1000; i++) {
      Fp.Wasm.multiplyReduce(x, x, x);
    }
    Fp.copy(z, x);
  },
  "mul reduce 1000"
);
