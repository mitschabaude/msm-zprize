import { pallasParams } from "../concrete/pasta.params.js";
import { Random } from "../testing/random.js";
import { createEquivalentWasm, wasmSpec } from "../testing/equivalent-wasm.js";
import { Field } from "./field.js";
import { createField as createFieldBigint } from "../bigint/field.js";

let p = pallasParams.modulus;
let R = (1n << 255n) % p;

const Fp = await Field.create(p);
const FpWasm = Fp.Wasm;
const FpBigint = createFieldBigint(p);
let Local = Fp.Memory.local;

// partial reduce
function reduce(x: bigint) {
  return x >> 204n <= p >> 204n ? x : x - p;
}

// property tests

let equiv = createEquivalentWasm(Fp.Memory, { logSuccess: true });
let fieldRng = Random.field(p);
let fieldWeaklyReducedRng = Random.map(Random.bignat(1n << 204n), (u) => p + u);

let field = wasmSpec(Fp.Memory, fieldRng, {
  size: Fp.size,
  there: (xPtr, x) => Fp.writeSingle(xPtr, x),
  back: (x) => Fp.readSingle(x),
});
let fieldWeaklyReduced = wasmSpec(Fp.Memory, fieldWeaklyReducedRng, {
  size: Fp.size,
  there: (xPtr, x) => Fp.writeSingle(xPtr, x),
  back: (x) => Fp.readSingle(x),
});

equiv(
  { from: [field], to: field },
  (x) => x,
  (out, x) => Fp.copy(out, x),
  "wasm roundtrip"
);

equiv(
  { from: [field, field], to: field },
  (x, y) => reduce(x + y),
  FpWasm.add,
  "add"
);

equiv(
  { from: [field, field], to: field },
  FpBigint.subtract,
  FpWasm.sub,
  "sub"
);

equiv(
  { from: [field, fieldWeaklyReduced], to: field },
  (x, y) => FpBigint.mod(x - y),
  FpWasm.sub,
  "sub: x - y + p < 0"
);
