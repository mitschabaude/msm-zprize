import { importMemory, Module, type Func, type JSFunction } from "wasmati";
import { pallasParams } from "../concrete/pasta.params.js";
import { createField } from "../bigint/field.js";
import { memoryHelpers } from "../wasm/memory-helpers.js";
import { Tuple } from "../types.js";
import { float52ToInt64, numberToBigint64 } from "./fma-js.js";
import { multiplyWasm } from "./fma.js";

export { Field, Memory };

// constants
let mask51 = (1n << 51n) - 1n;
let c51 = 2 ** 52;
let c51n = numberToBigint64(c51);

// TODO constants that need to be parameters
let Fp = createField(pallasParams.modulus);
let p = Fp.modulus;

let wasmMemory = importMemory({ min: 1000, max: 1000, shared: true });
let multiplyModule = Module({
  memory: wasmMemory,
  exports: { multiply: multiplyWasm },
});
let { instance: mulInstance } = await multiplyModule.instantiate();
let sizeField = 8 * 5;
let sizeFieldPair = 2 * sizeField;

const Memory = memoryHelpers(p, 51, 5, { memory: wasmMemory.value });
const memory = Memory.memoryBytes;

const Field = {
  multiply: mulInstance.exports.multiply,
  size: sizeFieldPair,

  copy(x: number, y: number) {
    memory.copyWithin(x, y, y + sizeFieldPair);
  },

  writeBigintPair(x: number, x0: bigint, x1: bigint) {
    let view = new DataView(memory.buffer, x, sizeFieldPair);

    for (let offset = 0; offset < sizeFieldPair; offset += 16) {
      // we write each limb of x0 and x1 next to each other, as one v128
      view.setBigInt64(offset, (x0 & mask51) | c51n, true);
      let x0F = view.getFloat64(offset, true);
      view.setFloat64(offset, x0F - c51, true);

      view.setBigInt64(offset + 8, (x1 & mask51) | c51n, true);
      let x1F = view.getFloat64(offset + 8, true);
      view.setFloat64(offset + 8, x1F - c51, true);

      x0 >>= 51n;
      x1 >>= 51n;
    }
  },
  // writes only one half of a pair and leaves the other as is
  writeBigint(x: number, x0: bigint) {
    let view = new DataView(memory.buffer, x, sizeFieldPair);

    for (let offset = 0; offset < sizeFieldPair; offset += 16) {
      view.setBigInt64(offset, (x0 & mask51) | c51n, true);
      let x0F = view.getFloat64(offset, true);
      view.setFloat64(offset, x0F - c51, true);
      x0 >>= 51n;
    }
  },

  readBigintPair(x: number) {
    let view = new DataView(memory.buffer, x, sizeFieldPair);
    let x0 = 0n;
    let x1 = 0n;

    for (let offset = sizeFieldPair - 16; offset >= 0; offset -= 16) {
      let x0F = view.getFloat64(offset, true);
      x0 = (x0 << 51n) | float52ToInt64(x0F);

      let x1F = view.getFloat64(offset + 8, true);
      x1 = (x1 << 51n) | float52ToInt64(x1F);
    }
    return [x0, x1] satisfies Tuple;
  },
  readBigint(x: number) {
    let view = new DataView(memory.buffer, x, sizeFieldPair);
    let x0 = 0n;
    for (let offset = sizeFieldPair - 16; offset >= 0; offset -= 16) {
      let x0F = view.getFloat64(offset, true);
      x0 = (x0 << 51n) | float52ToInt64(x0F);
    }
    return x0;
  },
};
