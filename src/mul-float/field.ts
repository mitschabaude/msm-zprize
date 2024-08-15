import { importMemory, Module } from "wasmati";
import { MemoryHelpers, memoryHelpers } from "../wasm/memory-helpers.js";
import { Tuple } from "../types.js";
import { float52ToInt64, numberToBigint64 } from "./fma-js.js";
import { Multiply } from "./fma.js";
import { assert } from "../util.js";

export { createFieldWasm, Field };

// constants
let mask51 = (1n << 51n) - 1n;
let c51 = 2 ** 52;
let c51n = numberToBigint64(c51);

let sizeField = 8 * 5;
let sizeFieldPair = 2 * sizeField;

function validateAssumptions(modulus: bigint) {
  assert(modulus < 1n << 255n);
}

async function createFieldWasm(p: bigint) {
  let wasmMemory = importMemory({ min: 1000, max: 1000, shared: true });
  let multiplyModule = Module({
    memory: wasmMemory,
    exports: { multiply: Multiply(p).multiply },
  });
  let { instance } = await multiplyModule.instantiate();

  const Memory = memoryHelpers(p, 51, 5, { memory: wasmMemory.value });
  const memory = Memory.memoryBytes;

  return new Field(p, memory, instance.exports, Memory);
}

type WasmIntf = {
  multiply: (z: number, x: number, y: number) => void;
};

class Field {
  size = sizeFieldPair;
  static size = sizeFieldPair;

  constructor(
    public modulus: bigint,
    public memory: Uint8Array,
    public Wasm: WasmIntf,
    public Memory: MemoryHelpers
  ) {
    validateAssumptions(modulus);
  }

  static create(p: bigint) {
    return createFieldWasm(p);
  }

  copy(x: number, y: number) {
    this.memory.copyWithin(x, y, y + sizeFieldPair);
  }

  writePair(x: number, x0: bigint, x1: bigint) {
    let view = new DataView(this.memory.buffer, x, sizeFieldPair);

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
  }

  // writes only one half of a pair and leaves the other as is
  write(x: number, x0: bigint) {
    let view = new DataView(this.memory.buffer, x, sizeFieldPair);

    for (let offset = 0; offset < sizeFieldPair; offset += 16) {
      view.setBigInt64(offset, (x0 & mask51) | c51n, true);
      let x0F = view.getFloat64(offset, true);
      view.setFloat64(offset, x0F - c51, true);
      x0 >>= 51n;
    }
  }

  writeSecond(x: number, x1: bigint) {
    this.write(x + 8, x1);
  }

  readPair(x: number) {
    let view = new DataView(this.memory.buffer, x, sizeFieldPair);
    let x0 = 0n;
    let x1 = 0n;

    for (let offset = sizeFieldPair - 16; offset >= 0; offset -= 16) {
      let x0F = view.getFloat64(offset, true);
      x0 = (x0 << 51n) | float52ToInt64(x0F);

      let x1F = view.getFloat64(offset + 8, true);
      x1 = (x1 << 51n) | float52ToInt64(x1F);
    }
    return [x0, x1] satisfies Tuple;
  }

  read(x: number) {
    let view = new DataView(this.memory.buffer, x, sizeFieldPair);
    let x0 = 0n;
    for (let offset = sizeFieldPair - 16; offset >= 0; offset -= 16) {
      let x0F = view.getFloat64(offset, true);
      x0 = (x0 << 51n) | float52ToInt64(x0F);
    }
    return x0;
  }

  readSecond(x: number) {
    return this.read(x + 8);
  }
}
