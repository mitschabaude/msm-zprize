import { call, func, i32, importFunc, importMemory, Module } from "wasmati";
import { MemoryHelpers, memoryHelpers } from "../wasm/memory-helpers.js";
import { Tuple } from "../types.js";
import { float52ToInt64, mask51, c51, c51n } from "./common.js";
import { Multiply } from "./fma.js";
import { assert } from "../util.js";
import { forLoop1, ImplicitMemory } from "../wasm/wasm-util.js";
import { bigintPairToData } from "./field-base.js";

export { createWasm, createWasmWithBenches, Field };

let sizeField = 8 * 5;
let sizeFieldPair = 2 * sizeField;

function validateAssumptions(modulus: bigint) {
  // slightly stronger than p < 2^255, we need some wiggle room to be able to not fully reduce
  assert(modulus + (1n << 206n) < 1n << 255n);
}

type WasmIntf = {
  multiply: (z: number, x: number, y: number) => void;
  multiplyReduce: (z: number, x: number, y: number) => void;
};

async function createWasm(p: bigint, { memSize = 1 << 16 } = {}) {
  let wasmMemory = importMemory({ min: memSize, max: memSize, shared: true });
  let implicitMemory = new ImplicitMemory(wasmMemory);
  let pSelectPtr = pSelect(p, implicitMemory);

  let { multiply } = Multiply(p, pSelectPtr);
  let { multiply: multiplyReduce } = Multiply(p, pSelectPtr, { reduce: true });

  let multiplyModule = Module({
    memory: wasmMemory,
    exports: { multiply, multiplyReduce, ...implicitMemory.getExports() },
  });
  let { instance } = await multiplyModule.instantiate();

  const Memory = memoryHelpers(p, 51, 5, { memory: wasmMemory.value });
  const memory = Memory.memoryBytes;

  return new Field<WasmIntf>(
    p,
    memory,
    instance.exports,
    Memory,
    multiplyModule.toBytes()
  );
}

function pSelect(p: bigint, implicitMemory: ImplicitMemory) {
  let p00 = implicitMemory.dataToOffset(bigintPairToData(0n, 0n));
  let p01 = implicitMemory.dataToOffset(bigintPairToData(0n, p));
  let p10 = implicitMemory.dataToOffset(bigintPairToData(p, 0n));
  let p11 = implicitMemory.dataToOffset(bigintPairToData(p, p));
  let pSelect = [p00, p01, p10, p11];
  pSelect.forEach((p) => assert(p < 256));
  return implicitMemory.data(pSelect);
}

class Field<Wasm> {
  size = sizeFieldPair;
  static size = sizeFieldPair;

  constructor(
    public modulus: bigint,
    public memory: Uint8Array,
    public Wasm: Wasm,
    public Memory: MemoryHelpers,
    public moduleBytes?: Uint8Array
  ) {
    validateAssumptions(modulus);
  }

  static create(p: bigint) {
    return createWasm(p);
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

// version with benchmark scripts

type WasmIntfWithBenches = {
  benchMultiply: (x: number, N: number) => void;
};

async function createWasmWithBenches(p: bigint) {
  let wasmMemory = importMemory({ min: 100, max: 100 });
  let implicitMemory = new ImplicitMemory(wasmMemory);
  let pSelectPtr = pSelect(p, implicitMemory);

  let { multiply } = Multiply(p, pSelectPtr, { reduce: true });

  const benchMultiply = func(
    { in: [i32, i32], locals: [i32], out: [] },
    ([x, N], [i]) => {
      forLoop1(i, 0, N, () => {
        call(multiply, [x, x, x]);
      });
    }
  );

  let multiplyModule = Module({
    memory: wasmMemory,
    exports: { multiply, benchMultiply },
  });
  let { instance } = await multiplyModule.instantiate();

  const Memory = memoryHelpers(p, 51, 5, { memory: wasmMemory.value });
  const memory = Memory.memoryBytes;

  return new Field<WasmIntfWithBenches>(p, memory, instance.exports, Memory);
}
