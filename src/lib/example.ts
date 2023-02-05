import { exportFunction } from "./export.js";
import { func, FunctionContext } from "./function.js";
import { i32, local, ops } from "./instruction.js";
import { Module } from "./module.js";
import assert from "node:assert";

let x = local("x", i32);
let y = local("y", i32);
let ctx: FunctionContext = {
  functions: [],
  instructions: [],
  locals: [],
  stack: [],
};

let myFunc = func(
  ctx,
  "myFunc",
  { args: [x, y], locals: [], results: [i32] },
  ([x, y]) => {
    i32.const(ctx, 0);
    local.get(ctx, x);
    i32.add(ctx, undefined);
    local.get(ctx, y);
    i32.add(ctx, undefined);
  }
);

let exportedFunc = func(
  ctx,
  "exportedFunc",
  { args: [x], locals: [y], results: [i32] },
  ([x], [y]) => {
    local.get(ctx, x);
    local.set(ctx, y);
    local.get(ctx, y);
    i32.const(ctx, 5);
    myFunc();
    // ops.unreachable(ctx, undefined);
  }
);

let module: Module = {
  imports: [],
  functions: ctx.functions,
  exports: [exportFunction(exportedFunc)],
  memory: undefined,
  start: undefined,
};

console.dir(module, { depth: 10 });
let wasmByteCode = Module.toBytes(module);
console.log(wasmByteCode);
let recoveredModule = Module.fromBytes(wasmByteCode);
assert.deepStrictEqual(recoveredModule, module);

let wasmModule = await WebAssembly.instantiate(Uint8Array.from(wasmByteCode));
console.log(wasmModule.instance);
console.log(wasmModule.instance.exports);
let { exportedFunc: exportedFunc_ } = wasmModule.instance.exports as any;
let result = exportedFunc_(10);
console.log({ result });
