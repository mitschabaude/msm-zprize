import { Dependency, JSFunction } from "wasmati";

export {
  AnyFunction,
  UnwrapPromise,
  WasmArtifacts,
  WasmFunctions,
  Tuple,
  AnyTuple,
  Get,
};

type AnyFunction = (...args: any) => any;

type UnwrapPromise<T> = T extends Promise<infer U> ? U : never;

type WasmArtifacts = {
  module: WebAssembly.Module;
  memory: WebAssembly.Memory;
};

type WasmFunctions<
  Exports extends Record<string, any>,
  Keys extends keyof Exports = keyof Exports,
> = Pick<
  {
    [K in keyof Exports]-?: Exports[K] extends Dependency.AnyFunc
      ? JSFunction<Exports[K]>
      : never;
  },
  {
    [K in Keys]-?: Exports[K] extends Dependency.AnyFunc ? K : never;
  }[Keys]
>;

type Tuple<T = any> = [T, ...T[]] | [];
type AnyTuple = Tuple<any>;

/**
 * helper to get property type from an object, in place of `T[Key]`
 *
 * assume `T extends { Key?: Something }`.
 * if we use `Get<T, Key>` instead of `T[Key]`, we allow `T` to be inferred _without_ the `Key` key,
 * and thus retain the precise type of `T` during inference
 */
type Get<T, Key extends string> = T extends { [K in Key]: infer Value }
  ? Value
  : undefined;
