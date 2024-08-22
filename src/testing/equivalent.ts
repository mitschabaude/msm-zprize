/**
 * Copyright 2023 o1Labs
 *
 * This code is taken from o1js: https://github.com/o1-labs/o1js
 *
 * contains helpers for testing equivalence of two implementations
 */
import { test, Random } from "../testing/property.js";
import { deepEqual } from "node:assert/strict";
import { Tuple } from "../types.js";
import { assertDeepEqual } from "./nested.js";
import { mod } from "../bigint/field-util.js";

export {
  equivalent,
  equivalentAsync,
  throwError,
  handleErrors,
  deepEqual as defaultAssertEqual,
  id,
};
export { spec, Spec, ToSpec, FromSpec, First, Second, Params1, Params2 };

// a `Spec` tells us how to compare two functions

type FromSpec<In1, In2> = {
  // `rng` creates random inputs to the first function
  rng: Random<In1>;

  // `there` converts to inputs to the second function
  there: (x: In1) => In2;
};

type ToSpec<Out1, Out2> = {
  // `back` converts outputs of the second function back to match the first function
  back: (x: Out2) => Out1;

  // `assertEqual` to compare outputs against each other; defaults to `deepEqual`
  assertEqual?: (x: Out1, y: Out1, message: string) => void;
};

type Spec<T1, T2> = FromSpec<T1, T2> & ToSpec<T1, T2>;

// equivalence tester

function equivalent<
  In extends Tuple<FromSpec<any, any>>,
  Out extends ToSpec<any, any>,
>({ from, to, verbose }: { from: In; to: Out; verbose?: boolean }) {
  return function run(
    f1: (...args: Params1<In>) => First<Out>,
    f2: (...args: Params2<In>) => Second<Out>,
    label?: string
  ) {
    let generators = from.map((spec) => spec.rng);
    let assertEqual = to.assertEqual ?? assertDeepEqual;
    return test.custom({ logSuccess: verbose })(
      label ?? "Eqivalence test",
      ...generators,
      (...args) => {
        let inputs = args as Params1<In>;
        handleErrors(
          () => f1(...inputs),
          () =>
            to.back(
              f2(...(inputs.map((x, i) => from[i].there(x)) as Params2<In>))
            ),
          (x, y) => assertEqual(x, y, label ?? "expect equal results"),
          label
        );
      }
    );
  };
}

// async equivalence

function equivalentAsync<
  In extends Tuple<FromSpec<any, any>>,
  Out extends ToSpec<any, any>,
>({ from, to }: { from: In; to: Out }, { runs = 5 } = {}) {
  return async function run(
    f1: (...args: Params1<In>) => Promise<First<Out>> | First<Out>,
    f2: (...args: Params2<In>) => Promise<Second<Out>> | Second<Out>,
    label = "expect equal results"
  ) {
    let generators = from.map((spec) => spec.rng);
    let assertEqual = to.assertEqual ?? deepEqual;

    let nexts = generators.map((g) => g.create());

    for (let i = 0; i < runs; i++) {
      let args = nexts.map((next) => next());
      let inputs = args as Params1<In>;
      try {
        await handleErrorsAsync(
          () => f1(...inputs),
          async () =>
            to.back(
              await f2(
                ...(inputs.map((x, i) => from[i].there(x)) as Params2<In>)
              )
            ),
          (x, y) => assertEqual(x, y, label),
          label
        );
      } catch (err) {
        console.log(...inputs);
        throw err;
      }
    }
  };
}

// creating specs

function id<T>(x: T) {
  return x;
}

function spec<T, S>(
  rng: Random<T>,
  spec: {
    there: (x: T) => S;
    back: (x: S) => T;
    assertEqual?: (x: T, y: T, message: string) => void;
  }
): Spec<T, S>;
function spec<T>(
  rng: Random<T>,
  spec?: { assertEqual?: (x: T, y: T, message: string) => void }
): Spec<T, T>;
function spec<T, S>(
  rng: Random<T>,
  spec: {
    there?: (x: T) => S;
    back?: (x: S) => T;
    assertEqual?: (x: T, y: T, message: string) => void;
  } = {}
): Spec<T, S> {
  return {
    rng,
    there: spec.there ?? (id as any),
    back: spec.back ?? (id as any),
    assertEqual: spec.assertEqual,
  };
}

// some useful specs

const unit: ToSpec<void, void> = { back: id, assertEqual() {} };
const boolean: Spec<boolean, boolean> = spec(Random.boolean);

function numberLessThan(n: number): Spec<number, number> {
  return spec(Random.nat(n - 1));
}

function field(p: bigint, options?: { relaxed?: boolean; inputsX2?: boolean }) {
  let rng = options?.inputsX2 ? Random.fieldx2(p) : Random.field(p);
  return spec(rng, {
    assertEqual: options?.relaxed
      ? (x, y, message) => assertDeepEqual(mod(x, p), mod(y, p), message)
      : undefined,
  });
}

const Spec = {
  unit,
  boolean,
  numberLessThan,
  constant,
  array,
  record,
  map,
  onlyIf,
  first,
  second,
  field,
};

// spec combinators

function constant<T, S>(spec: Spec<T, S>, x: T): Spec<T, S> {
  return { ...spec, rng: Random.constant(x) };
}

function array<T, S>(
  spec: Spec<T, S>,
  n: Random<number> | number
): Spec<T[], S[]> {
  return {
    rng: Random.array(spec.rng, n),
    there: (x) => x.map(spec.there),
    back: (x) => x.map(spec.back),
  };
}

function record<Specs extends { [k in string]: Spec<any, any> }>(
  specs: Specs
): Spec<
  { [k in keyof Specs]: First<Specs[k]> },
  { [k in keyof Specs]: Second<Specs[k]> }
> {
  return {
    rng: Random.record(mapObject(specs, (spec) => spec.rng)) as any,
    there: (x) => mapObject(specs, (spec, k) => spec.there(x[k])) as any,
    back: (x) => mapObject(specs, (spec, k) => spec.back(x[k])) as any,
  };
}

function map<T1, T2, S1, S2>(
  { from, to }: { from: FromSpec<T1, T2>; to: Spec<S1, S2> },
  there: (t: T1) => S1
): Spec<S1, S2> {
  return { ...to, rng: Random.map(from.rng, there) };
}

function onlyIf<T, S>(spec: Spec<T, S>, onlyIf: (t: T) => boolean): Spec<T, S> {
  return { ...spec, rng: Random.reject(spec.rng, (x) => !onlyIf(x)) };
}

function mapObject<K extends string, T, S>(
  t: { [k in K]: T },
  map: (t: T, k: K) => S
): { [k in K]: S } {
  return Object.fromEntries(
    Object.entries<T>(t).map(([k, v]) => [k, map(v, k as K)])
  ) as any;
}

function first<T, S>(spec: Spec<T, S>): Spec<T, T> {
  return { rng: spec.rng, there: id, back: id };
}
function second<T, S>(spec: Spec<T, S>): Spec<S, S> {
  return { rng: Random.map(spec.rng, spec.there), there: id, back: id };
}

// helper to ensure two functions throw equivalent errors

function handleErrors<T, S, R>(
  op1: () => T,
  op2: () => S,
  useResults?: (a: T, b: S) => R,
  label?: string
): R | undefined {
  let result1: T, result2: S;
  let error1: Error | undefined;
  let error2: Error | undefined;
  try {
    result1 = op1();
  } catch (err) {
    error1 = err as Error;
  }
  try {
    result2 = op2();
  } catch (err) {
    error2 = err as Error;
  }
  if (!!error1 !== !!error2) {
    error1 && console.log(error1);
    error2 && console.log(error2);
  }
  let message = `${(label && `${label}: `) || ""}equivalent errors`;
  deepEqual(!!error1, !!error2, message);
  if (!(error1 || error2) && useResults !== undefined) {
    return useResults(result1!, result2!);
  }
}

async function handleErrorsAsync<T, S, R>(
  op1: () => T,
  op2: () => S,
  useResults?: (a: Awaited<T>, b: Awaited<S>) => R,
  label?: string
): Promise<R | undefined> {
  let result1: Awaited<T>, result2: Awaited<S>;
  let error1: Error | undefined;
  let error2: Error | undefined;
  try {
    result1 = await op1();
  } catch (err) {
    error1 = err as Error;
  }
  try {
    result2 = await op2();
  } catch (err) {
    error2 = err as Error;
  }
  if (!!error1 !== !!error2) {
    error1 && console.log(error1);
    error2 && console.log(error2);
  }
  let message = `${(label && `${label}: `) || ""}equivalent errors`;
  deepEqual(!!error1, !!error2, message);
  if (!(error1 || error2) && useResults !== undefined) {
    return useResults(result1!, result2!);
  }
}

function throwError(message?: string): any {
  throw Error(message);
}

// infer input types from specs

type Param1<In extends FromSpec<any, any>> = In extends {
  there: (x: infer In) => any;
}
  ? In
  : never;
type Param2<In extends FromSpec<any, any>> = In extends {
  there: (x: any) => infer In;
}
  ? In
  : never;

type Params1<Ins extends Tuple<FromSpec<any, any>>> = {
  [k in keyof Ins]: Param1<Ins[k]>;
};
type Params2<Ins extends Tuple<FromSpec<any, any>>> = {
  [k in keyof Ins]: Param2<Ins[k]>;
};

type First<Out extends ToSpec<any, any>> = Out extends ToSpec<infer Out1, any>
  ? Out1
  : never;
type Second<Out extends ToSpec<any, any>> = Out extends ToSpec<any, infer Out2>
  ? Out2
  : never;
