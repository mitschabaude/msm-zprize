import { MsmField } from "../field-msm.js";
import { MemorySection } from "../wasm/memory-helpers.js";
import { Get, Tuple } from "../types.js";
import {
  First,
  FromSpec,
  Params1,
  Params2,
  Second,
  ToSpec,
  handleErrors,
  spec,
} from "./equivalent.js";
import { TestParams, test } from "./property.js";
import { Random } from "./random.js";
import { assertDeepEqual } from "./nested.js";

export { createEquivalentWasm, wasmSpec, WasmSpec };

type WasmFromSpec<In> = FromSpec<In, number> & { size: number };
type WasmToSpec<Out> = ToSpec<Out, number> & { size: number };
type WasmSpec<T> = WasmFromSpec<T> & WasmToSpec<T>;

type IsWasm<Spec> = Spec extends { size: number } ? true : false;

// wasm specs

// TODO for better comparison, need field spec which has the bigint in montgomery form
// and keeps bigints in unreduced range [0, 2p)

const WasmSpec = {
  fieldWithRng(
    Field: MsmField,
    rng: Random<bigint>,
    montgomeryTransform = true
  ) {
    if (!montgomeryTransform) {
      return wasmSpec(Field, rng, {
        size: Field.sizeField,
        there: Field.writeBigint,
        back: Field.readBigint,
      });
    } else {
      return wasmSpec(Field, rng, {
        size: Field.sizeField,
        there: Field.fromBigint,
        back: Field.toBigint,
      });
    }
  },

  field(Field: MsmField, { montgomeryTransform = true } = {}) {
    return WasmSpec.fieldWithRng(
      Field,
      Random.field(Field.p),
      montgomeryTransform
    );
  },
  fieldUnreduced(Field: MsmField, { montgomeryTransform = true } = {}) {
    return WasmSpec.fieldWithRng(
      Field,
      Random.fieldx2(Field.p),
      montgomeryTransform
    );
  },

  boolean: spec(Random.boolean, {
    there: (b): number => (b ? 1 : 0),
    back: (b) => b === 1,
  }),

  constant<T>(spec: WasmSpec<T>, x: T): WasmSpec<T> {
    return { ...spec, rng: Random.constant(x) };
  },
};

// wasm equivalence test

type WasmParamsWithScratch<
  Signature extends {
    from: Tuple<FromSpec<any, any>>;
    to: ToSpec<any, any>;
    scratch?: number | undefined;
  },
> = IsWasm<Signature["to"]> extends true
  ? Get<Signature, "scratch"> extends undefined
    ? [out: number, ...Params2<Signature["from"]>]
    : [scratch: number[], out: number, ...Params2<Signature["from"]>]
  : Get<Signature, "scratch"> extends undefined
    ? Params2<Signature["from"]>
    : [scratch: number[], ...Params2<Signature["from"]>];

type WasmReturn<Out extends ToSpec<any, any>> = IsWasm<Out> extends true
  ? void
  : Second<Out>;

function createEquivalentWasm(
  Memory: { local: MemorySection },
  testParams?: TestParams
) {
  return function run<
    const Signature extends {
      from: Tuple<FromSpec<any, any>>;
      to: ToSpec<any, any>;
      scratch?: number | undefined;
    },
  >(
    { from, to, scratch }: Signature,
    f1: (...args: Params1<Signature["from"]>) => First<Signature["to"]>,
    f2: (
      ...args: WasmParamsWithScratch<Signature>
    ) => WasmReturn<Signature["to"]>,
    label?: string
  ) {
    let generators = from.map((spec) => spec.rng);
    let assertEqual = to.assertEqual ?? assertDeepEqual;
    let isWasmOutput = "size" in to;
    let outPtr = isWasmOutput ? Memory.local.getPointer((to as any).size) : 0;
    let scratchPtrs =
      scratch === undefined ? [] : Memory.local.getPointers(scratch);

    return test.custom(testParams)(
      label ?? "Wasm eqivalence test",
      ...generators,
      (...args) => {
        let inputs = args as Params1<Signature["from"]>;

        handleErrors(
          () => f1(...inputs),
          () => {
            let inputs2 = inputs.map((x: any, i: number) => from[i].there(x));
            if (isWasmOutput) inputs2 = [outPtr, ...inputs2];
            if (scratch !== undefined) inputs2 = [scratchPtrs, ...inputs2];
            let result = f2(...(inputs2 as WasmParamsWithScratch<Signature>));
            return to.back(isWasmOutput ? outPtr : result);
          },
          (bigint, wasm) => {
            // we switch the order because the bigint output is typically the reference,
            // while `assertEqual` treats the inputs as (actual, expected)
            assertEqual(wasm, bigint, `${label}: Expect equal results`);
          },
          label
        );
      }
    );
  };
}

// helper for writing wasm specs

function wasmSpec<T>(
  Memory: { local: MemorySection },
  rng: Random<T>,
  {
    size,
    there,
    back,
    assertEqual,
  }: {
    size: number;
    there: (xPtr: number, x: T) => void;
    back: (xPtr: number) => T;
    assertEqual?: (x: T, y: T, message: string) => void;
  }
): WasmSpec<T> {
  return {
    rng,
    size,
    there(x) {
      let xPtr = Memory.local.getPointer(size);
      there(xPtr, x);
      return xPtr;
    },
    back(xPtr) {
      return back(xPtr);
    },
    assertEqual,
  };
}
