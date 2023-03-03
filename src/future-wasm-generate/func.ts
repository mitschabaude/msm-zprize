import { Binable, iso, record, tuple } from "./binable.js";
import * as Dependency from "./dependency.js";
import { U32, vec, withByteLength } from "./immediate.js";
import { Expression, Instruction } from "./instruction/instruction.js";
import { LocalContext, popStack, withContext } from "./local-context.js";
import {
  FunctionIndex,
  FunctionType,
  JSValue,
  TypeIndex,
  valueType,
  ValueType,
  ValueTypeLiteral,
  valueTypes,
} from "./types.js";

// external
export { func };
// internal
export { Dependency, Func, Code, JSFunctionType };

function func<
  Args extends Record<string, ValueTypeLiteral>,
  Locals extends Record<string, ValueTypeLiteral>,
  Results extends Tuple<ValueTypeLiteral>
>(
  ctx: LocalContext,
  {
    in: args,
    locals,
    out: results,
  }: {
    in: ToTypeRecord<Args>;
    locals: ToTypeRecord<Locals>;
    out: ToTypeTuple<Results>;
  },
  run: (args: ToLocal<Args>, locals: ToLocal<Locals>) => void
): {
  kind: "function";
  locals: ValueType[];
  body: Dependency.Instruction[];
  deps: Dependency.t[];
  type: FullFunctionType<Args, Results>;
} {
  ctx.stack = [];
  let argsArray = valueTypes(Object.values(args));
  let localsArray = valueTypes(Object.values(locals));
  let resultsArray = valueTypes(results);
  let type: FullFunctionType<Args, Results> & FunctionType = {
    args: argsArray as any,
    results: resultsArray as any,
  };
  let nArgs = argsArray.length;
  let argsInput = Object.fromEntries(
    Object.entries(args).map(([key], index) => [key, { index }])
  ) as ToLocal<Args>;
  let localsInput = Object.fromEntries(
    Object.entries(locals).map(([key], index) => [
      key,
      { index: index + nArgs },
    ])
  ) as ToLocal<Locals>;
  let { body, deps } = withContext(
    ctx,
    {
      locals: [...argsArray, ...localsArray],
      body: [],
      deps: [],
      return: resultsArray,
      labels: [resultsArray],
    },
    () => {
      run(argsInput, localsInput);
      popStack(ctx.stack, results as ValueType[]);
      // TODO nice error
      if (ctx.stack.length !== 0) throw Error("expected stack to be empty");
    }
  );
  let func = {
    kind: "function",
    type,
    body,
    deps,
    locals: localsArray,
  } satisfies Dependency.Func;
  return func;
}

// type inference of function signature

// example:
// type Test = JSFunctionType<FullFunctionType<{ x: "i64"; y: "i32" }, ["i32"]>>;
//       ^ (arg_0: bigint, arg_1: number) => number

type FullFunctionType<
  Args extends Record<string, ValueTypeLiteral>,
  Results extends readonly ValueTypeLiteral[]
> = {
  args: [
    ...ObjValueTuple<{
      [K in keyof Args]: { kind: Args[K] };
    }>
  ];
  results: {
    [K in keyof Results]: { kind: Results[K] };
  };
};

type JSValues<T extends readonly ValueType[]> = {
  [i in keyof T]: JSValue<T[i]>;
};
type ReturnValues<T extends readonly ValueType[]> = T extends []
  ? void
  : T extends [ValueType]
  ? JSValue<T[0]>
  : JSValues<T>;

type JSFunctionType_<Args extends ValueType[], Results extends ValueType[]> = (
  ...arg: JSValues<Args>
) => ReturnValues<Results>;

type JSFunctionType<T extends FunctionType> = JSFunctionType_<
  T["args"],
  T["results"]
>;

type Local<L extends ValueTypeLiteral> = { type?: { kind: L }; index: number };

type ToLocal<T extends Record<string, ValueTypeLiteral>> = {
  [K in keyof T]: Local<T[K]>;
};
type ToTypeRecord<T extends Record<string, ValueTypeLiteral>> = {
  [K in keyof T]: { kind: T[K] };
};
type ToTypeTuple<T extends readonly ValueTypeLiteral[]> = {
  [K in keyof T]: { kind: T[K] };
};

type Tuple<T> = [] | [T, ...T[]];

type ObjValueTuple<
  T,
  K extends any[] = UnionToTuple<keyof T>,
  R extends any[] = []
> = K extends [infer K, ...infer KT]
  ? ObjValueTuple<T, KT, [...R, T[K & keyof T]]>
  : R;

// bad hack :/

type UnionToIntersection<U> = (
  U extends any ? (arg: U) => any : never
) extends (arg: infer I) => void
  ? I
  : never;

type UnionToTuple<T> = UnionToIntersection<
  T extends any ? (t: T) => T : never
> extends (_: any) => infer W
  ? [...UnionToTuple<Exclude<T, W>>, W]
  : [];

type Func = {
  // TODO remove
  functionIndex: FunctionIndex;
  typeIndex: TypeIndex;
  type: FunctionType;
  locals: ValueType[];
  body: Instruction[];
};

// binable

const CompressedLocals = vec(tuple([U32, ValueType]));
const Locals = iso<[number, ValueType][], ValueType[]>(CompressedLocals, {
  to(locals) {
    let count: Record<string, number> = {};
    for (let local of locals) {
      count[local.kind] ??= 0;
      count[local.kind]++;
    }
    return Object.entries(count).map(([kind, count]) => [
      count,
      valueType(kind as ValueTypeLiteral),
    ]);
  },
  from(compressed) {
    let locals: ValueType[] = [];
    for (let [count, local] of compressed) {
      locals.push(...Array(count).fill(local));
    }
    return locals;
  },
});

type Code = { locals: ValueType[]; body: Expression };
const Code = withByteLength(
  record({ locals: Locals, body: Expression })
) satisfies Binable<Code>;
