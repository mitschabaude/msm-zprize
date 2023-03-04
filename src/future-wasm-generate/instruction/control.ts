import { Undefined } from "../binable.js";
import * as Dependency from "../dependency.js";
import { U32 } from "../immediate.js";
import {
  getFrameFromLabel,
  Label,
  labelTypes,
  popStack,
  pushStack,
  RandomLabel,
  setUnreachable,
} from "../local-context.js";
import { ResultType } from "../types.js";
import {
  baseInstruction,
  createExpressionWithType,
  FunctionTypeInput,
  resolveExpression,
  simpleInstruction,
} from "./base.js";
import { Block, IfBlock } from "./binable.js";

export { control };
export { unreachable, call, nop, block, loop };

const nop = simpleInstruction("nop", Undefined, {});

// TODO
const unreachable = baseInstruction("unreachable", Undefined, {
  create(ctx) {
    setUnreachable(ctx);
    return {};
  },
  resolve: () => undefined,
});

const block = baseInstruction("block", Block, {
  create(ctx, t: FunctionTypeInput, run: (label: RandomLabel) => void) {
    let { type, body, deps } = createExpressionWithType("block", ctx, t, run);
    console.log({ type });
    return {
      in: type.args,
      out: type.results,
      deps: [Dependency.type(type), ...deps],
      resolveArgs: [body],
    };
  },
  resolve([blockType, ...deps], body: Dependency.Instruction[]) {
    let instructions = resolveExpression(deps, body);
    return { blockType, instructions };
  },
});

const loop = baseInstruction("loop", Block, {
  create(ctx, t: FunctionTypeInput, run: (label: RandomLabel) => void) {
    let { type, body, deps } = createExpressionWithType("loop", ctx, t, run);
    return {
      in: type.args,
      out: type.results,
      deps: [Dependency.type(type), ...deps],
      resolveArgs: [body],
    };
  },
  resolve([blockType, ...deps], body: Dependency.Instruction[]) {
    let instructions = resolveExpression(deps, body);
    return { blockType, instructions };
  },
});

const if_ = baseInstruction("if", IfBlock, {
  create(
    ctx,
    t: FunctionTypeInput,
    runIf: (label: RandomLabel) => void,
    runElse?: (label: RandomLabel) => void
  ) {
    popStack(ctx, ["i32"]);
    let { type, body, deps } = createExpressionWithType("if", ctx, t, runIf);
    let ifArgs: ResultType = [...type.args, "i32"];
    if (runElse === undefined) {
      pushStack(ctx, ["i32"]);
      return {
        in: ifArgs,
        out: type.results,
        deps: [Dependency.type(type), ...deps],
        resolveArgs: [body, undefined],
      };
    }
    let elseExpr = createExpressionWithType("else", ctx, t, runElse);
    pushStack(ctx, ["i32"]);
    // if (!functionTypeEquals(type, elseExpr.type)) {
    //   throw Error(
    //     `Type signature of else branch doesn't match if branch.\n` +
    //       `If branch: ${printFunctionType(type)}\n` +
    //       `Else branch: ${printFunctionType(elseExpr.type)}`
    //   );
    // }
    return {
      in: ifArgs,
      out: type.results,
      deps: [Dependency.type(type), ...deps, ...elseExpr.deps],
      resolveArgs: [body, elseExpr.body],
    };
  },
  resolve(
    [blockType, ...deps],
    ifBody: Dependency.Instruction[],
    elseBody?: Dependency.Instruction[]
  ) {
    let ifDepsLength = ifBody.reduce((acc, i) => acc + i.deps.length, 0);
    let if_ = resolveExpression(deps.slice(0, ifDepsLength), ifBody);
    let else_ =
      elseBody && resolveExpression(deps.slice(ifDepsLength), elseBody);
    return { blockType, instructions: { if: if_, else: else_ } };
  },
});

// const br = baseInstruction("br", U32, {
//   create(ctx, label: Label | number) {
//     let [i, frame] = getFrameFromLabel(ctx, label);
//     let types = frame.startTypes;
//     popStack(ctx, types);
//   },
// });

const call = baseInstruction("call", U32, {
  create(_, func: Dependency.AnyFunc) {
    return { in: func.type.args, out: func.type.results, deps: [func] };
  },
  resolve: ([funcIndex]) => funcIndex,
});

const control = { nop, unreachable, block, loop, if: if_, call };
