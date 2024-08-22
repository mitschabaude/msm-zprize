import { Const, Module, call, drop, func, global, i32, memory } from "wasmati";
import { tic, toc } from "../../src/testing/tictoc.js";
import { multiplyMontgomery } from "../../src/wasm/multiply-montgomery.js";
import { memoryHelpers } from "../../src/wasm/memory-helpers.js";
import { writeWat } from "../../src/wasm/wat-helpers.js";
import { multiplySchoolbook } from "../../src/wasm/multiply-schoolbook.js";
import { multiplyBarrett } from "../../src/wasm/barrett.js";
import { FieldWithArithmetic } from "../../src/wasm/field-arithmetic.js";
import { ImplicitMemory, forLoop1 } from "../../src/wasm/wasm-util.js";
import { fieldInverse } from "../../src/wasm/inverse.js";
import { fieldExp } from "../../src/wasm/exp.js";
import { createSqrt } from "../../src/field-sqrt.js";
import { createConstants } from "../../src/field-msm.js";
import { mod, montgomeryParams } from "../../src/bigint/field-util.js";
import { fastInverse } from "../../src/inverse/faster-inverse-wasm.js";
import {
  bigintFromBytes,
  bigintFromBytes32,
  bigintToBytes,
  bigintToBytes32,
  log2,
  randomBytes,
} from "../../src/util.js";
import { randomGenerators } from "../../src/bigint/field-random.js";
import { createWasmWithBenches } from "../../src/51x5/field.js";

export { benchmark };

async function benchmark(
  { p, t }: { p: bigint; t: bigint },
  { doWrite = false, onlyQuick = false } = {}
) {
  let { randomField, randomFieldx2 } = randomGenerators(p);
  let N = 1e7;
  let Ninv = 5e5;
  let Npow = 5e4;

  if (p < 1n << 255n) {
    let Fp = await createWasmWithBenches(p);
    let x = Fp.Memory.local.getPointer(Fp.size);
    Fp.writePair(x, randomField(), randomField());
    bench("multiply 51x5", Fp.Wasm.benchMultiply, { x, N }, 2);
    Fp.writePair(x, randomField(), randomField());
    bench("multiply 51x5", Fp.Wasm.benchMultiply, { x, N }, 2);

    Fp.writeSingle(x, randomField());
    bench("multiply 51x5 single", Fp.Wasm.benchMultiplySingle, { x, N });

    Fp.writePair(x, randomField(), randomField());
    bench("multiply 51x5 no fma", Fp.Wasm.benchMultiplyNoFma, { x, N }, 2);
  }

  for (let w of [29]) {
    let { n } = montgomeryParams(p, w);
    let {
      benchMultiply: benchMontgomery,
      benchSquare,
      multiply,
      leftShift,
      square,
    } = multiplyMontgomery(p, w, n, { countMultiplications: false });
    let { benchMultiply: benchSchoolbook, multiply: multiplySchoolbook_ } =
      multiplySchoolbook(p, w, n);
    let { benchMultiply: benchBarrett } = multiplyBarrett(
      p,
      w,
      n,
      multiplySchoolbook_
    );

    const Field = {
      ...FieldWithArithmetic(p, w, n),
      multiply,
      leftShift,
      square,
    };

    const benchAdd = func(
      { in: [i32, i32], locals: [i32], out: [] },
      ([x, N], [i]) => {
        forLoop1(i, 0, N, () => {
          for (let i = 0; i < 3; i++) {
            call(Field.add, [x, x, x]);
          }
        });
      }
    );

    let implicitMemory = new ImplicitMemory(memory({ min: 100 }));

    let { inverse } = fieldInverse(implicitMemory, Field);

    const benchInverse = func(
      { in: [i32, i32, i32, i32], locals: [i32], out: [] },
      ([scratch, x, y, N], [i]) => {
        forLoop1(i, 0, N, () => {
          // x <- x + y
          call(Field.add, [x, x, y]);
          // y <- 1/x
          call(inverse, [scratch, y, x]);
        });
      }
    );

    let { almostInverse } = fastInverse(implicitMemory, Field);

    const benchFastAlmostInverse = func(
      { in: [i32, i32, i32, i32], locals: [i32], out: [] },
      ([scratch, x, y, N], [i]) => {
        forLoop1(i, 0, N, () => {
          // x <- x + y
          call(Field.add, [x, x, y]);
          // y <- 1/x
          call(almostInverse, [scratch, y, x]);
          drop();
        });
      }
    );

    let module = Module({
      exports: {
        benchMontgomery,
        benchSchoolbook,
        benchBarrett,
        benchSquare,
        benchAdd,
        benchInverse,
        benchFastAlmostInverse,
        exp: fieldExp(Field),

        memory: implicitMemory.memory,
        dataOffset: global(Const.i32(implicitMemory.dataOffset)),

        // stuff needed for sqrt
        copy: Field.copy,
        add: Field.add,
        reduce: Field.reduce,
        isEqual: Field.isEqual,
        isZero: Field.isZero,
        multiply: Field.multiply,
        square: Field.square,
        inverse,
      },
    });
    if (doWrite) {
      await writeWat(
        import.meta.url.slice(7).replace(".ts", ".wat"),
        module.toBytes()
      );
    }

    let wasm = (await module.instantiate()).instance.exports;
    let helpers = memoryHelpers(p, w, n, wasm);
    let { writeBigint, readBigint, getPointer, getPointers } = helpers;

    function benchMultiplyBigint(x0: number, N: number) {
      let x = readBigint(x0);
      for (let i = 0; i < N; i++) {
        x = (x * x) % p;
      }
      (globalThis as any).x = x;
      return x;
    }

    let constants = createConstants(helpers, {
      zero: 0n,
      mg1: mod(1n * Field.R, p),
      mg2: mod(2n * Field.R, p),
    });
    let { sqrt } = createSqrt(Field, wasm, helpers, constants);

    function benchSqrt(x: number, y: number, N: number) {
      let scratch = getPointers(5);
      for (let i = 0; i < N; i++) {
        wasm.add(x, x, y);
        sqrt(scratch, x, x);
      }
      return x;
    }

    let t1 = getPointer();
    writeBigint(t1, (t - 1n) / 2n);

    function benchPow(x: number, N: number) {
      let scratch = getPointer();
      for (let i = 0; i < N; i++) {
        wasm.exp(scratch, x, x, t1);
      }
      return x;
    }

    let [scratch] = getPointers(2);
    let x = getPointer();
    let y = getPointer();
    writeBigint(x, randomFieldx2());
    writeBigint(y, randomFieldx2());

    console.log(`w=${w}, n=${n}, nw=${n * w}, op x ${N}\n`);

    let tMul = bench("multiply montgomery", wasm.benchMontgomery, { x, N });
    bench("multiply barrett", wasm.benchBarrett, { x, N });
    bench("multiply schoolbook", wasm.benchSchoolbook, { x, N });
    bench("multiply square", wasm.benchSquare, { x, N });

    // bench("multiply bigint", benchMultiplyBigint, { x, N });
    bench("add", wasm.benchAdd, { x, N }, 3);

    if (onlyQuick) continue;

    writeBigint(x, randomFieldx2());
    writeBigint(y, randomFieldx2());

    bench2("inverse", () => wasm.benchInverse(scratch, x, y, Ninv), {
      N: Ninv,
      tMul,
    });
    bench2(
      "fast inverse",
      () => wasm.benchFastAlmostInverse(scratch, x, y, Ninv),
      { N: Ninv, tMul }
    );
    bench2("pow", () => benchPow(x, Npow), { N: Npow, tMul });
    bench2("sqrt", () => benchSqrt(x, y, Npow), { N: Npow, tMul });

    let bytess: Uint8Array[] = Array(Ninv);
    let bigints = Array<bigint>(Ninv);
    let sizeInBits = log2(p);
    let sizeInBytes = Math.ceil(sizeInBits / 8);

    bench2(
      "randomBytes",
      () => {
        for (let i = 0; i < Ninv; i++) {
          bytess[i] = randomBytes(sizeInBytes);
        }
      },
      { N: Ninv, tMul }
    );

    // bigint from bytes
    bench2(
      "bigintFromBytes",
      () => {
        let n = bytess.length;
        for (let i = 0; i < n; i++) {
          bigints[i] = bigintFromBytes(bytess[i]);
        }
      },
      { N: Ninv, tMul }
    );

    bench2(
      "bigintFromBytes32",
      () => {
        let n = bytess.length;
        for (let i = 0; i < n; i++) {
          bigints[i] = bigintFromBytes32(bytess[i]);
        }
      },
      { N: Ninv, tMul }
    );

    // bigint to bytes
    bench2(
      "bigintToBytes",
      () => {
        let n = bigints.length;
        for (let i = 0; i < n; i++) {
          bytess[i] = bigintToBytes(bigints[i], sizeInBytes);
        }
      },
      { N: Ninv, tMul }
    );

    bench2(
      "bigintToBytes32",
      () => {
        let n = bigints.length;
        for (let i = 0; i < n; i++) {
          bytess[i] = bigintToBytes32(bigints[i]);
        }
      },
      { N: Ninv, tMul }
    );
  }
}

function bench(
  name: string,
  compute: (x: number, N: number) => void,
  { x, N }: { x: number; N: number },
  /**
   * parameter to use if the operation is performed multiple times
   */
  scale = 1
) {
  let Nscaled = Math.round(N / scale);
  name = name.padEnd(20, " ");
  tic();
  compute(x, Nscaled);
  let time = toc();
  console.log(`${name} \t ${(N / time / 1e3).toFixed(1).padStart(4)}M ops/s`);
  console.log(`${name} \t ${((time / N) * 1e6).toFixed(0)}ns`);
  console.log();
  return time / N;
}

function bench2(
  name: string,
  compute: () => void,
  { N, tMul }: { N: number; tMul: number }
) {
  name = name.padEnd(20, " ");
  tic();
  compute();
  let time = toc();
  console.log(`${name} \t ${(N / time).toFixed(0).padStart(4)}K ops/s`);
  console.log(
    `${name} \t ${(time / N / tMul).toFixed(0)} muls / ${(
      (time / N) *
      1e6
    ).toFixed(0)}ns`
  );
  console.log();
}
