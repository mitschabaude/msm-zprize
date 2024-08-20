// this demonstrates how v128 relative operations work
import { $, func, i32, i32x4, i64, i64x2, local, Module, v128 } from "wasmati";

const bin = func(
  { in: [i64, i64, i64, i64], locals: [v128], out: [i32, i32] },
  ([x0, y0, x1, y1], [z]) => {
    i64x2.splat(x0);
    local.get(x1);
    i64x2.replace_lane(1);

    i64x2.splat(y0);
    local.get(y1);
    i64x2.replace_lane(1);

    i64x2.lt_s();

    local.tee(z);
    i32x4.extract_lane(0);
    i32.eq($, -1);
    local.get(z);
    i32x4.extract_lane(2);
    i32.eq($, -1);
  }
);
let module = Module({ exports: { bin } });
let { instance } = await module.instantiate();

console.log(instance.exports.bin(1n, 2n, 23n, 22n));
