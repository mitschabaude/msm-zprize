// this demonstrates how v128 relative operations work
import {
  $,
  func,
  i32,
  i32x4,
  i64,
  i64x2,
  i8x16,
  local,
  Module,
  v128,
} from "wasmati";

const bin = func(
  { in: [i64, i64], locals: [v128], out: [i64, i64] },
  ([x0, x1], [z]) => {
    // create z = [x0, x1]
    i64x2.splat(x0);
    local.get(x1);
    i64x2.replace_lane(1);
    local.set(z);

    i8x16.swizzle(
      z,
      v128.const(
        "i8x16",
        [8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7]
      )
    );
    local.set(z);

    // return [x1, x0]
    local.get(z);
    i64x2.extract_lane(0);
    local.get(z);
    i64x2.extract_lane(1);
  }
);
let module = Module({ exports: { bin } });
let { instance } = await module.instantiate();

console.log(instance.exports.bin(1_000_000_000_987n, 2_000_000_000_345n));
