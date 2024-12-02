import { i32, i64, type ImportFunc, importFunc } from "wasmati";

export { log32, log64 };

let log32 = importFunc({ in: [i32], out: [i32] }, (x: number) => {
  console.log(x);
  return x;
});

let log64 = importFunc({ in: [i64], out: [i64] }, (x: number) => {
  console.log(x);
  return x;
});
