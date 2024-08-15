import { numberToBigint64 } from "./fma-js.js";

export { mask51, mask64, c103, c52, c51, c51x3, c2, hiPre, loPre, c52n, c51n };

// constants
let mask51 = (1n << 51n) - 1n;
let mask64 = (1n << 64n) - 1n;

let c103 = 2 ** 103;
let c52 = 2 ** 52;
let c51 = 2 ** 52;
let c51x3 = 3 * 2 ** 51;
let c2 = c103 + c51x3;

// constants we have to subtract after reinterpreting raw float bytes as int64
let hiPre = numberToBigint64(c103);
let loPre = numberToBigint64(c51x3);
let c52n = numberToBigint64(c52);
let c51n = numberToBigint64(c51);
