import {
  add,
  multiply,
  reduceInPlace,
  storeFieldIn,
  subtract,
} from "./src/finite-field.wat.js";
import {
  field,
  fieldFromUint64Array,
  fieldToUint64Array,
  leftShiftInPlace,
  mod,
  modInverseMontgomery,
  randomBaseField,
  rightShiftInPlace,
} from "./src/finite-field.js";
import { getScratchSpace } from "./src/curve.js";
import { readField, writeFieldInto } from "./src/wasm.js";

let { p, toWasm, ofWasm } = field;

let [x, y, z, ...scratch] = getScratchSpace(10);

function test() {
  let x0 = randomBaseField() + randomBaseField();
  let y0 = randomBaseField() + randomBaseField();
  toWasm(x0, x);
  toWasm(y0, y);

  // multiply
  let z0 = mod(x0 * y0, p);
  multiply(z, x, y);
  let z1 = ofWasm(scratch, z);
  if (z0 !== z1) throw Error("multiply");

  // add
  z0 = mod(x0 + y0, p);
  add(z, x, y);
  z1 = ofWasm(scratch, z);
  if (z0 !== z1) throw Error("add");

  // subtract
  z0 = mod(x0 - y0, p);
  subtract(z, x, y);
  z1 = ofWasm(scratch, z);
  if (z0 !== z1) throw Error("subtract");

  // reduceInPlace
  z0 = x0 >= p ? x0 - p : x0;
  storeFieldIn(x, z);
  reduceInPlace(z);
  z1 = ofWasm(scratch, z);
  if (z0 !== z1) throw Error("reduceInPlace");

  // inverse
  modInverseMontgomery(scratch, z, x);
  multiply(z, z, x);
  z1 = ofWasm(scratch, z);
  if (z1 !== 1n) throw Error("inverse");

  // right shift
  storeFieldIn(x, z);
  z0 = fieldFromUint64Array(readField(z));
  rightShiftInPlace(z, 27);
  z0 >>= 27n;
  z1 = fieldFromUint64Array(readField(z));
  if (z0 !== z1) throw Error("rightShiftInPlace");

  // left shift
  writeFieldInto(z, fieldToUint64Array(2934n));
  z0 = fieldFromUint64Array(readField(z));
  leftShiftInPlace(z, 28);
  z0 <<= 28n;
  z1 = fieldFromUint64Array(readField(z));
  if (z0 !== z1) throw Error("leftShiftInPlace");
}

for (let i = 0; i < 20; i++) {
  test();
}
