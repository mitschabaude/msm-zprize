import { createField } from "../bigint/field.js";
import { p as pPasta, q as qPasta } from "./pasta.params.js";
import { p as pBls12381, q as qBls12381 } from "./bls12-381.params.js";
import { p as pBls12377 } from "./bls12-377.params.js";
import {
  p as pEdBls12377,
  q as qEdBls12377,
} from "./ed-on-bls12-377.params.js";

export { exampleFields };

let pSmall = 101n;
let pM31 = (1n << 31n) - 1n;
let pBabybear = (1n << 31n) - (1n << 27n) + 1n;
let pGoldilocks = (1n << 64n) - (1n << 32n) + 1n;
let p25519 = (1n << 255n) - 19n;
let pSecp256k1 = (1n << 256n) - (1n << 32n) - 0b1111010001n;
let pSecq256k1 = (1n << 256n) - 0x14551231950b75fc4402da1732fc9bebfn;

let exampleFields = {
  pastaFp: createField(pPasta),
  pastaFq: createField(qPasta),
  small: createField(pSmall),
  m31: createField(pM31),
  babybear: createField(pBabybear),
  goldilocks: createField(pGoldilocks),
  f25519: createField(p25519),
  secp256k1: createField(pSecp256k1),
  secq256k1: createField(pSecq256k1),
  bls12381: createField(pBls12381),
  bls12381Scalar: createField(qBls12381),
  bls12377: createField(pBls12377),
  // bls12377Scalar: createField(qBls12377), // same as edBls12377
  edBls12377: createField(pEdBls12377),
  edBls12377Scalar: createField(qEdBls12377),
};
