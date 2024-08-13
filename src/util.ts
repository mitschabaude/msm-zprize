export {
  bigintFromBytes,
  bigintFromBytes32,
  bigintToBytes,
  bigintToBytes32,
  bigintToBits,
  bigintToLimbs,
  bigintToLimbsRelaxed,
  bigintFromLimbs,
  logBytesAsBigint,
  log2,
  extractBitSlice,
  mapRange,
  randomBytes,
  bytesEqual,
  divide,
  scale,
  max,
  abs,
  sign,
  assert,
  TODO,
  chunk,
};

function bigintFromBytes(bytes: Uint8Array) {
  let x = 0n;
  let bitPosition = 0n;
  for (let i = 0, n = bytes.length; i < n; i++) {
    x += BigInt(bytes[i]) << bitPosition;
    bitPosition += 8n;
  }
  return x;
}

function bigintToBytes(x: bigint, length: number | undefined): Uint8Array {
  let bytes = [];
  for (; x > 0; x >>= 8n) {
    bytes.push(Number(x & 0xffn));
  }
  let array = new Uint8Array(bytes);
  if (length === undefined) return array;
  if (array.length > length)
    throw Error("bigint doesn't fit into" + length + " bytes.");
  let sizedArray = new Uint8Array(length);
  sizedArray.set(array);
  return sizedArray;
}

function bigintFromBytes32(bytes: Uint8Array) {
  let words = new BigUint64Array(bytes.buffer, bytes.byteOffset, 4);
  return words[0] | (words[1] << 64n) | (words[2] << 128n) | (words[3] << 192n);
}

const mask64 = (1n << 64n) - 1n;

function bigintToBytes32(x: bigint): Uint8Array {
  let words = new BigUint64Array(4);
  words[0] = x & mask64;
  words[1] = (x >> 64n) & mask64;
  words[2] = (x >> 128n) & mask64;
  words[3] = x >> 192n;
  return new Uint8Array(words.buffer);
}

function bigintToBits(x: bigint, bitLength?: number): boolean[] {
  let bits = Array(bitLength || 0);
  for (let i = 0; bitLength ? i < bitLength : x > 0n; i++) {
    bits[i] = !!Number(x & 1n);
    x >>= 1n;
  }
  return bits;
}

/**
 * Split bigint into n w-bit limbs, which are also bigints
 * @param x0
 * @param w word size
 * @param n number of limbs
 */
function bigintToLimbs(x0: bigint, w: number, n: number) {
  let limbs: bigint[] = Array(n);
  let wn = BigInt(w);
  let wordMax = (1n << wn) - 1n;
  for (let i = 0; i < n; i++) {
    limbs[i] = x0 & wordMax;
    x0 >>= wn;
  }
  if (x0 !== 0n) throw Error("input too large");
  return limbs;
}

/**
 * Split bigint into n w-bit limbs, which are also bigints
 *
 * In contrast to {@link bigintToLimbs}, this function allows high limbs larger than w bits.
 *
 * @param x0
 * @param w word size
 * @param n number of limbs
 */
function bigintToLimbsRelaxed(x0: bigint, w: number, n: number) {
  let limbs: bigint[] = Array(n);
  let wn = BigInt(w);
  let wordMax = (1n << wn) - 1n;
  for (let i = 0; i < n; i++) {
    limbs[i] = x0 & wordMax;
    x0 >>= wn;
  }
  return limbs;
}

/**
 * Split bigint into n w-bit limbs, which are also bigints
 * @param x
 * @param w word size
 * @param n number of limbs
 */
function bigintToLimbsSigned(x: bigint, w: number, n: number) {
  let limbs: bigint[] = Array(n);
  let wn = BigInt(w);
  let max = 1n << wn;
  let halfMax = 1n << (wn - 1n);
  for (let i = 0; i < n; i++) {
    let limb = x & (max - 1n);
    if (limb >= halfMax) {
      limb -= max;
      x += max;
    }
    limbs[i] = limb;
    x >>= wn;
  }
  assert(x === 0n, `input too large`);
  return limbs;
}
// let X = 2n ** 29n;
// console.log(
//   bigintToLimbsSigned(((X / 2n - 1n) * (X ** 9n - 1n)) / (X - 1n), 29, 9)
// );

function bigintFromLimbs(x: BigUint64Array | bigint[], w: number, n: number) {
  let wn = BigInt(w);
  let x0 = x[n - 1];
  for (let i = n - 2; i >= 0; i--) {
    x0 = x[i] + (x0 << wn);
  }
  return x0;
}

function logBytesAsBigint(bytes: Uint8Array) {
  let x = bigintFromBytes(bytes);
  console.log(x);
}

/**
 * ceil(log2(n))
 * = smallest k such that n <= 2^k
 */
function log2(n: number | bigint) {
  if (typeof n === "number") n = BigInt(n);
  if (n === 1n) return 0;
  return (n - 1n).toString(2).length;
}

/**
 * divide two bigints to return a float of given precision
 */
function divide(x: bigint, y: bigint, prec = 10) {
  let length = y.toString(10).length;
  let exp = BigInt(length - prec);
  return Number(x / 10n ** exp) / Number(y / 10n ** exp);
}

/**
 * scale bigint by a float
 */
function scale(c: number, x: bigint, prec = 10) {
  return (BigInt(Math.round(c * 10 ** prec)) * x) / 10n ** BigInt(prec);
}

function max(a: bigint, b: bigint) {
  return a > b ? a : b;
}

function abs(x: bigint) {
  return x < 0n ? -x : x;
}

function sign(x: bigint) {
  return x >= 0 ? 1n : -1n;
}

/**
 *
 * @param {Uint8Array} bytes
 * @param {number} startBit
 * @param {number} bitLength
 */
function extractBitSlice(
  bytes: Uint8Array,
  startBit: number,
  bitLength: number
) {
  let endBit = startBit + bitLength;
  let startByte = startBit >> 3;
  startBit -= startByte << 3;
  let endByte = endBit >> 3;
  endBit -= endByte << 3;
  if (startByte === endByte) {
    return ((bytes[startByte] || 0) & ((1 << endBit) - 1)) >> startBit;
  }
  let slice = (bytes[startByte] || 0) >> startBit;
  let position = 8 - startBit;
  for (let i = startByte + 1; i < endByte; i++) {
    slice += (bytes[i] || 0) << position;
    position += 8;
  }
  slice += ((bytes[endByte] || 0) & ((1 << endBit) - 1)) << position;
  return slice;
}

function randomBytes(n: number) {
  let arr = new Uint8Array(n);
  for (let i = 0; i < n; i += 65536) {
    let m = Math.min(n - i, 65536);
    globalThis.crypto.getRandomValues(arr.subarray(i, i + m));
  }
  return arr;
}

function mapRange<T>(n: number, callback: (i: number) => T) {
  return Array(n)
    .fill(0)
    .map((_, i) => callback(i));
}

function chunk<T>(array: T[], size: number): T[][] {
  assert(array.length % size === 0, "invalid input length");
  return Array.from({ length: array.length / size }, (_, i) =>
    array.slice(size * i, size * (i + 1))
  );
}

function bytesEqual(b1: Uint8Array, b2: Uint8Array) {
  if (b1.length !== b2.length) return false;
  for (let i = 0; i < b1.length; i++) {
    if (b1[i] !== b2[i]) return false;
  }
  return true;
}

function assert(condition: boolean, message?: string): asserts condition {
  if (!condition)
    throw Error(
      message === undefined
        ? "Assertion failed"
        : `Assertion failed: ${message}`
    );
}

function TODO(message?: string, FALSE: false = false): asserts FALSE {
  throw Error("TODO: " + (message ?? "this function is not implemented yet"));
}
