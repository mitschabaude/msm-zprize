/**
 * Copyright 2023 o1Labs
 *
 * This code is taken from o1js: https://github.com/o1-labs/o1js
 */
import { randomGenerators } from "../bigint/field-random.js";
import { Tuple } from "../types.js";
import { bigintFromBytes, log2, randomBytes } from "../util.js";

export { Random, sample, sampleOne };

type Random<T> = {
  create(): () => T;
  invalid?: Random<T>;
};
type RandomWithInvalid<T> = Required<Random<T>>;

type InferRandom<T extends Random<any>> = T extends Random<infer U> ? U : never;

function Random_<T>(
  next: () => T,
  toInvalid?: (valid: Random<T>) => Random<T>
): Random<T> {
  let rng: Random<T> = { create: () => next };
  if (toInvalid !== undefined) rng.invalid = toInvalid(rng);
  return rng;
}

function sample<T>(rng: Random<T>, size: number) {
  let next = rng.create();
  return Array.from({ length: size }, next);
}

function sampleOne<T>(rng: Random<T>) {
  return rng.create()();
}

const boolean = Random_(() => drawOneOf8() < 4);
const uint32 = biguintWithInvalid(32);
const uint64 = biguintWithInvalid(64);
const byte = Random_(drawRandomByte);

const Random = Object.assign(Random_, {
  constant,
  field: fieldWithInvalid,
  fieldx2,
  uniformField,
  int,
  nat,
  fraction,
  boolean,
  byte,
  bytes,
  uniformBytes,
  array,
  record,
  tuple,
  map: Object.assign(map, { withInvalid: mapWithInvalid }),
  step,
  oneOf,
  dependent,
  apply,
  reject,
  uint32,
  uint64,
  biguint: biguintWithInvalid,
  bignat: bignatWithInvalid,
});

function constant<T>(t: T) {
  return Random_(() => t);
}

function bytes(size: number | Random<number>): Random<number[]> {
  return arrayValid(byte, size);
}

function uniformBytes(size: number | Random<number>): Random<number[]> {
  let size_ = typeof size === "number" ? constant(size) : size;
  return {
    create() {
      let nextSize = size_.create();
      return () => [...randomBytes(nextSize())];
    },
  };
}

function isGenerator<T>(rng: any): rng is Random<T> {
  return typeof rng === "object" && rng && "create" in rng;
}

function oneOf<Types extends readonly any[]>(
  ...values: {
    [K in keyof Types]:
      | Random<Types[K]>
      | RandomWithInvalid<Types[K]>
      | Types[K];
  }
): Random<Types[number]> {
  let gens = values.map(maybeConstant);
  let valid = {
    create() {
      let nexts = gens.map((rng) => rng.create());
      return () => {
        let i = drawUniformUint(values.length - 1);
        return nexts[i]();
      };
    },
  };
  let invalidGens = gens
    .filter((g) => g.invalid !== undefined)
    .map((g) => g.invalid!);
  let nInvalid = invalidGens.length;
  if (nInvalid === 0) return valid;
  let invalid = {
    create() {
      let nexts = invalidGens.map((rng) => rng.create());
      return () => {
        let i = drawUniformUint(nInvalid - 1);
        return nexts[i]();
      };
    },
  };
  return Object.assign(valid, { invalid });
}

/**
 * map a list of generators to a new generator, by specifying the transformation which maps samples
 * of the input generators to a sample of the result.
 */
function map<T extends readonly any[], S>(
  ...args: [...rngs: { [K in keyof T]: Random<T[K]> }, to: (...values: T) => S]
): Random<S> {
  const to = args.pop()! as (...values: T) => S;
  let rngs = args as { [K in keyof T]: Random<T[K]> };
  return {
    create() {
      let nexts = rngs.map((rng) => rng.create());
      return () => to(...(nexts.map((next) => next()) as any));
    },
  };
}
/**
 * dependent is like {@link map}, with the difference that the mapping contains a free variable
 * whose samples have to be provided as inputs separately. this is useful to create correlated generators, where
 * multiple generators are all dependent on the same extra variable which is sampled independently.
 *
 * dependent can be used in two different ways:
 * - as a function from a random generator of the free variable to a random generator of the result
 * - as a random generator whose samples are _functions_ from free variable to result: `Random<(arg: Free) => Result>`
 */
function dependent<T extends readonly any[], Result, Free>(
  ...args: [
    ...rngs: { [K in keyof T]: Random<T[K]> },
    to: (free: Free, values: T) => Result,
  ]
): Random<(arg: Free) => Result> & ((arg: Random<Free>) => Random<Result>) {
  const to = args.pop()! as (free: Free, values: T) => Result;
  let rngs = args as { [K in keyof T]: Random<T[K]> };
  let rng: Random<(arg: Free) => Result> = {
    create() {
      let nexts = rngs.map((rng) => rng.create());
      return () => (free) => to(free, nexts.map((next) => next()) as any);
    },
  };
  return Object.assign(function (free: Random<Free>): Random<Result> {
    return {
      create() {
        let freeNext = free.create();
        let nexts = rngs.map((rng) => rng.create());
        return () => to(freeNext(), nexts.map((next) => next()) as any);
      },
    };
  }, rng);
}

function step<T extends readonly any[], S>(
  ...args: [
    ...rngs: { [K in keyof T]: Random<T[K]> },
    step: (current: S, ...values: T) => S,
    initial: S,
  ]
): Random<S> {
  let initial = args.pop()! as S;
  const step = args.pop()! as (current: S, ...values: T) => S;
  let rngs = args as { [K in keyof T]: Random<T[K]> };
  return {
    create() {
      let nexts = rngs.map((rng) => rng.create());
      let next = initial;
      let current = initial;
      return () => {
        current = next;
        next = step(current, ...(nexts.map((next) => next()) as any as T));
        return current;
      };
    },
  };
}

function arrayValid<T>(
  element: Random<T>,
  size: number | Random<number>,
  { reset = false } = {}
): Random<T[]> {
  let size_ = typeof size === "number" ? constant(size) : size;
  return {
    create() {
      let nextSize = size_.create();
      let nextElement = element.create();
      return () => {
        let nextElement_ = reset ? element.create() : nextElement;
        return Array.from({ length: nextSize() }, nextElement_);
      };
    },
  };
}

function recordValid<T extends {}>(gens: {
  [K in keyof T]: Random<T[K]>;
}): Random<T> {
  return {
    create() {
      let keys = Object.keys(gens);
      let nexts = keys.map((key) => gens[key as keyof T].create());
      return () =>
        Object.fromEntries(keys.map((key, i) => [key, nexts[i]()])) as T;
    },
  };
}

function tupleValid<T extends Tuple<Random<any>>>(
  gens: T
): Random<{ [K in keyof T]: InferRandom<T[K]> }> {
  return {
    create() {
      let nexts = (gens as Random<any>[]).map((gen) => gen.create());
      return () => nexts.map((next) => next()) as any;
    },
  };
}

function reject<T>(rng: Random<T>, isRejected: (t: T) => boolean): Random<T> {
  return {
    create() {
      let next = rng.create();
      return () => {
        while (true) {
          let t = next();
          if (!isRejected(t)) return t;
        }
      };
    },
  };
}

type Action<S> = Random<(s: S) => S>;
function apply<S>(
  rng: Random<S>,
  howMany: number | Random<number>,
  ...actions: Action<S>[]
): Random<S> {
  let howMany_ = maybeConstant(howMany);
  let action = oneOf(...actions);
  return {
    create() {
      let next = rng.create();
      let nextSize = howMany_.create();
      let nextAction = action.create();
      return () => {
        let state = next();
        let size = nextSize();
        for (let i = 0; i < size; i++) {
          let action = nextAction();
          state = action(state);
        }
        return state;
      };
    },
  };
}

function maybeConstant<T>(c: T | Random<T>): Random<T> {
  return isGenerator(c) ? c : constant(c);
}

/**
 * uniform distribution over range [min, max]
 * with bias towards special values 0, 1, -1, 2, min, max
 */
function int(min: number, max: number): Random<number> {
  if (max < min) throw Error("max < min");
  // set of special numbers that will appear more often in tests
  let specialSet = new Set<number>();
  if (-1 >= min && -1 <= max) specialSet.add(-1);
  if (1 >= min && 1 <= max) specialSet.add(1);
  if (2 >= min && 2 <= max) specialSet.add(2);
  specialSet.add(min);
  specialSet.add(max);
  let special = [...specialSet];
  if (0 >= min && 0 <= max) special.unshift(0, 0);
  let nSpecial = special.length;
  return {
    create: () => () => {
      // 25% of test cases are special numbers
      if (drawOneOf8() < 3) {
        let i = drawUniformUint(nSpecial - 1);
        return special[i];
      }
      // the remaining follow a uniform distribution
      return min + drawUniformUint(max - min);
    },
  };
}

/**
 * log-uniform distribution over range [0, max]
 * with bias towards 0, 1, 2
 */
function bignat(max: bigint): Random<bigint> {
  if (max < 0n) throw Error("max < 0");
  if (max === 0n) return constant(0n);
  let bits = max.toString(2).length;
  let bitBits = bits.toString(2).length;
  // set of special numbers that will appear more often in tests
  let special = [0n, 0n, 1n];
  if (max > 1n) special.push(2n);
  let nSpecial = special.length;
  return {
    create: () => () => {
      // 25% of test cases are special numbers
      if (drawOneOf8() < 3) {
        let i = drawUniformUint(nSpecial - 1);
        return special[i];
      }
      // the remaining follow a log-uniform / cut off exponential distribution:
      // we sample a bit length (within a target range) and then a number with that length
      while (true) {
        // draw bit length from [1, 2**bitBits); reject if > bit length of max
        let bitLength = 1 + drawUniformUintBits(bitBits);
        if (bitLength > bits) continue;
        // draw number from [0, 2**bitLength); reject if > max
        let n = drawUniformBigUintBits(bitLength);
        if (n <= max) return n;
      }
    },
  };
}

/**
 * log-uniform distribution over range [0, max]
 * with bias towards 0, 1, 2
 */
function nat(max: number): Random<number> {
  return map(bignat(BigInt(max)), (n) => Number(n));
}

function fraction(fixedPrecision = 3) {
  let denom = 10 ** fixedPrecision;
  if (fixedPrecision < 1) throw Error("precision must be > 1");
  let next = () => (drawUniformUint(denom - 2) + 1) / denom;
  return { create: () => next };
}

let specialBytes = [0, 0, 0, 1, 1, 2, 255, 255];
/**
 * log-uniform distribution over range [0, 255]
 * with bias towards 0, 1, 2, 255
 */
function drawRandomByte() {
  // 25% of test cases are special numbers
  if (drawOneOf8() < 2) return specialBytes[drawOneOf8()];
  // the remaining follow log-uniform / cut off exponential distribution:
  // we sample a bit length from [1, 8] and then a number with that length
  let bitLength = 1 + drawOneOf8();
  return drawUniformUintBits(bitLength);
}

/**
 * log-uniform distribution over 2^n-bit range
 * with bias towards 0, 1, 2, max
 * outputs are bigints
 */
function biguint(bits: number): Random<bigint> {
  let max = (1n << BigInt(bits)) - 1n;
  let special = [0n, 0n, 0n, 1n, 1n, 2n, max, max];
  let bitsBits = Math.log2(bits);
  if (!Number.isInteger(bitsBits)) throw Error("bits must be a power of 2");
  return {
    create: () => () => {
      // 25% of test cases are special numbers
      if (drawOneOf8() < 2) return special[drawOneOf8()];
      // the remaining follow log-uniform / cut off exponential distribution:
      // we sample a bit length from [1, 8] and then a number with that length
      let bitLength = 1 + drawUniformUintBits(bitsBits);
      return drawUniformBigUintBits(bitLength);
    },
  };
}

/**
 * uniform positive integer in [0, max] drawn from secure randomness,
 */
function drawUniformUint(max: number) {
  if (max === 0) return 0;
  let bitLength = Math.floor(Math.log2(max)) + 1;
  while (true) {
    // values with same bit length can be too large by a factor of at most 2; those are rejected
    let n = drawUniformUintBits(bitLength);
    if (n <= max) return n;
  }
}

/**
 * uniform positive integer drawn from secure randomness,
 * given a target bit length
 */
function drawUniformUintBits(bitLength: number) {
  let byteLength = Math.ceil(bitLength / 8);
  // draw random bytes, zero the excess bits
  let bytes = randomBytes(byteLength);
  if (bitLength % 8 !== 0) {
    bytes[byteLength - 1] &= (1 << bitLength % 8) - 1;
  }
  // accumulate bytes to integer
  let n = 0;
  let bitPosition = 0;
  for (let byte of bytes) {
    n += byte << bitPosition;
    bitPosition += 8;
  }
  return n;
}

/**
 * uniform positive bigint drawn from secure randomness,
 * given a target bit length
 */
function drawUniformBigUintBits(bitLength: number) {
  let byteLength = Math.ceil(bitLength / 8);
  // draw random bytes, zero the excess bits
  let bytes = randomBytes(byteLength);
  if (bitLength % 8 !== 0) {
    bytes[byteLength - 1] &= (1 << bitLength % 8) - 1;
  }
  return bigintFromBytes(bytes);
}

/**
 * draw number between 0,..,7 using secure randomness
 */
function drawOneOf8() {
  return randomBytes(1)[0] >> 5;
}

// generators for invalid samples
// note: these only cover invalid samples with a _valid type_.
// for example, numbers that are out of range or base58 strings with invalid characters.
// what we don't cover is something like passing numbers where strings are expected

// convention is that invalid generators sit next to valid ones
// so you can use uint64.invalid, array(uint64, 10).invalid, etc

/**
 * we get invalid uints by sampling from a larger range plus negative numbers
 */
function biguintWithInvalid(bits: number): RandomWithInvalid<bigint> {
  let valid = biguint(bits);
  let max = 1n << BigInt(bits);
  let double = biguint(2 * bits);
  let negative = map(double, (uint) => -uint - 1n);
  let tooLarge = map(valid, (uint) => uint + max);
  let invalid = oneOf(negative, tooLarge);
  return Object.assign(valid, { invalid });
}

function bignatWithInvalid(max: bigint): RandomWithInvalid<bigint> {
  let valid = bignat(max);
  let double = bignat(2n * max);
  let negative = map(double, (uint) => -uint - 1n);
  let tooLarge = map(valid, (uint) => uint + max);
  let invalid = oneOf(negative, tooLarge);
  return Object.assign(valid, { invalid });
}

function fieldWithInvalid(p: bigint): RandomWithInvalid<bigint> {
  let { randomField } = randomGenerators(p);

  let uniformField = Random_(randomField);
  let specialField = oneOf(0n, 1n, 2n, p - 1n, p - 2n);

  let logp = log2(p);
  let roughLogSize = 1 << Math.ceil(Math.log2(logp) - 1);
  let uint = biguint(roughLogSize);

  let power2Like = map(
    Random.int(0, logp - 2),
    oneOf(-1n, 0n, 1n),
    (k, eps) => (1n << BigInt(k)) + eps
  );

  let field = oneOf<bigint[]>(uniformField, uint, specialField, power2Like);

  let tooLarge = map(field, (x) => x + p);
  let negative = map(field, (x) => -x - 1n);
  let invalid = oneOf(tooLarge, negative);
  return Object.assign(field, { invalid });
}

function fieldx2(p: bigint): Random<bigint> {
  let { randomField } = randomGenerators(2n * p);

  let uniformField = Random_(randomField);
  let specialField = oneOf(0n, 1n, 2n, p - 1n, p, p + 1n, 2n * p - 1n);

  let log2p = log2(2n * p);
  let roughLogSize = 1 << Math.ceil(Math.log2(log2p) - 1);
  let uint = biguint(roughLogSize);

  let power2Like = map(
    Random.int(0, log2p - 2),
    oneOf(-1n, 0n, 1n),
    (k, eps) => (1n << BigInt(k)) + eps
  );

  let field = oneOf<bigint[]>(uniformField, uint, specialField, power2Like);

  return field;
}

function uniformField(p: bigint): Random<bigint> {
  let { randomField } = randomGenerators(p);
  return Random_(randomField);
}

/**
 * invalid arrays are sampled by generating an array with exactly one invalid input (and any number of valid inputs);
 * (note: invalid arrays have the same length distribution as valid ones, except that they are never empty)
 */
function array<T>(
  element: Random<T>,
  size: number | Random<number>,
  options?: { reset?: boolean }
): Random<T[]> {
  let valid = arrayValid(element, size, options);
  if (element.invalid === undefined) return valid;
  let invalid = map(valid, element.invalid, (arr, invalid) => {
    if (arr.length === 0) return [invalid];
    let i = drawUniformUint(arr.length - 1);
    arr[i] = invalid;
    return arr;
  });
  return { ...valid, invalid };
}
/**
 * invalid records are similar to arrays: randomly choose one of the fields that have an invalid generator,
 * and set it to its invalid value
 */
function record<T extends {}>(gens: {
  [K in keyof T]: Random<T[K]>;
}): Random<T> {
  let valid = recordValid(gens);
  let invalidFields: [string & keyof T, Random<any>][] = [];
  for (let key in gens) {
    let invalid = gens[key].invalid;
    if (invalid !== undefined) {
      invalidFields.push([key, invalid]);
    }
  }
  let nInvalid = invalidFields.length;
  if (nInvalid === 0) return valid;
  let invalid = {
    create() {
      let next = valid.create();
      let invalidNexts = invalidFields.map(
        ([key, rng]) => [key, rng.create()] as const
      );
      return () => {
        let value = next();
        let i = drawUniformUint(nInvalid - 1);
        let [key, invalidNext] = invalidNexts[i];
        value[key] = invalidNext();
        return value;
      };
    },
  };
  return { ...valid, invalid };
}
/**
 * invalid tuples are like invalid records
 */
function tuple<T extends Tuple<Random<any>>>(
  gens: T
): Random<{ [K in keyof T]: InferRandom<T[K]> }> {
  let valid = tupleValid(gens);
  let invalidFields: [number & keyof T, Random<any>][] = [];
  (gens as Random<any>[]).forEach((gen, i) => {
    let invalid = gen.invalid;
    if (invalid !== undefined) {
      invalidFields.push([i, invalid]);
    }
  });
  let nInvalid = invalidFields.length;
  if (nInvalid === 0) return valid;
  let invalid = {
    create() {
      let next = valid.create();
      let invalidNexts = invalidFields.map(
        ([key, rng]) => [key, rng.create()] as const
      );
      return () => {
        let value = next();
        let i = drawUniformUint(nInvalid - 1);
        let [key, invalidNext] = invalidNexts[i];
        value[key] = invalidNext();
        return value;
      };
    },
  };
  return { ...valid, invalid };
}
/**
 * map assuming that invalid inputs can be mapped just like valid ones.
 * _one_ of the inputs is sampled as invalid
 */
function mapWithInvalid<T extends readonly any[], S>(
  ...args: [...rngs: { [K in keyof T]: Random<T[K]> }, to: (...values: T) => S]
): Random<S> {
  const to = args.pop()! as (...values: T) => S;
  let rngs = args as { [K in keyof T]: Random<T[K]> };
  let valid = map<T, S>(...rngs, to);
  let invalidInput = tuple(rngs as Tuple<Random<any>>).invalid;
  if (invalidInput === undefined) return valid;
  let invalid = {
    create() {
      let nextInput = invalidInput!.create();
      return () => to(...(nextInput() as any as T));
    },
  };
  return { ...valid, invalid };
}
