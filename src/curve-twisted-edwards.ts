/**
 * Operations on a twisted edwards curve, with a = -1
 *
 * -x^2 + y^2 = 1 + d*x^2*y^2
 *
 * The representation uses extended coordinates (X, Y, Z, Z) where
 *
 * x = X/Z
 * y = Y/Z
 * T = XY/Z
 */
import type * as W from "wasmati"; // for type names
import { MsmField } from "./field-msm.js";
import { mod, randomGenerators } from "./field-util.js";
import { assert, bigintToBits } from "./util.js";

export { createCurveTwistedEdwards, CurveTwistedEdwards };

type CurveTwistedEdwards = ReturnType<typeof createCurveTwistedEdwards>;

function createCurveTwistedEdwards(
  Field: MsmField,
  d: bigint,
  cofactor: bigint
) {
  const CurveBigint = createBigintTwistedEdwards(Field, d, cofactor);

  // write d to memory
  let [dPtr] = Field.local.getStablePointers(1);
  Field.writeBigint(dPtr, d);
  Field.toMontgomery(dPtr);

  // convert the cofactor to bits
  let cofactorBits = bigintToBits(cofactor);

  const { sizeField, square, multiply, add, copy, memoryBytes, p } = Field;

  // memory layout: x | y | z | t | isNonZero
  let size = 4 * sizeField + 4;

  function coords(pointer: number) {
    return [
      pointer,
      pointer + sizeField,
      pointer + 2 * sizeField,
      pointer + 3 * sizeField,
    ];
  }
  function copyPoint(target: number, source: number) {
    memoryBytes.copyWithin(target, source, source + size);
  }
  function isZero(pointer: number) {
    return !memoryBytes[pointer + 4 * sizeField];
  }
  function setNonZero(pointer: number) {
    memoryBytes[pointer + 4 * sizeField] = 1;
  }
  function setZero(pointer: number) {
    memoryBytes[pointer + 4 * sizeField] = 0;
  }

  /**
   * projective point addition with assignment, P1 += P2
   *
   * @param scratch
   * @param P1
   * @param P2
   */
  function addAssign(scratch: number[], P1: number, P2: number) {
    if (isZero(P1)) {
      copy(P1, P2);
      return;
    }
    if (isZero(P2)) return;
    setNonZero(P1);
    let [X1, Y1, Z1, T1] = coords(P1);
    let [X2, Y2, Z2, T2] = coords(P2);
    assert(false, "TODO");
  }

  /**
   * projective point doubling with assignment, P *= 2
   *
   * @param scratch
   * @param P
   */
  function doubleInPlace(scratch: number[], P: number) {
    if (isZero(P)) return;
    let [X1, Y1, Z1, T1] = coords(P);
    assert(false, "TODO");
  }

  function scale(
    scratch: number[],
    result: number,
    point: number,
    scalar: boolean[]
  ) {
    setZero(result);
    let n = scalar.length;
    for (let i = n - 1; i >= 0; i--) {
      if (scalar[i]) addAssign(scratch, result, point);
      if (i === 0) break;
      doubleInPlace(scratch, result);
    }
  }

  function toSubgroupInPlace(
    [tmp, _tmpy, _tmpz, _tmpt, _tmpInf, ...scratch]: number[],
    point: number
  ) {
    if (cofactor === 1n) return;
    copy(tmp, point);
    scale(scratch, point, tmp, cofactorBits);
  }

  let { randomFields } = randomGenerators(p);

  /**
   * sample random curve points
   *
   * expects the points as an array of pointers which can hold an affine point
   *
   * strategy: try random x coordinates until one of them fits the curve equation
   * if one doesn't work, increment until it does
   * just use the returned square root
   *
   * if the curve has a cofactor, we multiply by it to get points in the subgroup
   * (in that case, the cofactor multiplication is by far the dominant part)
   */
  function randomPoints(scratch: number[], points: number[]) {
    let n = points.length;
    let xs = randomFields(n);

    for (let i = 0; i < n; i++) {
      let x0 = xs[i];
      let x = points[i];

      // copy x into memory
      Field.writeBigint(x, x0);
      Field.toMontgomery(x);

      while (true) {
        // TODO
        assert(false, "TODO");
      }
      setNonZero(x);
    }

    if (cofactor !== 1n) {
      for (let i = 0; i < n; i++) {
        toSubgroupInPlace(scratch, points[i]);
      }
    }
    return points;
  }

  // note: this fails on zero
  function assertOnCurve([y2, y2_]: number[], p: number) {
    let [X, Y, Z, T] = coords(p);
    assert(false, "TODO");
  }

  function toBigint(point: number): BigintPoint {
    if (isZero(point)) return CurveBigint.zero;
    let [x, y, z, t] = coords(point);
    Field.fromMontgomery(x);
    Field.fromMontgomery(y);
    Field.fromMontgomery(z);
    Field.fromMontgomery(t);
    let pointBigint = {
      X: Field.readBigint(x),
      Y: Field.readBigint(y),
      Z: Field.readBigint(z),
      T: Field.readBigint(t),
    };
    Field.toMontgomery(x);
    Field.toMontgomery(y);
    Field.toMontgomery(z);
    Field.toMontgomery(t);
    return pointBigint;
  }

  function writeBigint(point: number, P: BigintPoint) {
    if (CurveBigint.isZero(P)) {
      setZero(point);
      return;
    }
    let { X, Y, Z, T } = P;
    let [xPtr, yPtr, zPtr, tPtr] = coords(point);
    Field.writeBigint(xPtr, X);
    Field.writeBigint(yPtr, Y);
    Field.writeBigint(zPtr, Z);
    Field.writeBigint(tPtr, T);
    Field.toMontgomery(xPtr);
    Field.toMontgomery(yPtr);
    Field.toMontgomery(zPtr);
    Field.toMontgomery(tPtr);
    setNonZero(point);
  }

  return {
    addAssign,
    doubleInPlace,
    size,
    scale,
    toSubgroupInPlace,
    assertOnCurve,
    isZero,
    setZero,
    setNonZero,
    copyPoint,
    toBigint,
    writeBigint,
    randomPoints,
  };
}

type BigintPoint = { X: bigint; Y: bigint; Z: bigint; T: bigint };

function createBigintTwistedEdwards(
  Field: MsmField,
  d: bigint,
  cofactor: bigint
) {
  let k = 2n * d;
  let { p } = Field;

  const zero = { X: 0n, Y: 1n, Z: 1n, T: 0n } satisfies BigintPoint;

  /**
   * Addition, P1 + P2
   *
   * Strongly unified
   */
  function add(P1: BigintPoint, P2: BigintPoint): BigintPoint {
    let { X: X1, Y: Y1, Z: Z1, T: T1 } = P1;
    let { X: X2, Y: Y2, Z: Z2, T: T2 } = P2;
    // http://hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html#addition-add-2008-hwcd-3
    // Assumptions: k=2*d.

    // A = (Y1-X1)*(Y2-X2)
    let A = mod((Y1 - X1) * (Y2 - X2), p);
    // B = (Y1+X1)*(Y2+X2)
    let B = mod((Y1 + X1) * (Y2 + X2), p);
    // C = T1*k*T2
    let C = mod(T1 * T2, p);
    C = mod(C * k, p);
    // D = Z1*2*Z2
    let D = mod(2n * Z1 * Z2, p);
    // E = B-A
    let E = ffSub(B, A, p);
    // F = D-C
    let F = ffSub(D, C, p);
    // G = D+C
    let G = ffAdd(D, C, p);
    // H = B+A
    let H = ffAdd(B, A, p);
    // X3 = E*F
    let X3 = mod(E * F, p);
    // Y3 = G*H
    let Y3 = mod(G * H, p);
    // T3 = E*H
    let T3 = mod(E * H, p);
    // Z3 = F*G
    let Z3 = mod(F * G, p);

    return { X: X3, Y: Y3, Z: Z3, T: T3 };
  }

  /**
   * Doubling, 2*P
   *
   * Strongly unified
   */
  function double(P: BigintPoint) {
    return add(P, P);
  }

  /**
   * Negation, -P
   */
  function negate(P: BigintPoint): BigintPoint {
    return { X: ffNegate(P.X, p), Y: P.Y, Z: P.Z, T: ffNegate(P.T, p) };
  }

  function isEqual(P1: BigintPoint, P2: BigintPoint, p: bigint) {
    return (
      // protect against invalid points with z=0
      mod(P1.Z, p) !== 0n &&
      mod(P2.Z, p) !== 0n &&
      // multiply out with Z
      mod(P1.X * P2.Z - P2.X * P1.Z, p) === 0n &&
      mod(P1.Y * P2.Z - P2.Y * P1.Z, p) === 0n &&
      // redundant for valid points, but this function should work if one input is invalid
      mod(P1.T * P2.Z - P2.T * P1.Z, p) === 0n
    );
  }

  function isZero({ X, Y, Z, T }: BigintPoint): boolean {
    return (
      mod(Z, p) !== 0n &&
      mod(X, p) === 0n &&
      mod(T, p) === 0n &&
      mod(Y - Z, p) === 0n
    );
  }

  /**
   * Scalar multiplication, s*P
   */
  function scale(s: bigint, P: BigintPoint): BigintPoint {
    let Q = zero;
    let bits = bigintToBits(s);
    for (let i = bits.length - 1; i >= 0; i--) {
      Q = double(Q);
      if (bits[i]) Q = add(Q, P);
    }
    return Q;
  }

  /**
   * Project a point to the correct subgroup
   */
  function toSubgroup(P: BigintPoint): BigintPoint {
    if (cofactor === 1n) return P;
    return scale(cofactor, P);
  }

  /**
   * Check if a point is on the curve
   *
   * In projective coordinates, the curve equation is
   *
   * -X^2 Z^2 + Y^2 Z^2 = Z^4 + d X^2 Y^2
   *
   * or, after dividing by Z^2 and using T = XY/Z,
   *
   * -X^2 + Y^2 = Z^2 + d T^2
   */
  function isOnCurve(P: BigintPoint): boolean {
    let { X, Y, T, Z } = P;
    // validity of Z
    if (mod(Z, p) === 0n) return false;
    // validity of T
    if (mod(T * Z - X * Y, p) !== 0n) return false;
    // curve equation
    return mod(-X * X + Y * Y - Z * Z - d * mod(T * T, p), p) === 0n;
  }

  return {
    zero,
    add,
    double,
    negate,
    scale,
    toSubgroup,
    isOnCurve,
    isEqual,
    isZero,
  };
}

function ffAdd(x: bigint, y: bigint, p: bigint) {
  let z = x + y;
  return z >= p ? z - p : z;
}

function ffSub(x: bigint, y: bigint, p: bigint) {
  let z = x - y;
  return z < 0 ? z + p : z;
}

function ffNegate(x: bigint, p: bigint) {
  return x === 0n ? 0n : p - x;
}
