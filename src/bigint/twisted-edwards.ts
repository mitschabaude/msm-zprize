import { assert, bigintToBits } from "../util.js";
import { computeEndoConstants } from "./curve-endomorphism.js";
import { createField } from "./field.js";

export { createCurveTwistedEdwards, BigintPoint, CurveParams };

type BigintPoint = { X: bigint; Y: bigint; Z: bigint; T: bigint };

type CurveParams = {
  modulus: bigint;
  order: bigint;
  cofactor: bigint;
  d: bigint;
  generator: { x: bigint; y: bigint };
};

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
function createCurveTwistedEdwards(params: CurveParams) {
  let { modulus: p, order: q, cofactor, d, generator } = params;
  let k = 2n * d;
  const Fp = createField(p);
  const Fq = createField(q);

  const zero = { X: 0n, Y: 1n, Z: 1n, T: 0n } satisfies BigintPoint;

  function fromAffine({ x, y }: { x: bigint; y: bigint }): BigintPoint {
    return { X: x, Y: y, Z: 1n, T: Fp.mul(x, y) };
  }
  function toAffine({ X, Y, Z }: BigintPoint): { x: bigint; y: bigint } {
    assert(!Fp.equal(Z, 0n), "Not an affine point");
    return { x: Fp.mul(X, Fp.inv(Z)), y: Fp.mul(Y, Fp.inv(Z)) };
  }

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
    let A = Fp.mul(Y1 - X1, Y2 - X2);
    // B = (Y1+X1)*(Y2+X2)
    let B = Fp.mul(Y1 + X1, Y2 + X2);
    // C = T1*k*T2
    let C = Fp.mul(T1, T2);
    C = Fp.mul(C, k);
    // D = Z1*2*Z2
    let D = Fp.mul(2n * Z1, Z2);
    // E = B-A
    let E = Fp.sub(B, A);
    // F = D-C
    let F = Fp.sub(D, C);
    // G = D+C
    let G = Fp.add(D, C);
    // H = B+A
    let H = Fp.add(B, A);
    // X3 = E*F
    let X3 = Fp.mul(E, F);
    // Y3 = G*H
    let Y3 = Fp.mul(G, H);
    // T3 = E*H
    let T3 = Fp.mul(E, H);
    // Z3 = F*G
    let Z3 = Fp.mul(F, G);

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
    return { X: Fp.neg(P.X), Y: P.Y, Z: P.Z, T: Fp.neg(P.T) };
  }

  function isEqual(P1: BigintPoint, P2: BigintPoint) {
    return (
      // protect against invalid points with z=0
      !Fp.equal(P1.Z, 0n) &&
      !Fp.equal(P2.Z, 0n) &&
      // multiply out with Z
      Fp.equal(P1.X * P2.Z, P2.X * P1.Z) &&
      Fp.equal(P1.Y * P2.Z, P2.Y * P1.Z) &&
      // redundant for valid points, but this function should work if one input is invalid
      Fp.equal(P1.T * P2.Z, P2.T * P1.Z)
    );
  }

  function isZero({ X, Y, Z, T }: BigintPoint): boolean {
    return (
      !Fp.equal(Z, 0n) && Fp.equal(X, 0n) && Fp.equal(T, 0n) && Fp.equal(Y, Z)
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
    if (Fp.equal(Z, 0n)) return false;
    // validity of T
    if (!Fp.equal(T * Z, X * Y)) return false;
    // curve equation
    return Fp.equal(-X * X + Y * Y, Z * Z + d * Fp.mul(T, T));
  }

  function random(): BigintPoint {
    // random x
    let x = Fp.random();
    let y: bigint | undefined;

    while (y === undefined) {
      x = Fp.add(x, 1n);
      // solve -x^2 + y^2 = 1 + d x^2 y^2 for y
      // => y^2 = (1 + x^2) / (1 - d x^2)
      let y2 = Fp.mul(1n + Fp.mul(x, x), Fp.inv(1n - Fp.mul(d, Fp.mul(x, x))));
      y = Fp.sqrt(y2);
    }

    let P = { X: x, Y: y, Z: 1n, T: Fp.mul(x, y) };
    return toSubgroup(P);
  }

  return {
    Field: Fp,
    Scalar: Fq,
    ...params,

    zero,
    one: fromAffine(generator),
    add,
    double,
    negate,
    scale,
    toSubgroup,
    isOnCurve,
    isEqual,
    isZero,
    random,
    fromAffine,
    toAffine,
  };
}
