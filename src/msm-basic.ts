/**
 * MSM implementation that only relies on the most basic curve interface: add, double, zero.
 *
 * We use this for Twisted Edwards curves which have neither endomorphisms nor a cheap batched addition algorithm.
 * Also works well for Weierstrass curves in projective coordinates.
 */
import type { MsmField } from "./field-msm.js";
import type { MemorySection } from "./wasm/memory-helpers.js";
import { createLog, splitBuckets, windowSize } from "./msm-common.js";
import { log2 } from "./util.js";
import { THREADS, barrier, isMain, range, thread } from "./threads/threads.js";
import { broadcastFromMain } from "./threads/global-pool.js";

export { createMsmBasic, msmBasic };

type MsmInputCurve = {
  Field: MsmField;
  Scalar: {
    sizeInBits: number;
    sizeField: number;
    extractBitSlice: (ptr: number, start: number, length: number) => number;
  };
  Curve: {
    size: number;
    setZero: (P: number) => void;
    add: (scratch: number[], P3: number, P1: number, P2: number) => void;
    double: (scratch: number[], R: number, P: number) => void;
    addMixed: (scratch: number[], P3: number, P1: number, P2: number) => void;
    subMixed: (scratch: number[], P3: number, P1: number, P2: number) => void;
    copy: (target: number, source: number) => void;
  };
};

function createMsmBasic(inputs: MsmInputCurve) {
  return async function (
    scalarPtr: number,
    pointPtr: number,
    N: number,
    options?: { c?: number }
  ) {
    return await msmBasic(inputs, scalarPtr, pointPtr, N, options);
  };
}

async function msmBasic(
  inputs: MsmInputCurve,
  scalarPtr: number,
  pointPtr: number,
  N: number,
  { c: c_ }: { c?: number } = {}
) {
  let { tic, toc, log, getLog } = createLog(isMain());
  tic("msm total");
  let { Field, Scalar, Curve } = inputs;
  let b = Scalar.sizeInBits;
  let n = log2(N);
  let c = c_ ?? windowSize(Field, n);
  let K = Math.ceil((b + 1) / c);
  let L = 1 << (c - 1);
  let params = { N, K, L, c, b };
  log({ n, K, c });

  let { add, addMixed, subMixed, double, setZero, size } = Curve;
  let { sizeField: sizeScalar } = Scalar;
  let result = Field.global.getPointer(Curve.size);

  using _l = Field.local.atCurrentOffset;
  let scratch = Field.local.getPointers(40);

  tic("points to bucket map");
  // TODO parallelize points to bucket sorting
  let pointsToBucket = await broadcastFromMain("pointsToBucket", () => {
    // K x N -> l in [0, L-1]
    let pointsToBucket: Uint32Array[] = Array(K);
    for (let k = 0; k < K; k++) {
      pointsToBucket[k] = new Uint32Array(new SharedArrayBuffer(4 * N));
    }
    return pointsToBucket;
  });

  let [i, iend] = range(N);

  for (let si = scalarPtr + i * sizeScalar; i < iend; i++, si += sizeScalar) {
    for (let k = 0, carry = 0; k < K; k++) {
      let l = Scalar.extractBitSlice(si, k * c, c) + carry;
      if (l > L) {
        l = 2 * L - l;
        carry = 1;
      } else {
        carry = 0;
      }
      pointsToBucket[k][i] = l | (carry << 31);
    }
  }
  await barrier();
  toc();

  tic("compute work split");
  let { chunksPerThread, chunkSumsPerPartition } = splitBuckets(
    inputs,
    params,
    THREADS
  );
  toc();

  tic("accumulate and reduce");
  for (let { j, k, lstart, length } of chunksPerThread[thread]) {
    tic();
    using _l = Field.local.atCurrentOffset;
    let buckets = Uint32Array.from(Field.local.getPointers(length, Curve.size));
    for (let l = 0; l < length; l++) setZero(buckets[l]);

    // accumulation
    let nAdditions = 0;

    for (let i = 0, pi = pointPtr; i < N; i++, pi += size) {
      let l = pointsToBucket[k][i];
      let carry = l >>> 31;
      l &= 0x7f_ff_ff_ff;
      l -= lstart;
      if (l < 0 || l >= length) continue;
      nAdditions++;
      let bucket = buckets[l];
      if (carry === 1) {
        subMixed(scratch, bucket, bucket, pi);
      } else {
        addMixed(scratch, bucket, bucket, pi);
      }
    }
    let t = toc();
    log(
      `accumulate (chunk ${k}.${j}): ${t.toFixed(0)}ms, ${nAdditions} adds, ${(
        (t / nAdditions) *
        1e6
      ).toFixed(1)}ns / add`
    );

    tic(`reduce (chunk ${k}.${j})`);
    // reduce into sum
    let sum = chunkSumsPerPartition[k][j];
    reduceBucketsChunk(inputs, sum, buckets, lstart);
    toc();
  }
  toc();

  tic("barrier");
  await barrier();
  toc();
  if (!isMain()) return { result: 0, log: [] };

  // aggregate
  tic("final summation");
  let partitionSums: number[] = Array(K);

  for (let k = 0; k < K; k++) {
    let chunkSums = chunkSumsPerPartition[k];
    let partitionSum = chunkSums[0];
    for (let j = 1, n = chunkSums.length; j < n; j++) {
      Curve.add(scratch, partitionSum, partitionSum, chunkSums[j]);
    }
    partitionSums[k] = partitionSum;
  }

  // final summation
  Curve.copy(result, partitionSums[K - 1]);
  for (let k = K - 2; k >= 0; k--) {
    for (let i = 0; i < c; i++) {
      double(scratch, result, result);
    }
    add(scratch, result, result, partitionSums[k]);
  }
  toc();
  toc();
  log(Field.global.printMaxSizeUsed());
  log(Field.local.printMaxSizeUsed());
  return { result, log: getLog() };
}

/**
 * computes a "column" of the bucket reduction sum:
 *
 * column := sum_{l=lstart..lend} l * buckets[l - lstart]
 *
 * defining L = lend - lstart, we can write the sum as
 *
 * = sum_{l=0..L} (lstart + l) * buckets[l]
 * = (sum_{l=0..L} (l + 1) * buckets[l]) + (lstart - 1) * (sum_{l=0..L} buckets[l])
 * =: triangle + (lstart - 1) * row
 *
 * - triangle and row are computed together in 2L additions
 * - (lstart - 1) * row is computed with double-and-add with O(log(lstart)) effort which is usually negligible
 */
function reduceBucketsChunk(
  inputs: { Field: { local: MemorySection }; Curve: MsmInputCurve["Curve"] },
  column: number,
  buckets: Uint32Array,
  lstart: number
) {
  let { Field, Curve } = inputs;
  let L = buckets.length;
  let { add, double, setZero } = Curve;

  using _ = Field.local.atCurrentOffset;
  let scratch = Field.local.getPointers(20);
  let [triangle, row] = Field.local.getPointers(2, Curve.size);
  setZero(triangle);
  setZero(row);

  // compute triangle and row
  for (let l = L - 1; l >= 0; l--) {
    add(scratch, row, row, buckets[l]);
    add(scratch, triangle, triangle, row);
  }

  // triangle += (lstart - 1) * row
  lstart--;
  while (true) {
    if (lstart & 1) add(scratch, triangle, triangle, row);
    if ((lstart >>= 1) === 0) break;
    double(scratch, row, row);
  }

  Curve.copy(column, triangle);
}
