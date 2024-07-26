import { assert } from "./util.js";
import type { MemorySection } from "./wasm/memory-helpers.js";

export { windowSize, windowSizeAffine, splitBuckets, Chunk, createLog };

const REMOVE_ALL_LOGS = false;

function windowSize(Field: { sizeInBits: number }, n: number) {
  return (
    windowSizeTable[Field.sizeInBits > 260 ? "large" : "small"][n] ??
    Math.max(n - 1, 1)
  );
}

function windowSizeAffine(Field: { sizeInBits: number }, n: number) {
  return (
    windowSizeTable[Field.sizeInBits > 260 ? "large-affine" : "small-affine"][
      n
    ] ?? Math.max(n - 1, 1)
  );
}

/**
 * tables of the form `n: c`, which has msm window sizes for different n.
 * n is the log-size of scalar and point inputs.
 *
 * table was optimized with 16 threads on my laptop, with two different types of curves:
 * - 'large' (~384 bit base field)
 * - 'small' (~256 bit base field)
 *
 * @param c window size
 */
const windowSizeTable: {
  [k in "large" | "large-affine" | "small" | "small-affine"]: Record<
    number,
    number | undefined
  >;
} = {
  // TODO
  large: {},
  "large-affine": {
    14: 13,
    15: 14,
    16: 14,
    17: 14,
    18: 14,
    19: 18,
    20: 18,
  },
  // TODO
  small: {
    16: 14,
  },
  "small-affine": {
    16: 12,
  },
};

/**
 * Represents a chunk of buckets, to be processed by a single thread.
 */
type Chunk = {
  /**
   * partition index
   */
  k: number;
  /**
   * index of this chunk within the partition
   */
  j: number;
  /**
   * index of the first bucket in this chunk, among all buckets of this partition
   */
  lstart: number;
  /**
   * number of buckets in this chunk
   */
  length: number;
};

/**
 * Split buckets among threads
 *
 * Note: this takes into account
 * - bucket sizes not being distributed evenly in the final partition
 * - peculiarity in the bucket distribution when the signed digits algorithm is used
 */
function splitBuckets(
  {
    Field,
    Curve,
  }: {
    Field: { global: MemorySection };
    Curve: { size: number };
  },
  params: {
    b: number;
    c: number;
    K: number;
    L: number;
  },
  THREADS: number
) {
  let { b, c, K, L } = params;
  // Ll = (upper bound on) number of non-empty buckets in final partition
  // it's crucial for correctness that this is precise, because we ignore the other buckets
  // there are 2^c0 buckets for c0 bits, minus 1 for the 0 bucket, PLUS 1 because of the carry in the signed digit algorithm
  let overlapBits = b % c; // number of bits that can be set in the highest window
  let Ll = 1 << overlapBits;
  let wl = overlapBits === 0 ? L / Ll / 2 ** 5 : L / Ll; // reduced weight in the 1 bit overflow case

  let totalWork = (K - 1) * L + Ll * wl; // = K*L except in the overflow case
  let workPerThread = Math.ceil(totalWork / THREADS);

  let chunksPerThread: Chunk[][] = [];
  let nChunksPerPartition: number[] = Array(K);

  for (let thread = 0; thread < THREADS; thread++) {
    chunksPerThread[thread] = [];
  }

  let thread = 0;
  let remainingForCurrentThread = workPerThread;

  for (let k = 0; k < K - 1; k++) {
    let j = 0;
    let remainingInThisPartition = L;
    let lstart = 1;
    while (remainingInThisPartition > 0) {
      let length = Math.min(
        remainingInThisPartition,
        remainingForCurrentThread
      );
      chunksPerThread[thread].push({ k, j, lstart, length });
      j++;
      remainingInThisPartition -= length;
      remainingForCurrentThread -= length;
      lstart += length;
      if (remainingForCurrentThread <= 0) {
        thread++;
        remainingForCurrentThread = workPerThread;
      }
    }
    nChunksPerPartition[k] = j;
  }
  {
    let k = K - 1;
    let j = 0;
    let remainingWorkInThisPartition = Ll * wl;
    let remainingBucketsInThisPartition = L;
    let lstart = 1;
    while (remainingWorkInThisPartition > 0) {
      let length = Math.min(
        Math.ceil(remainingForCurrentThread / wl),
        remainingBucketsInThisPartition
      );
      chunksPerThread[thread].push({ k, j, lstart, length });
      j++;
      remainingWorkInThisPartition -= wl * length;
      remainingBucketsInThisPartition -= length;
      remainingForCurrentThread -= wl * length;
      lstart += length;
      if (remainingWorkInThisPartition <= 0) {
        assert(lstart > Ll);
      }
      if (remainingForCurrentThread <= 0) {
        thread++;
        remainingForCurrentThread = workPerThread;
      }
    }
    assert(thread <= THREADS);

    nChunksPerPartition[k] = j;
  }

  // allocate space for different threads' contribution to each partition
  // K x (#chunks in this partition)
  let chunkSumsPerPartition: Uint32Array[] = Array(K);
  for (let k = 0; k < K; k++) {
    let nChunks = nChunksPerPartition[k];
    // note: each thread must compute these pointers independently in the same way
    chunkSumsPerPartition[k] = Uint32Array.from(
      Field.global.getPointers(nChunks, Curve.size)
    );
  }

  return { chunksPerThread, chunkSumsPerPartition };
}

// timing/logging helpers

function createLog(isActive: boolean) {
  let timingStack: [string | undefined, number][] = [];
  let deferredLog: any[][] = [];

  if (REMOVE_ALL_LOGS)
    return {
      printLog: () => {},
      log: () => {},
      tic: () => {},
      toc: () => 0,
      getLog: () => [],
    };

  function printLog() {
    deferredLog.forEach((log) => isActive && console.log(...log));
    deferredLog = [];
  }

  function getLog() {
    return deferredLog;
  }

  function log(...args: any[]) {
    deferredLog.push(args);
  }

  function tic(label?: string) {
    timingStack.push([label, performance.now()]);
  }

  function toc() {
    let [label, start] = timingStack.pop()!;
    let time = performance.now() - start;
    if (label !== undefined) log(`${label}... ${time.toFixed(1)}ms`);
    return time;
  }

  return { printLog, getLog, log, tic, toc };
}
