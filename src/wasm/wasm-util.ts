import {
  Const,
  Dependency,
  Local,
  br_if,
  data,
  global,
  i32,
  local,
  loop,
} from "wasmati";

export { ImplicitMemory, forLoop, forLoop1, forLoop4 };

// collect memory + data segments together
class ImplicitMemory {
  memory: Dependency.AnyMemory;
  dataOffset: number = 0;
  dataSegments: Dependency.Data[] = [];

  constructor(memory: Dependency.AnyMemory) {
    this.memory = memory;
  }

  data(bytes: Uint8Array | number[]) {
    let offset = Const.i32(this.dataOffset);
    let dataSegment = data({ offset, memory: this.memory }, bytes);
    this.dataSegments.push(dataSegment);
    this.dataOffset += bytes.length;
    return global(offset);
  }

  dataToOffset(bytes: Uint8Array | number[]) {
    let offset = this.dataOffset;
    let dataSegment = data(
      { offset: Const.i32(offset), memory: this.memory },
      bytes
    );
    this.dataSegments.push(dataSegment);
    this.dataOffset += bytes.length;
    return offset;
  }

  getExports() {
    return {
      memory: this.memory,
      dataOffset: global(Const.i32(this.dataOffset)),
    };
  }
}

// helper
function forLoop(
  {
    incr,
    i,
    start,
    end,
  }: {
    incr: number;
    i: Local<i32>;
    start: number | Local<i32>;
    end: number | Local<i32>;
  },
  callback: () => void
) {
  if (typeof start === "number") i32.const(start);
  else local.get(start);
  local.set(i);
  loop({}, () => {
    callback();
    // i += incr
    i32.add(i, incr);
    local.tee(i);
    // (...) !== end
    if (typeof end === "number") {
      i32.const(incr * end);
    } else {
      if (incr === 1) {
        local.get(end);
      } else {
        i32.mul(incr, end);
      }
    }
    i32.ne();
    // if (...) continue
    br_if(0);
  });
}

function forLoop4(
  i: Local<i32>,
  start: number | Local<i32>,
  end: number | Local<i32>,
  callback: () => void
) {
  forLoop({ incr: 4, i, start, end }, callback);
}
/**
 *
 * @param i
 * @param start
 * @param end not inclusive
 * @param callback
 */
function forLoop1(
  i: Local<i32>,
  start: number | Local<i32>,
  end: number | Local<i32>,
  callback: () => void
) {
  forLoop({ incr: 1, i, start, end }, callback);
}
