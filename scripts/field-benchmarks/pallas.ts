import { benchmark } from "./field-benchmark.js";
import { Pallas } from "../../src/concrete/pasta.js";

await benchmark(Pallas.Field, { onlyQuick: true });
