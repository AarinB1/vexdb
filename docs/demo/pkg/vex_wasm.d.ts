/* tslint:disable */
/* eslint-disable */

/**
 * A vex index living in WebAssembly memory.
 */
export class VexIndex {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Insert a vector with an optional JSON payload string.
     */
    add(id: bigint, vector: Float32Array, payload_json?: string | null): void;
    dim(): number;
    /**
     * Load an index from `.vex` snapshot bytes (e.g. a `fetch`ed file).
     */
    static fromSnapshot(bytes: Uint8Array): VexIndex;
    isEmpty(): boolean;
    len(): number;
    /**
     * Create an empty HNSW index (metric: "cosine" | "l2" | "dot").
     */
    static newHnsw(dim: number, metric: string, m: number, ef_construction: number): VexIndex;
    /**
     * The stored payload for an id.
     */
    payloadOf(id: bigint): any;
    /**
     * Top-k search. `ef` tunes the HNSW beam width (0 = index default);
     * `filter_json` is the same filter language the HTTP API accepts.
     * Returns `[{id, distance, payload?}, ...]` sorted ascending.
     */
    search(query: Float32Array, k: number, ef: number, filter_json?: string | null): any;
    /**
     * `{index, metric, dim, count}`.
     */
    stats(): any;
    /**
     * Serialize the index back to `.vex` bytes (e.g. to download an index
     * built in the browser).
     */
    toSnapshot(): Uint8Array;
    /**
     * The stored vector for an id — search with it for "more like this".
     */
    vectorOf(id: bigint): Float32Array | undefined;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_vexindex_free: (a: number, b: number) => void;
    readonly vexindex_add: (a: number, b: bigint, c: number, d: number, e: number, f: number) => [number, number];
    readonly vexindex_dim: (a: number) => number;
    readonly vexindex_fromSnapshot: (a: number, b: number) => [number, number, number];
    readonly vexindex_isEmpty: (a: number) => number;
    readonly vexindex_len: (a: number) => number;
    readonly vexindex_newHnsw: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly vexindex_payloadOf: (a: number, b: bigint) => [number, number, number];
    readonly vexindex_search: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number];
    readonly vexindex_stats: (a: number) => [number, number, number];
    readonly vexindex_toSnapshot: (a: number) => [number, number, number, number];
    readonly vexindex_vectorOf: (a: number, b: bigint) => [number, number];
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
