// Real, reproducible verification for build_wasm/csa.js -- run this
// after any change to src/wasm_shim.cpp or bench/build_wasm.sh, before
// trusting the module enough to wire it into docs/index.html or
// demo/csa_demo.html. Cross-checks the newest exports (KdTree3i's
// range_query/k_nearest) against a pure-JS brute-force reference, and
// re-confirms the pre-existing exports (general compress/decompress,
// geo3d compress/decompress) still round-trip correctly -- catching a
// regression in either direction, not just "the new thing works."
//
// Usage: `bash bench/build_wasm.sh && node bench/verify_wasm.mjs`
import CsaModule from '../build_wasm/csa.js';

const Module = await CsaModule();
let failed = false;
const check = (label, ok) => { console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}`); if (!ok) failed = true; };

function mallocBytes(u8) {
  const ptr = Module._malloc(u8.length);
  Module.HEAPU8.set(u8, ptr);
  return ptr;
}
function malloc32(arr) {
  const ptr = Module._malloc(arr.length * 4);
  Module.HEAP32.set(arr, ptr / 4);
  return ptr;
}

const compress = Module.cwrap('csa_wasm_compress', 'number', ['number', 'number', 'number']);
const decompress = Module.cwrap('csa_wasm_decompress', 'number', ['number', 'number', 'number']);
const geo3dCompress = Module.cwrap('csa_wasm_compress_geo3d', 'number', ['number', 'number', 'number']);
const geo3dDecompress = Module.cwrap('csa_wasm_decompress_geo3d', 'number', ['number', 'number', 'number']);
const kdtreeBuild = Module.cwrap('csa_wasm_kdtree_build', null, ['number', 'number']);
const kdtreeRange = Module.cwrap('csa_wasm_kdtree_range_query', 'number', ['number', 'number', 'number']);
const kdtreeKnn = Module.cwrap('csa_wasm_kdtree_knn', 'number', ['number', 'number', 'number']);
const wasmFree = Module.cwrap('csa_wasm_free', null, ['number']);

// --- pre-existing: general compress/decompress round-trip ---
{
  const text = new TextEncoder().encode('the quick brown fox jumps over the lazy dog, '.repeat(200));
  const inPtr = mallocBytes(text);
  const outSizePtr = Module._malloc(8);
  const compPtr = compress(inPtr, text.length, outSizePtr);
  const compBytes = Module.HEAPU8.slice(compPtr, compPtr + Module.HEAPU32[outSizePtr / 4]);
  wasmFree(compPtr);

  const inPtr2 = mallocBytes(compBytes);
  const outSizePtr2 = Module._malloc(8);
  const decPtr = decompress(inPtr2, compBytes.length, outSizePtr2);
  const decBytes = Module.HEAPU8.slice(decPtr, decPtr + Module.HEAPU32[outSizePtr2 / 4]);
  wasmFree(decPtr);

  check(`general compress/decompress (${text.length} -> ${compBytes.length} bytes)`,
        new TextDecoder().decode(decBytes) === new TextDecoder().decode(text));
}

// --- pre-existing: geo3d compress/decompress round-trip ---
{
  const raw = [];
  for (let i = 0; i < 500; i++) raw.push(Math.floor(Math.sin(i) * 100000), Math.floor(Math.cos(i) * 100000), i * 10);
  const flat = new Int32Array(raw);
  const flatPtr = malloc32(flat);
  const outSizePtr = Module._malloc(8);
  const compPtr = geo3dCompress(flatPtr, flat.length / 3, outSizePtr);
  const compSize = Module.HEAPU32[outSizePtr / 4];
  const compBytes = Module.HEAPU8.slice(compPtr, compPtr + compSize);
  wasmFree(compPtr);

  const inPtr = mallocBytes(compBytes);
  const outCountPtr = Module._malloc(8);
  const decPtr = geo3dDecompress(inPtr, compBytes.length, outCountPtr);
  const decCount = Module.HEAPU32[outCountPtr / 4];
  const decFlat = Module.HEAP32.slice(decPtr / 4, decPtr / 4 + decCount * 3);
  wasmFree(decPtr);

  let match = decFlat.length === flat.length;
  for (let i = 0; i < flat.length && match; i++) if (decFlat[i] !== flat[i]) match = false;
  check(`geo3d compress/decompress (${raw.length / 3} points, ${flat.length * 4} -> ${compSize} bytes)`, match);
}

// --- new: KdTree3i range_query and k_nearest, against a JS brute-force reference ---
{
  const pts = [];
  for (let i = 0; i < 2000; i++)
    pts.push([Math.floor(Math.sin(i * 0.37) * 100000), Math.floor(Math.cos(i * 0.53) * 100000), Math.floor((i * 977) % 200000 - 100000)]);
  const flat = new Int32Array(pts.length * 3);
  for (let i = 0; i < pts.length; i++) { flat[i*3]=pts[i][0]; flat[i*3+1]=pts[i][1]; flat[i*3+2]=pts[i][2]; }
  const flatPtr = malloc32(flat);
  kdtreeBuild(flatPtr, pts.length);
  Module._free(flatPtr);

  const lo = [-50000, -50000, -50000], hi = [50000, 50000, 50000];
  const loPtr = malloc32(new Int32Array(lo)), hiPtr = malloc32(new Int32Array(hi));
  const outCountPtr = Module._malloc(8);
  const resPtr = kdtreeRange(loPtr, hiPtr, outCountPtr);
  const indices = Array.from(Module.HEAPU32.subarray(resPtr / 4, resPtr / 4 + Module.HEAPU32[outCountPtr / 4])).sort((a,b)=>a-b);
  wasmFree(resPtr); Module._free(loPtr); Module._free(hiPtr); Module._free(outCountPtr);

  const expected = [];
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    if (p[0]>=lo[0]&&p[0]<=hi[0]&&p[1]>=lo[1]&&p[1]<=hi[1]&&p[2]>=lo[2]&&p[2]<=hi[2]) expected.push(i);
  }
  check(`kdtree range_query (found ${indices.length}, expected ${expected.length})`,
        JSON.stringify(indices) === JSON.stringify(expected));

  const q = [1000, -2000, 3000], k = 10;
  const qPtr = malloc32(new Int32Array(q));
  const outCountPtr2 = Module._malloc(8);
  const resPtr2 = kdtreeKnn(qPtr, k, outCountPtr2);
  const knnIndices = Array.from(Module.HEAPU32.subarray(resPtr2 / 4, resPtr2 / 4 + Module.HEAPU32[outCountPtr2 / 4]));
  wasmFree(resPtr2); Module._free(qPtr); Module._free(outCountPtr2);

  const distSq = (p) => (p[0]-q[0])**2 + (p[1]-q[1])**2 + (p[2]-q[2])**2;
  const bruteKnn = pts.map((p,i)=>({i, d: distSq(p)})).sort((a,b)=>a.d-b.d).slice(0,k).map(x=>x.i);
  check(`kdtree k_nearest (${knnIndices.length} results)`, JSON.stringify(knnIndices) === JSON.stringify(bruteKnn));
}

process.exit(failed ? 1 : 0);
