# CSA vs. real specialized geometric compressors

`BENCHMARKS.md`, `USE_CASES.md`, and `REAL_CORPUS_BENCHMARK.md` all compare CSA against *general-purpose* compressors (gzip/bz2/lzma). That's the wrong comparison for the Rod-Joint Transform / Pantograph Lift's actual target domain: point clouds and trajectories have real, mature, *specialized* production compressors built specifically for them. This file asks the harder, more relevant question -- how does CSA do against those, on real (not synthetic, not generated-for-this-project) data?

Two real datasets, two real specialized competitors:

1. **LiDAR point cloud vs. [LASzip](https://github.com/LASzip/LASzip)** -- the actual industry-standard LAS/LAZ point-cloud compressor used throughout the airborne-LiDAR and GIS industry (PDAL, libLAS, QGIS, ArcGIS all use it under the hood).
2. **GPS trajectory vs. Google's Encoded Polyline Algorithm Format** -- the real, widely-deployed specialized encoding Google Maps/Mapbox/OSRM use for compact polyline transport.

## Data provenance (read this before trusting the numbers)

- **LiDAR**: `stadium-utm.laz`, a real captured airborne LiDAR scan of Autzen Stadium (Eugene, Oregon), sourced from [PDAL's public test-data repository](https://github.com/PDAL/data) (`autzen/stadium-utm.laz`, fetched via git+LFS). This is real sensor data used as a standard regression/benchmark corpus across the point-cloud tooling ecosystem, not captured by this project -- 693,895 points.
- **GPS trajectory**: one real, continuous, single-trip `.plt` file (`20090403011657.plt`, 14,186 points) from user `000` of Microsoft Research's public [Geolife Trajectories 1.3](https://www.microsoft.com/en-us/download/details.aspx?id=52367) dataset (182 real users' GPS logs, collected 2007-2012). Downloaded directly from Microsoft's official distribution.

## Building the LASzip comparison fairly

A LAS file's compressed size normally includes intensity, classification, GPS time, return number, and other per-point attributes that CSA's geometric mode was never designed to touch. To make this an honest apples-to-apples comparison:

1. Every point's raw (unscaled) integer X/Y/Z was extracted from the source `.laz` via a small driver written against LASzip's own C API (`laszip_open_reader`/`laszip_read_point`) -- this is the *identical* integer stream both codecs then compress.
2. That stream was re-packed into a **fresh, minimal LAS file containing only X/Y/Z** (point format 0, no VLRs, scale=1/offset=0 so the stored integers are byte-identical to the extracted stream) and compressed with LASzip -- this is LASzip's real geometry-only number, not inflated or deflated by attributes CSA doesn't model.
3. CSA's `compress-geo3d` ran on the exact same extracted integer stream (`--scale 1`, so no re-quantization -- the transform sees the identical integers LASzip sees).
4. Every number below was round-trip verified byte-exact (or, for CSA, exact after accounting for its text I/O format) against the original extracted stream before being trusted.

For reference, the *original downloaded* `stadium-utm.laz` (all attributes, LASzip's default settings) is 2,748,303 bytes -- not used in the ratio comparisons below since it isn't a geometry-only number, but included so nobody can accuse the geometry-only repack of being cherry-picked to make LASzip look worse than its real-world deployment.

## Result 1: LiDAR point cloud (693,895 real points)

| method | size (bytes) | vs. raw packed | compress time | decompress time | round-trip |
|---|---:|---:|---:|---:|:---:|
| raw packed (4-byte int × 3 axes) | 8,326,740 | -- | -- | -- | (reference) |
| gzip -9 | 4,116,642 | 50.6% smaller | 0.74s | -- | (reference) |
| bz2 -9 | 3,414,235 | 59.0% smaller | 0.51s | -- | (reference) |
| lzma -9 | 2,208,084 | 73.5% smaller | 3.47s | -- | (reference) |
| **CSA `compress-geo3d`** | **1,336,093** | **84.0% smaller** | **29.3s** | **10.6s** | PASS |
| **LASzip (geometry-only)** | **1,012,384** | **87.8% smaller** | **0.39s** | **0.42s** | PASS |
| *(original stadium-utm.laz, all attributes, for context only)* | *2,748,303* | *(not geometry-only)* | -- | -- | -- |

## Result 2: GPS trajectory (14,186 real points, one continuous trip)

| method | size (bytes) | vs. raw packed | round-trip |
|---|---:|---:|:---:|
| raw packed (4-byte int × 2 axes, 1e6 precision) | 113,488 | -- | (reference) |
| bz2 -9 | 53,447 | 52.9% smaller | (reference) |
| gzip -9 | 50,956 | 55.1% smaller | (reference) |
| lzma -9 | 31,484 | 72.3% smaller | (reference) |
| Google Encoded Polyline (1e5 precision -- lossy by design, ~0.56m max error) | 31,189 | 72.5% smaller | PASS (lossy) |
| **CSA `compress-geo2d`** (exact, 1e6 precision) | **28,335** | **75.0% smaller** | **PASS (exact)** |

## Honest verdict

- **LiDAR: CSA loses to the specialized competitor, clearly and on both axes.** LASzip is 24.2% smaller *and* ~75x faster to compress (0.39s vs. 29.3s) *and* ~25x faster to decompress (0.42s vs. 10.6s). This is the credibility-critical result, and it is a real loss, not a close call. LASzip's per-point predictor (built around LAS's scan-order structure: return number, scan direction, GPS-time-correlated locality) is tuned specifically for how airborne LiDAR scanners actually sweep a surface; CSA's Rod-Joint Transform assumes a calibrated rotation+scale relationship between consecutive edge vectors, which is a good model for a genuinely curving path (a helix, a GPS trace) but not for an approximately-random-order point cloud sampled off a bumpy real surface. **CSA does still meaningfully beat every general-purpose compressor tested here, including lzma -9 (by 39.5%)** -- so the honest position is "beats gzip/bz2/lzma on real LiDAR geometry, loses decisively to the actual specialized competitor," not "doesn't work here."

- **GPS trajectory: CSA wins, including against the real specialized competitor.** 28,335 bytes beats not just every general-purpose compressor tested (72-75% smaller than raw packed, ahead of lzma -9) but also Google's Encoded Polyline format -- and does it **exactly losslessly**, while the polyline format is lossy by design at its standard 1e5 precision (~0.56m of real error introduced here). This is a genuinely encouraging, narrow-but-real result: a single continuous real human trip (walking/driving, turning, pausing) is close enough to the "calibrated rotation predicts the next edge" model that the Rod-Joint Transform actually earns its keep here, on real recorded human movement, not just synthetic spirals.

- **The overall picture is genuinely mixed, not a clean win.** The Rod-Joint Transform's structural assumption (locally consistent rotation+scale between consecutive edges) is a real, exploitable property of *trajectory-shaped* data (GPS traces, vehicle paths, camera paths) but not of *point-cloud-shaped* data (LiDAR scans, whose point ORDER reflects a scan pattern, not a single continuous physical path through space). That is a meaningful, specific finding about where this transform's model actually applies -- not "point clouds" broadly, but "single continuous paths" specifically. Anyone building on this should target trajectory/path compression (fleet GPS, drone flight logs, AR/VR head-tracking paths, robot odometry), not point-cloud storage, where LASzip (or the newer Draco) already has a decisively better answer.

- **Performance, not just ratio, is a real gap on the LiDAR side.** Even setting the ratio loss aside, CSA's compress/decompress times (29.3s / 10.6s for 693,895 points) are 25-75x slower than LASzip's production-tuned codec on the same data. That gap doesn't show up at all on the 14,186-point GPS track (0.3s), so it's specifically a scaling problem, not a fixed per-call overhead -- worth investigating (likely the per-block candidate-lag search's cost) before this could be called production-viable at LiDAR-scale point counts even where the ratio was competitive.

## Regenerating these numbers

`bench/real_geo_benchmark.py` regenerates the GPS-side numbers directly (it only needs `scissorc` built and the Geolife `.plt` file). The LiDAR side needs a one-time setup that isn't included in that script because it depends on a third-party library build and a multi-hundred-MB download -- see the comment at the top of the script for the exact steps (clone+build LASzip with `-DLASZIP_DYN_LINK=1 -DLASZIP_SOURCE=1`, build the small `laz_geo_tool` C API driver against it, fetch `autzen/stadium-utm.laz` from `PDAL/data` via git+LFS).
