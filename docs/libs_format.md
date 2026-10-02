# The `.libs` file format

`.libs` is the open, self-contained format pylibs reads. It is a **NumPy
`.npz` archive**: a standard zip holding three uncompressed `.npy` members.
Nothing about it is proprietary, it depends on no private code, and any
language with a zip library and a `.npy` reader can read or write one.

This document is the specification. The reference implementations are
`pylibs.core.io.readers.libs_reader` and `pylibs.core.io.writers.libs_writer`.

## Reading one without pylibs

```python
import numpy as np, json

with np.load("sample1_1.libs", allow_pickle=False) as archive:
    data = archive["data"]  # (n_scans, n_pixels) float
    wavelengths = archive["wavelengths"]  # (n_pixels,) float64, nm
    params = json.loads(str(archive["params"]))  # dict
```

That is the whole format. `allow_pickle=False` is safe and deliberate: no
member is ever a pickled object.

## Members

| member | shape | dtype | meaning |
|---|---|---|---|
| `data` | (n_scans, n_pixels) | float64 or float32 | one spectrum per ablation position |
| `wavelengths` | (n_pixels,) | float64 | the wavelength axis, nanometres, strictly increasing |
| `params` | scalar | unicode string | a JSON object, described below |

All three are stored uncompressed (`ZIP_STORED`). That is required, not
incidental: it is what lets a reader memory-map `data` at its offset inside
the archive and stream a scan larger than memory, instead of inflating the
whole thing first. A file written with `numpy.savez_compressed` still reads
correctly, but loses that property.

`data` rows and `wavelengths` entries correspond by position: `data[k, j]` is
the intensity at `wavelengths[j]` for scan `k`. Intensity units are whatever
the instrument produced; pylibs treats them as arbitrary and never converts
them.

## Scan order

**Rows of `data` are in serpentine order.** Reading a raster row by row, rows
run top to bottom; within a row, even rows (0, 2, 4, …) run left to right and
odd rows run right to left. This mirrors how the stage physically travels, so
consecutive rows of `data` are physically adjacent measurements.

```
row 0:   0  1  2  3  4        data row order: 0 1 2 3 4
row 1:   5  6  7  8  9                        9 8 7 6 5
row 2:  10 11 12 13 14                       10 11 12 13 14
```

There are no per-pixel coordinate arrays. Position is reconstructed from
`HeightPixels` and `WidthPixels`, optionally narrowed by
`PixelAssignmentMatrix`. `pylibs.core.reporting.raster.to_raster` undoes the
ordering when placing values on a map.

Data arriving from another format is almost always in plain raster order, so
`write_libs` takes `order="raster"` by default and reverses odd rows on the
way in. Storing raster-ordered data without that conversion mirrors alternate
rows of every map produced from the file.

## The `params` object

A JSON object stored as a NumPy unicode scalar. Every field is optional as
far as the file format goes; what each one unlocks in pylibs is below.

### Required for spatial output

| field | type | meaning |
|---|---|---|
| `HeightPixels` | int | raster rows |
| `WidthPixels` | int | raster columns |

`HeightPixels * WidthPixels` must equal `n_scans`, unless
`PixelAssignmentMatrix` is present, in which case its set cells must. Without
these two, feature extraction and every distribution metric still work, but
feature maps and the `local` bootstrap resampling method cannot.

### Optional, physical extent

| field | type | meaning |
|---|---|---|
| `WidthIMG` | float | scanned width, millimetres |
| `HeightIMG` | float | scanned height, millimetres |
| `Resolution` | float | pixel pitch, millimetres per pixel |
| `SpatialUnits` | string | units the extent is expressed in; written by `write_libs` |

Map axes read in millimetres when `WidthIMG` and `HeightIMG` are present, or
when `Resolution` is, in which case the extent is `Resolution` times the pixel
count. Each axis resolves independently. With none of them, axes read in
pixels.

### Optional, partial scans

| field | type | meaning |
|---|---|---|
| `PixelAssignmentMatrix` | nested list of 0/1 | which cells of the frame were measured |

Shape `(HeightPixels, WidthPixels)` with exactly `n_scans` set cells, for a
sample that does not fill its rectangular frame, such as a round pellet.
Scans are placed at the set cells in serpentine order and unmeasured cells
are drawn as blanks. A matrix whose shape or count cannot place the sample's
scans is logged and ignored rather than guessed at.

### Optional, instrument and provenance

| field | type | meaning |
|---|---|---|
| `Detectors` | object | per-detector channel masks, keyed by detector number |
| `nDetectors` | int | number of spectrometers |
| `NbLambda` | int | wavelength count; must equal `n_pixels` |
| `NbPixels` | int | scan count; must equal `n_scans` |
| `LambdaMinA`, `LambdaMaxA`, `LambdaMinB`, … | float | per-detector wavelength ranges |
| `slambda` | list of float | the instrument's own wavelength calibration |
| `AnalysisDate` | string | acquisition date |
| `AnalysisTime` | string | acquisition duration |
| `filename`, `stem` | string | the file's own name, for report titles |

None of these affect computation. `Detectors` and `slambda` are recorded on
the sample when present and reported as missing when not. Any additional key
is preserved and ignored, so an instrument's own metadata can ride along.

## Writing one

Through pylibs, which validates and derives the geometry fields:

```python
from pylibs.core import api

api.write_libs(
    "converted.libs",
    spectra,  # (n_scans, n_pixels), a memmap, or blocks
    wavelengths,  # (n_pixels,) strictly increasing
    grid=(41, 21),  # or coordinates=(n_scans, 2)
    step=0.1,
    units="mm",
    metadata={"Instrument": "..."},
)
```

Or by hand, with NumPy alone:

```python
import numpy as np, json

np.savez(
    "converted.libs",
    data=spectra,
    wavelengths=wavelengths,
    params=np.array(json.dumps({"HeightPixels": 41, "WidthPixels": 21, "Resolution": 0.1})),
)
```

The hand-written version is a valid `.libs` file. It skips the validation,
and it stores whatever scan order the array was already in, so the serpentine
convention is the caller's responsibility.

## Validity checklist

A file pylibs will process:

1. is a zip holding `data.npy`, `wavelengths.npy` and `params.npy`;
2. `data` is 2D, float32 or float64, finite;
3. `wavelengths` is 1D, float64, strictly increasing, finite, and as long as
   `data`'s second axis;
4. `params` parses as a JSON object;
5. if it carries `HeightPixels` and `WidthPixels`, their product equals the
   scan count, or the `PixelAssignmentMatrix` set-cell count does.
