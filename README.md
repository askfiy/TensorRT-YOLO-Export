# TensorRT-YOLO Export Pipeline

Interactive helper for converting the latest model in `models/` into:

- plain ONNX in `models/onnx`
- TensorRT-YOLO compatible ONNX in `models/trt-onnx`

It supports two paths:

1. `PT -> ONNX -> TRT-YOLO ONNX`
2. `ONNX -> TRT-YOLO ONNX`

## Features

- Automatically picks the latest `.pt` in `models/`
- Or automatically picks the latest plain `.onnx` in `models/onnx`
- Interactive prompt for normal NMS vs `e2e`
- Uses `trtyolo-export` to convert plain ONNX into TensorRT-YOLO ONNX
- Includes a small metadata patch step for ONNX outputs before conversion

## Directory Layout

```text
models/
  *.pt
  onnx/
    *.onnx
  trt-onnx/
    *_trtyolo.onnx
```

## Usage

Install dependencies:

```bash
uv sync
```

Run the pipeline:

```bash
uv run main.py
```

Then choose:

- `1`: export from the latest `.pt`
- `2`: convert from the latest existing `.onnx`

You will also be asked whether the model is `e2e`.

- If `e2e = no`, the script passes `--iou-thres 0.45`
- If `e2e = yes`, the script skips `--iou-thres`

## Dependencies

This project depends on:

- `ultralytics`
- `trtyolo-export >= 2.0.0`
- `onnx`

## Notes

- This project targets the `onnx -> trt-onnx` conversion path.
- `trtyolo-export` converts an existing ONNX model; it does not export directly from `.pt`.
- For YOLO26 NMS-free detect models, the installed `trtyolo_export` package may require a small local patch so the generated `det_boxes` output keeps valid dtype/shape metadata.

## Upstream

- Runtime / exporter ecosystem: [laugh12321/TensorRT-YOLO](https://github.com/laugh12321/TensorRT-YOLO)
- ONNX converter: [laugh12321/trtyolo-export](https://github.com/laugh12321/trtyolo-export)
