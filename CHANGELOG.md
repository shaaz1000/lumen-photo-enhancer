# Changelog

All notable changes to this project are listed here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.0] - 2026-10-13

First public release.

### Added

* In-browser AI super resolution with Real-ESRGAN general v3, implemented directly on TensorFlow.js ops and run on the GPU through WebGL.
* 2× and 4× AI upscaling; 8× as 4× AI followed by 2× bicubic, labelled in the interface.
* Noise and JPEG artifact removal through checkpoint interpolation between `realesr-general-x4v3` and `realesr-general-wdn-x4v3`.
* Tiled inference with 24 px overlap, real tile progress and cancellation.
* Face detection with the tiny face detector, enlarging small images so small faces are found.
* Identity preserving face protection: gentler denoising, reduced sharpening and more original texture on faces.
* Image analysis for size, format, blur, JPEG blocking, noise, faces and a recommended scale.
* Seven modes: Auto, Restoration, Portrait, Screenshot to HD, Ultra Sharp, Natural and Social Ready.
* Controls for scale, denoise, sharpness, texture preservation, colour correction, face protection and strength.
* Halo-free sharpening, bounded white balance and auto levels.
* Conventional fallback engine (bicubic, bilateral denoise, sharpening), labelled as not AI.
* Viewer with draggable comparison slider, side by side, single views, wheel and pinch zoom, pan, fit and 1:1 pixel zoom.
* Export to PNG, JPEG or WebP with quality, size presets and measured file size.
* Upload by drag and drop, file picker or paste, validated by file signature.
* Light and dark themes, phone and tablet layouts, reduced motion support.
* FastAPI server with Real-ESRGAN general v3 and x4plus on ONNX Runtime, built from official checkpoints without PyTorch.
* Optional GFPGAN v1.4 face restoration on the server, off by default and blended at most 60%.
* Server upload validation, decompression bomb guard, pixel caps, per client rate limiting, concurrency limits and in-memory processing.
* Optional hosted inference proxy through Replicate with a server-side key.
* NumPy weight converter, reference model and fixture generator.
* 42 frontend tests, 15 backend tests and a headless browser smoke test.
* Docker image and weight download script.
