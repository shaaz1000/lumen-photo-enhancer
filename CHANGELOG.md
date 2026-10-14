# Changelog

All notable changes to this project are listed here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.1.0] - 2026-10-14

Real restoration instead of mild upscaling. 1.0.0 did run Real-ESRGAN, but its small model plus finishing steps that pulled the result back toward plain bicubic made many outputs look almost unchanged.

### Fixed

* Finishing no longer blends bicubic pixels back into the AI result. That blend (15 to 35%, and more on faces) was the main reason results looked soft.
* Face protection no longer weakens reconstruction on faces. Faces now get the full model, with only sharpening eased on skin.
* Sharpening allows a small, bounded overshoot so edges look crisp at upscaled sizes, still without halos.
* Tile seams: neighbouring tiles now overlap and are blended with linear ramps, matching whole-image inference.
* JPEG compression detection no longer flags clean q80 files as heavy, and blurred images are no longer called heavily compressed because they compress well.
* Export defaults to lossless PNG, so the download holds exactly the previewed pixels.
* The comparison renders original and enhanced with the same resampling; 1.0 showed a smoothed original next to a pixelated result when zoomed in.
* 8× is no longer recommended automatically; it is 4× AI plus resampling and stays available by hand.

### Added

* Real-ESRGAN x4plus (RRDBNet, 16.7 M parameters) running in the browser on WebGL, verified against the PyTorch-layout reference. Downloaded on first use (33 MB in three parts).
* Deep Restoration mode for soft, low resolution internet photos.
* Reconstruction control: Faithful (fast model, minimal reconstruction), Balanced (deep model, default), Maximum Detail (strongest, may add inferred texture).
* Detail matching: the analysis estimates how much real detail an image holds; soft images are processed at that scale so the model rebuilds edges instead of enlarging blur.
* Analysis now reports real detail, the main issue, and content type, and flags strong focus or motion blur as beyond repair instead of promising a fix.
* Texture control now mixes in the weak-denoise model's output (natural grain) instead of bicubic.
* Output verification: results that do not match the original at its own size (black, NaN or shifted output from a failing GPU) are discarded with an error, never shown or exported.
* Processing report listing the models, every stage, sizes, time and the verification result.
* Zoom presets at 100, 200 and 400%, and a Face button that frames each detected face for comparison.
* Face detection now includes five alignment points.
* Server: GFPGAN v1.4 face restoration on ONNX Runtime (no PyTorch) with adjustable strength, aligned from the app's landmarks; deep or general model selection; texture source returned in the same response; nothing stored.
* Desktop output limit raised to 34 MP (1080p at 4×); phones keep the 16.7 MP iOS limit.
* GPU Docker image (`backend/Dockerfile.gpu`) and checksum-verified model downloads.
* Benchmark suite (`bench/`): six degradation cases from public domain and CC0 images, PSNR, SSIM, detail and blur measures, face identity distance, timing, and comparison crops, against bicubic and the 1.0.0 pipeline.

### Changed

* The Natural to Stronger slider is replaced by the Reconstruction control.
* Server API: new `model`, `weak` and `landmarks` fields; `face_restore` now uses landmarks sent by the client.

## [1.0.0] - 2026-10-13

First public release.

### Added

* In-browser AI super resolution with Real-ESRGAN general v3 on TensorFlow.js and WebGL.
* 2× and 4× AI upscaling; 8× as 4× AI followed by 2× bicubic, labelled in the interface.
* Noise and JPEG artifact removal through checkpoint interpolation.
* Face detection, image analysis, seven modes, comparison viewer, export, conventional fallback.
* FastAPI server with Real-ESRGAN general v3 and x4plus on ONNX Runtime, optional GFPGAN.
* Upload validation, rate limiting, in-memory processing, Docker image, tests.
