# Changelog

All notable changes to this project are listed here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.1.3] - 2026-10-15

### Fixed

* Enhance not starting in the web artifact. The legacy TensorFlow.js build evaluates code from strings, which a strict content security policy blocks, leaving TensorFlow.js half loaded (`setBackend is not a function`). The app now loads the ES2017 build, which needs no `eval`, and a test runs the app in a sandboxed iframe under such a policy.
* Desktop runs that took many minutes with no sign of progress. The app measures this device's speed at start and only starts the deep model on its own when it should finish within about two minutes; Maximum Detail still forces it.

### Added

* Estimated time on this device, shown before each run.
* A note in the progress card when a step runs long, with what to try instead.
* Any unexpected error is shown on screen instead of failing silently.
* Copy diagnostics (in What runs where) for bug reports.

## [1.1.2] - 2026-10-14

### Fixed

* The page reloading on iPhone right after an enhancement finished, which brought back the start image. A full phone screenshot at 2× made a 12 MP result, and finishing, encoding and showing it exceeded the memory iOS allows a page.
  * Phones now have a 6 MP output budget. 2× stays available and is shrunk to fit; the input is resized first so the model never produces pixels that are thrown away.
  * Finishing works on one copy in place instead of allocating a new full-size image per step.
  * On phones, the model's 2× output is used when only a small resample remains, instead of a 4× image four times the size.

### Added

* Crop tool: drag over the part to keep, then enhance only that. Screenshots get a tip suggesting it.

## [1.1.1] - 2026-10-14

### Fixed

* Enhance failing on phones. Large phone photos ran out of browser memory: iOS limits canvas memory per page and kills or blanks tabs that use too much.
  * The preview now shows image elements backed by the exported PNG instead of extra canvases, and canvases are released right after use.
  * Tiles are blended straight into the output bytes, dropping two full-size float buffers.
  * On phones, photos over 0.3 megapixels use the fast model unless Maximum Detail is chosen, the second texture pass is skipped for large outputs, and the recommended scale keeps results under 8 MP.
  * The deep model uses larger tiles on phones (less wasted overlap) and the GPU texture pool is capped.
* A lost GPU context (the browser resetting WebGL under memory pressure) is now reported with a clear message instead of producing a broken result.

### Added

* Progress shows an estimated time left.
* The downloaded PNG is byte for byte the image shown in the preview (checked by the end-to-end test).

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
