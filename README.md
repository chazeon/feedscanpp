# feedscanpp

A robust document and receipt scanner optimized for ADF and thermal paper.

Tested with Epson ES-C200 scanner using `scanimage` set to color mode.

## Features

- **Automatic Skew Correction**: FFT-based skew detection to straighten tilted documents
- **Smart Trimming**: Removes white borders automatically
- **Tint Removal**: Eliminates yellowing and discoloration from aged thermal paper
- **Intelligent Color Detection**: Distinguishes between Color, Grayscale, and Black & White documents
- **Adaptive Enhancement**: Optimizes image quality based on detected content type

## Installation

### Using pip

```bash
pip install feedscanpp
```

### Using uvx (no installation required)

```bash
uvx --from git+https://github.com/chazeon/feedscanpp.git feedscanpp scan-*.png -o output-{index}.png
```

### From source

```bash
git clone https://github.com/chazeon/feedscanpp.git
cd feedscanpp
pip install -e .
```

## Usage

```bash
feedscanpp scan-*.png -o processed-{index}.png
```

The output path supports placeholders:
- `{index}`: Sequential index (0, 1, 2, ...)
- `{name}`: Original filename with extension
- `{stem}`: Original filename without extension

### Options

You can disable individual processing steps:

```bash
--no-rotate    Disable rotation correction
--no-trim      Disable trimming
--no-tint      Disable tint removal
--no-enhance   Disable enhancement
```

Example:

```bash
# Only apply rotation and trimming
feedscanpp scan.png -o output.png --no-tint --no-enhance
```

## How It Works

The processing pipeline consists of four stages:

1. **Rotation Correction**
   - Detects skew angle using FFT magnitude spectrum analysis
   - Rotates image to align text horizontally (±5 degrees)

2. **Trimming**
   - Finds content bounding box by detecting non-white pixels
   - Removes excess white borders

3. **Tint Removal**
   - Estimates background color using morphological dilation
   - Normalizes brightness to eliminate yellowing
   - Applies CLAHE for local contrast enhancement

4. **Analysis & Enhancement**
   - Detects document type: Color, Grayscale, or Black & White
   - Applies Otsu's thresholding for B&W documents
   - Preserves color/grayscale when appropriate

## License

MIT
