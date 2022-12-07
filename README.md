# Multiple-Person Background Removal Using Orbbec Femto Developer Kit

## Team Name: Farmhand

## Folder Structure

```txt
(SolutionDir)
├─MultiplePersonBackgroundRemoval.sln  // Visual Studio 2022 solution
|
├─MultiplePersonBackgroundRemoval      // Project files
│
└─SDK               //SDK headers and libraries
    ├─include       //SDK headers
    │
    └─lib           //SDK libraries
```

## OS Used for Development

Microsoft Windows [Version 10.0.19042.2130], 64-bit.

## Build the Solution

Start Microsoft Visual Studio 2022 and open the solution file (.sln).

Build targets: x64, Debug | Release

## Requirements

* OpenCV C++ Library

[`vcpkg`](https://vcpkg.io/en/index.html) should be used to install the library.

`vcpkg list` command shows the following OpenCV packages on my PC:

```txt
opencv4:x64-windows                                4.5.5#7          computer vision library
opencv4[default-features]:x64-windows                               Platform-dependent default features
opencv4[dnn]:x64-windows                                            Enable dnn module
opencv4[jpeg]:x64-windows                                           JPEG support for opencv
opencv4[png]:x64-windows                                            PNG support for opencv
opencv4[quirc]:x64-windows                                          Enable QR code module
opencv4[tiff]:x64-windows                                           TIFF support for opencv
opencv4[webp]:x64-windows                                           WebP support for opencv
opencv:x64-windows                                 4.5.5#1          Computer vision library
opencv[default-features]:x64-windows                                Platform-dependent default features
```

## License

© Copyright 2022 Farmhand.

Licensed under the [MIT license](LICENSE).
