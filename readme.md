# mini-project-cpp

the goal of this project is to show random images available in `data/small-Voc2007`, predict what is it and show where
is the interest of the prediction.


> **Warning**
>
> The compilation only works with Windows platform and with the Visual Studion compiler. (see our CMakeList.txt)

## Install it

create a directory named `dependencies` and
extract [this release](https://github.com/opencv/opencv/releases/download/4.7.0/opencv-4.7.0-windows.exe) inside.

Once you have done that, you have to run the `CMakeList.txt` with `cmake`, it will create a cmake directory.

Copy / Paste all `dll` files of
[this release of opencv downloaded just before](https://github.com/opencv/opencv/releases/download/4.7.0/opencv-4.7.0-windows.exe)
and put them near to the `.exe`, available in `cmake-build-release-visual-studio`
directory.

you can now run perfectly the project



