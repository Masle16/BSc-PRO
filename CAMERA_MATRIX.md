# Camera matrix

$$
\begin{bmatrix}
    i \cdot w \\
    j \cdot w \\
    w
\end{bmatrix} =
\begin{bmatrix}
    f & 0 & c_{i} \\
    0 & f & c_{j} \\
    0 & 0 & 1
\end{bmatrix} \cdot
\begin{bmatrix}
    x \\
    y \\
    z
\end{bmatrix} =
\begin{bmatrix}
    \frac{width}{2tan (\frac{FOV}{2}) } & 0 & \frac{width - 1}{2} \\
    0 & \frac{width}{2tan (\frac{FOV}{2}) } & \frac{height - 1}{2} \\
    0 & 0 & 1
\end{bmatrix} \cdot
\begin{bmatrix}
    x \\
    y \\
    z
\end{bmatrix}
$$

From the [cameras datasheet](https://www.aliexpress.com/item/11-Megapixel-Autofocus-USB-Camera-High-Resolution-USB2-0-SONY-IMX214-Color-CMOS-Mini-Webcam-Camera/32909008612.html?fbclid=IwAR2Xkw5758lqk2K-LyN4otorUFYzV6witqcCv09OpLgGKhtF_joL4774Byc) the FOV is found to be 75 degrees. Futhermore, a resolution of 720 x 1280 is used.

Thereby, the camera matrix becomes:

$$
\begin{bmatrix}
    i \cdot w \\
    j \cdot w \\
    w
\end{bmatrix} =
\begin{bmatrix}
    834.0642 & 0 & 639.5 \\
    0 & 834.0642 & 359.5 \\
    0 & 0 & 1
\end{bmatrix} \cdot
\begin{bmatrix}
    x \\
    y \\
    z
\end{bmatrix}
$$