# Ported from GLSL code https://www.shadertoy.com/view/mtyGWy

import taichi as ti

ti.init(arch=ti.gpu)  # Initialize Taichi with GPU (if available)

width, height = 800, 800
pixels = ti.Vector.field(4, dtype=float, shape=(width, height))

@ti.func
def fract(x):
    return x - ti.floor(x)

@ti.func
def palette(t):
    a = ti.Vector([0.5, 0.5, 0.5])
    b = ti.Vector([0.5, 0.5, 0.5])
    c = ti.Vector([1.0, 1.0, 1.0])
    d = ti.Vector([0.263, 0.416, 0.557])
    return a + b * ti.cos(6.28318 * (c * t + d))

@ti.kernel
def paint(t: float):
    for i, j in pixels:
        finalColor = ti.Vector([0.0, 0.0, 0.0])
        uv = ti.Vector([i / width - 0.5, j / height - 0.5]) * 2.0
        uv.x *= width / height
        uv0 = uv # keep the big circle

        for p in range(4): # loop for small circles
            uv = fract(uv * 1.5) - 0.5 # small circles
            d = uv.norm() * ti.exp(-uv0.norm())  # big circle
            color = palette(uv0.norm() + p * 0.4 + t * 0.2)  # color gradient + time shift
            d = ti.sin(d * 8 + t) / 8  # sin wave repetition
            d = ti.abs(d)  # negative numbers are black, this makes the inside bright
            d = ti.pow(0.01 / d, 1.2) # brightness

            finalColor += color * d

        pixels[i, j] = ti.Vector([finalColor[0], finalColor[1], finalColor[2], 1.0])

gui = ti.GUI("Taichi Shader", res=(width, height))

iTime = 0.0

while gui.running:
    if gui.res != (width, height):
        # Update the resolution
        width, height = gui.res
        print(gui.res)
        pixels = ti.Vector.field(4, dtype=float, shape=(width, height))
    paint(iTime)
    gui.set_image(pixels)
    gui.show()
    iTime += 0.02
