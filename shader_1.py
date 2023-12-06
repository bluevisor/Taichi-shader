# Ported from GLSL code https://www.shadertoy.com/view/mtyGWy

import taichi as ti # pip install taichi

ti.init(arch=ti.gpu)  # Initialize Taichi with GPU (if available)

width, height = 1728, 984
pixels = ti.Vector.field(4, dtype=float, shape=(width, height))

speed = 0.01 # 0.01
complexity = 4 # 4
scale = 2 # 2 bigger number = zoomed out
fragment = 1.5 # 1.5 bigger number = smaller pieces 
repeat = 4 # 4
smallCircleCenter = 0.5 # 0.5
bigCircleCenter = {'x': 0.5, 'y': 0.5} # 0.5, 0.5
glow = 0.01 # 0.01
glowPow = 1.2 # 1.2
pPow = 0.4 # 0.4
tPow = 0.2 # 0.2

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
        uv = ti.Vector([i / width - bigCircleCenter['x'], j / height - bigCircleCenter['y']]) * scale
        uv.x *= width / height
        uv0 = uv # keep the big circle

        for p in range(complexity): # loop for small circles
            uv = fract(uv * fragment) - smallCircleCenter # small circles
            d = uv.norm() * ti.exp(-uv0.norm())  # big circle
            color = palette(uv0.norm() + p * pPow + t * tPow)  # color gradient + time shift
            d = ti.sin(d * repeat + t) / repeat  # sin wave repetition
            d = ti.abs(d)  # negative numbers are black, this makes the inside bright
            d = ti.pow(glow / d, glowPow) # glow / brightness

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
    iTime += speed
