import time
import board
import neopixel

light_matrix = neopixel.NeoPixel(board.D10, 16, brightness = 0.3)
light_matrix.fill((0, 0, 0))