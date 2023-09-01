import time
import board
import neopixel

light_matrix = neopixel.NeoPixel(board.D10, 16, brightness = 0.3)

for i in range(0, 256):
    light_matrix.fill((i, i, 50))
    time.sleep(0.1)
    
light_matrix.fill((0, 0, 0))