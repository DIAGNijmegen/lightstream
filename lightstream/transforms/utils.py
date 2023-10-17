import pyvips

def draw_grid(image: pyvips.Image, grid_size):
    # Draw grid lines
    for i in range(0, image.width, grid_size):
        image = image.draw_line([0, 0, 0], i, 0, i, image.height)
    for j in range(0, image.height, grid_size):
        image = image.draw_line([0, 0, 0], 0, j, image.width, j)

    return image