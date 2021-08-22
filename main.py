import multiprocessing
from os import listdir, makedirs
from typing import Any

import numpy as np
from PIL import Image

# Size of small square images, used for generation
GRID_SIZE = 16
RESOLUTION_MULTIPLIER = 2


# TODO: fix function signature, if possible
def average_color(img: Image) -> tuple[Any, ...]:
    return tuple(np.average(np.asarray(img).reshape(-1, 3), axis=0))


def vector_dist(a: tuple[Any, ...], b: tuple[Any, ...]) -> Any:
    # For a proper euclidean distance it needs to return a sqrt of below
    # In this context, however, it doesnt matter, as only the min distance is calculated
    return (a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2

    # Works on a, b of any length, and looks better.
    # Is noticeably slower (~30%)
    # return sqrt(sum([(x - y)**2 for x, y in zip(a, b)]))

    # Might be faster in more than 3 dimensions, but not in this case
    # return np.linalg.norm(a - b)


def process_image(input_file: str, mosaic_images: list[Image.Image],
                  mosaic_images_color_avg: list[Any]) -> None:
    print(f' - Processing {input_file}...', end='\t', flush=True)
    with Image.open("input/" + input_file) as img:
        # Resize the image to be divisible into the grid
        width, height = img.size
        width = (width // GRID_SIZE) * GRID_SIZE
        height = (height // GRID_SIZE) * GRID_SIZE
        img = img.resize((width, height))

        out_img = Image.new(mode='RGB',
                            size=(width * RESOLUTION_MULTIPLIER,
                                  height * RESOLUTION_MULTIPLIER))

        # Process the image blocks
        for y in range(0, height - GRID_SIZE + 1, GRID_SIZE):
            for x in range(0, width - GRID_SIZE + 1, GRID_SIZE):
                box = (x, y, x + GRID_SIZE, y + GRID_SIZE)
                img_frag = img.crop(box)

                frag_avg = average_color(img_frag)
                # Calculate Euclidean distances to mosaic fragment average color vectors
                # For some reason this method is much faster than manual iteration
                distances = [
                    vector_dist(frag_avg, mosaic_color)
                    for mosaic_color in mosaic_images_color_avg
                ]

                mosaic_frag = mosaic_images[distances.index(min(distances))]

                # mosaic_frag = None
                # min_dist = None
                # for i, mosaic_color in enumerate(mosaic_images_color_avg):
                #     dist = vector_dist(frag_avg, mosaic_color)
                #     if min_dist is None or dist < min_dist:
                #         min_dist = dist
                #         mosaic_frag = mosaic_images[i]

                out_img.paste(mosaic_frag,
                              tuple(x * RESOLUTION_MULTIPLIER for x in box))
        out_img.save("output/" + input_file)
    print("Done.")


if __name__ == "__main__":
    # Create directories
    # TODO: check that the code does not break if input/source dirs are empty
    makedirs("input", exist_ok=True)
    makedirs("source", exist_ok=True)
    makedirs("output", exist_ok=True)

    # Prepare mosaic images
    # Load them and make them square
    print("Loading mosaic images...")
    mosaic_images: list[Image.Image] = []
    mosaic_images_color_avg: list[Any] = []

    for f in listdir("source"):
        # TODO: PNGs are broken for some reason. Investigate.
        if f.endswith((".jpg", ".jpeg")):
            img = Image.open("source/" + f)

            min_size = min(img.size)
            # img.crop((0, 0, min_size, min_size))
            # img = img.resize((min_size, min_size))
            img = img.resize((GRID_SIZE * RESOLUTION_MULTIPLIER,
                              GRID_SIZE * RESOLUTION_MULTIPLIER))
            mosaic_images.append(img)

            # Calculate the average color of the image
            mosaic_images_color_avg.append(average_color(img))
    print(f'Loaded {len(mosaic_images)} mosaic images.')
    print('========')

    print("Processing input images...")
    files: list[str] = []
    for input_file in listdir("input"):
        if input_file.endswith((".jpg", ".jpeg")):
            files.append(input_file)

    # TODO: try to use Manager to share the memory with child processes
    #       Right now it *might* be copying both image arrays for every job
    # with multiprocessing.Manager() as mgr:
    #   dictionary = mgr.dict()
    #   dictionary[1] = mosaic_images
    #   dictionary[2] = mosaic_images_color_avg
    #
    # TODO: fix printing progress
    with multiprocessing.Pool(None) as p:
        jobs = [
            p.apply_async(process_image,
                          (file, mosaic_images, mosaic_images_color_avg))
            for file in files
        ]
        for job in jobs:
            job.get()

    print("Finished processing images.")
