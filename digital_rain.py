#!/usr/bin/env python

"""Shows your webcam video - Matrix style"""

import signal
import typing as t
from random import choice, randint
from time import perf_counter

import _curses
import cv2
import numpy as np
import numpy.typing as npt
from mediapipe.python.solutions.selfie_segmentation import SelfieSegmentation

Image = npt.NDArray[np.uint8]
curses: t.Any = _curses  # ignore mypy


class AsciiImage:
    """Turns a numpy image into rich-CLI ascii image"""

    ASCII_CHARS = [" ", "@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]
    bin_size = (255 + len(ASCII_CHARS)) // len(ASCII_CHARS)

    def __init__(self, height: int, width: int):
        self.size = width, height

    def convert(self, image: Image) -> str:

        image = cv2.resize(image, self.size)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return "".join(self.ASCII_CHARS[pixel // self.bin_size] for pixel in gray.flatten())

    def convert_with_linebreak(self, image: Image) -> str:

        image = cv2.resize(image, self.size)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        last_col = size[0] - 1
        ascii_str = ""
        for (_, x), pixel in np.ndenumerate(gray):  # pylint: disable=C0103
            ascii_str += self.ASCII_CHARS[pixel // self.bin_size]
            if x == last_col:
                ascii_str += "\n"
        return ascii_str


class Selfie:
    """
      Draw selfie segmentation on the background image.

     The background can be customized.
       a) Load an image (with the same width and height of the input image)
          to be the background
       b) Blur the input image by applying image filtering, e.g.,
          bg_image = cv2.GaussianBlur(image,(55,55),0)
    """
    def __init__(self):
        self.segmentation = None
        self.bg_image: t.Optional[Image] = None

    def  __enter__(self):
        self.segmentation = SelfieSegmentation(model_selection=1)
        self.segmentation.__enter__()
        return self

    def  __exit__(self, type, value, traceback):
        self.segmentation.__exit__(type, value, traceback)

    def process_image(self, image):
        """Segement human from image and draw on background"""

        # Flip the image horizontally for a later selfie-view display,
        # and convert the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        results = self.segmentation.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.95

        if self.bg_image is None:
            self.bg_image = np.zeros(image.shape, dtype=np.uint8)

        return np.where(condition, image, self.bg_image)


class DigitalRain:
    """TODO"""

    def __init__(self, height, width, letters, probability, updates_per_second):
        self.height = height
        self.width = width
        self.probability = probability
        self.updates_per_second = updates_per_second
        self.letters = letters
        self.color = curses.color_pair(1)

        self.updates = 0
        self.start_time = perf_counter()

        self.characters = (
            [chr(c) for c in range(0x30A0, 0x30FB)]
            + [chr(c) for c in range(ord("0"), ord("9") + 1)]
            + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        )

        # dispense: columns where new droplets appear from.
        # foreground: list of positions of droplets
        # background: matrix of the actual letters (not lit up) -- the underlay.
        self.dispense: list[int] = []
        self.foreground: list[tuple[int, int]] = []
        self.background = self.rand_string()
        self.bg_refresh_counter = randint(3, 7)

    def apply(self, screen):
        """TODO"""

        now = perf_counter()
        self.updates += (now - self.start_time) * self.updates_per_second
        update_matrix = self.updates >= 1
        self.start_time = now

        for index, (row, col) in enumerate(self.foreground):
            # Delete bottom row
            if row >= self.height - 1:
                del self.foreground[index]
                continue

            if update_matrix:
                # Move droplet down
                self.foreground[index] = (row + 1, col)

            # Draw droplet using character from background
            loc = row * self.height + col
            screen.addstr(row, col, self.background[loc], self.color)

        if update_matrix:
            self.updates -= 1

            # Start new droplets
            self.dispense.extend(
                randint(0, self.width - 1) for _ in range(self.letters)
            )
            for index, column in enumerate(self.dispense):
                # Extend droplets
                self.foreground.append((0, column))
                # End droplets
                if not randint(0, self.probability - 1):
                    del self.dispense[index]

        self.bg_refresh_counter -= 1
        if self.bg_refresh_counter <= 0:
            self.background = self.rand_string()
            self.bg_refresh_counter = randint(3, 7)

    def rand_string(self) -> str:
        """
        Returns a random string.
        character_set -- the characters to choose from.
        length        -- the length of the string.
        """
        length = self.height * self.width
        return "".join(choice(self.characters) for _ in range(length))


def init_curses() -> t.Any:
    """Initializes curses library"""

    screen = curses.initscr()
    screen.keypad(True)  # if not set will end program on arrow keys etc
    screen.nodelay(True)  # Don't block waiting for input.

    curses.curs_set(False)  # no blinking cursor
    curses.noecho()  # do not echo keypress
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)


    return screen


def terminate_curses(screen: t.Any) -> None:
    screen.keypad(False)
    curses.echo()
    curses.endwin()


def terminate_camera(stream: t.Any) -> None:
    if stream:
        stream.release()


def process_camera(stream, ascii_image, screen, selfie):
    success, image = stream.read()
    if not success:
        return

    if selfie:
        # Segment human image
        image = selfie.process_image(image)

    # Convert image to ascii
    string = ascii_image.convert(image)

    screen.clear()
    color = curses.color_pair(1)

    # Write image to screen
    width = ascii_image.size[0]
    for index, val in enumerate(string[:-1]):
        if val:
            screen.addstr(index // width, index % width, val, color)


def main(
    camera: int = 0,  # help="The camera index.",

    letters: int = 2,  # help="Characters produced per update.",
    probability: int = 5,  # help="1/p probability of a droplet deactivating each tick.",
    updates_per_second: int = 15,  # help="The number of updates to perform per second.",

    use_camera: bool = True,
    segment: bool = True,
    rain: bool = True,
) -> None:
    """Main loop."""

    stream = None
    ascii_image = None
    digital_rain = None

    screen = init_curses()
    signal.signal(signal.SIGINT, lambda signal, frame: terminate_curses(screen))
    height, width = screen.getmaxyx()

    if use_camera:
        ascii_image = AsciiImage(height, width)
        stream = cv2.VideoCapture(camera)
        if not stream.isOpened():
            print("No VideoCapture found!")
            stream.release()
            return

    if rain:
        digital_rain = DigitalRain(height, width, letters, probability, updates_per_second)

    with Selfie() as selfie:
        while True:
            if stream:
                if not stream.isOpened():
                    break
                process_camera(stream, ascii_image, screen, selfie if segment else None)

            if digital_rain:
                digital_rain.apply(screen)

            screen.refresh()
            if screen.getch() in (3, 27):  # ESC pressed
                break

    terminate_camera(stream)
    terminate_curses(screen)


if __name__ == "__main__":
    main()
