"""Generates the transparent NeuronCite splash screen PNG.

Renders "NeuronCite" text on a fully transparent RGBA canvas with:
  - "Neuron" in bright white (#e8e8f0)
  - "Cite" with a purple-to-cyan diagonal gradient (#a855f7 -> #22d3ee)
  - Gaussian-blurred glow layer behind the text in purple, 40% opacity
  - A subtle loading bar below the text (static gradient line)

Output: crates/neuroncite-web/assets/splash.png
Canvas: 960x360 pixels (2x HiDPI, displays at 480x180 logical pixels)
Font:   Segoe UI Black (seguibl.ttf) at 104px, falling back to Segoe UI Bold

Run this script from the repository root:
    python tools/generate_splash.py

The output PNG is committed to the repository and embedded into the Rust
binary via include_bytes! at compile time.
"""

import math
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# --- Configuration ---

# Canvas dimensions at 2x for HiDPI displays.
# The tao window is 480x180 logical pixels.
WIDTH = 960
HEIGHT = 360

# Brand colors from the Dark Neon Tech design system
COLOR_WHITE = (232, 232, 240)       # #e8e8f0 -- "Neuron" text color
COLOR_PURPLE = (168, 85, 247)       # #a855f7 -- gradient start, glow color
COLOR_CYAN = (34, 211, 238)         # #22d3ee -- gradient end

# Font size at 2x scale (52px logical * 2 = 104px)
FONT_SIZE = 104
LETTER_SPACING = -0.02              # em units, matching the HTML splash

# Glow parameters
GLOW_BLUR_RADIUS = 16               # pixels at 2x
GLOW_OPACITY = 0.4                  # 40% opacity for the purple glow layer
GLOW_SPREAD = 2                     # number of blur passes for a softer spread

# Loading bar parameters (static gradient line below the text)
BAR_WIDTH = 240                     # pixels at 2x (120px logical)
BAR_HEIGHT = 4                      # pixels at 2x (2px logical)
BAR_Y_OFFSET = 40                   # pixels below the text baseline at 2x


def find_font() -> ImageFont.FreeTypeFont:
    """Locates the Segoe UI Black or Bold font on the system.

    Tries Segoe UI Black (weight 900) first for the heaviest strokes,
    then falls back to Segoe UI Bold (weight 700), then to the
    default Pillow font as a last resort.
    """
    fonts_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    candidates = [
        os.path.join(fonts_dir, "seguibl.ttf"),   # Segoe UI Black (900)
        os.path.join(fonts_dir, "segoeuib.ttf"),   # Segoe UI Bold (700)
        os.path.join(fonts_dir, "seguisb.ttf"),    # Segoe UI Semibold (600)
        # Linux / macOS fallbacks
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, FONT_SIZE)

    print("WARNING: No suitable font found, using Pillow default", file=sys.stderr)
    return ImageFont.load_default()


def draw_text_with_spacing(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: float,
    y: float,
    font: ImageFont.FreeTypeFont,
    fill: tuple,
) -> float:
    """Draws text character by character with custom letter spacing.

    Returns the total width of the rendered text in pixels. The letter
    spacing is applied as a fraction of the font size (em units) between
    each character pair.
    """
    spacing_px = LETTER_SPACING * FONT_SIZE
    cursor_x = x
    for char in text:
        draw.text((cursor_x, y), char, font=font, fill=fill)
        bbox = font.getbbox(char)
        char_width = bbox[2] - bbox[0]
        cursor_x += char_width + spacing_px
    return cursor_x - x


def measure_text_with_spacing(
    text: str,
    font: ImageFont.FreeTypeFont,
) -> float:
    """Measures the total width of text with custom letter spacing applied.

    Does not draw anything -- used for layout calculations to center
    the composite "NeuronCite" text on the canvas.
    """
    spacing_px = LETTER_SPACING * FONT_SIZE
    total = 0.0
    for i, char in enumerate(text):
        bbox = font.getbbox(char)
        char_width = bbox[2] - bbox[0]
        total += char_width
        if i < len(text) - 1:
            total += spacing_px
    return total


def create_gradient_mask_fast(
    text: str,
    font: ImageFont.FreeTypeFont,
    text_x: float,
    text_y: float,
    canvas_width: int,
    canvas_height: int,
) -> Image.Image:
    """Creates a gradient-filled text image matching the CSS gradient from the GUI.

    The frontend CSS applies linear-gradient(135deg, #a855f7 30%, #22d3ee 100%)
    to the <span>Cite</span> element. The gradient is relative to the text
    bounding box, not the full canvas. This function replicates that behavior:
    the first 30% of the 135-degree diagonal stays solid purple, then
    transitions linearly to cyan over the remaining 70%.
    """
    # Render text as a white-on-transparent mask
    mask_img = Image.new("L", (canvas_width, canvas_height), 0)
    mask_draw = ImageDraw.Draw(mask_img)
    spacing_px = LETTER_SPACING * FONT_SIZE
    cursor_x = text_x
    for char in text:
        mask_draw.text((cursor_x, text_y), char, font=font, fill=255)
        bbox = font.getbbox(char)
        char_width = bbox[2] - bbox[0]
        cursor_x += char_width + spacing_px

    # Compute the text bounding box so the gradient is relative to "Cite",
    # not the full 960x360 canvas. This matches CSS background-clip: text
    # where the gradient origin is the element's own box.
    text_w = measure_text_with_spacing(text, font)
    ascent, descent = font.getmetrics()
    text_h = ascent + descent
    text_left = text_x
    text_top = text_y

    # Generate the gradient across the full canvas, but compute the diagonal
    # interpolation factor relative to the "Cite" bounding box.
    # CSS linear-gradient(135deg) runs from top-left to bottom-right at 135 degrees.
    # The normalized diagonal position within the text box determines the color.
    # CSS color stops: purple at 30%, cyan at 100% -- meaning pure purple for
    # the first 30% of the diagonal, then a linear ramp to cyan.
    gradient = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
    for y_pos in range(canvas_height):
        # Normalized vertical position within the text bounding box (0..1)
        ny = (y_pos - text_top) / text_h if text_h > 0 else 0.0
        row_data = []
        for x_pos in range(canvas_width):
            # Normalized horizontal position within the text bounding box (0..1)
            nx = (x_pos - text_left) / text_w if text_w > 0 else 0.0
            # 135-degree diagonal factor: equal weight of x and y
            diag = nx * 0.5 + ny * 0.5
            diag = max(0.0, min(1.0, diag))

            # CSS color stops: purple at 30%, cyan at 100%
            # Before 30%: solid purple. After 30%: linear ramp to cyan.
            if diag <= 0.3:
                t = 0.0
            else:
                t = (diag - 0.3) / 0.7

            r = int(COLOR_PURPLE[0] + (COLOR_CYAN[0] - COLOR_PURPLE[0]) * t)
            g = int(COLOR_PURPLE[1] + (COLOR_CYAN[1] - COLOR_PURPLE[1]) * t)
            b = int(COLOR_PURPLE[2] + (COLOR_CYAN[2] - COLOR_PURPLE[2]) * t)
            row_data.extend([r, g, b, 255])
        gradient.paste(
            Image.frombytes("RGBA", (canvas_width, 1), bytes(row_data)),
            (0, y_pos),
        )

    # Apply the text mask to the gradient
    result = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
    result.paste(gradient, mask=mask_img)
    return result


def create_loading_bar(canvas_width: int, center_y: int) -> Image.Image:
    """Creates a static gradient loading bar (purple to cyan) centered horizontally.

    The bar is a thin horizontal line with the brand gradient, providing
    a visual cue that the application is starting. Unlike the animated
    HTML version, this is a static element in the PNG.
    """
    bar = Image.new("RGBA", (canvas_width, BAR_HEIGHT), (0, 0, 0, 0))
    bar_draw = ImageDraw.Draw(bar)
    for x_pos in range(BAR_WIDTH):
        t = x_pos / BAR_WIDTH
        r = int(COLOR_PURPLE[0] + (COLOR_CYAN[0] - COLOR_PURPLE[0]) * t)
        g = int(COLOR_PURPLE[1] + (COLOR_CYAN[1] - COLOR_PURPLE[1]) * t)
        b = int(COLOR_PURPLE[2] + (COLOR_CYAN[2] - COLOR_PURPLE[2]) * t)
        bar_start_x = (canvas_width - BAR_WIDTH) // 2
        bar_draw.line(
            [(bar_start_x + x_pos, 0), (bar_start_x + x_pos, BAR_HEIGHT - 1)],
            fill=(r, g, b, 255),
        )
    return bar


def main():
    repo_root = Path(__file__).resolve().parent.parent
    output_path = repo_root / "crates" / "neuroncite-web" / "assets" / "splash.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    font = find_font()

    # Measure both parts of the logo to center the composite text
    neuron_width = measure_text_with_spacing("Neuron", font)
    cite_width = measure_text_with_spacing("Cite", font)
    total_width = neuron_width + cite_width

    # Center horizontally on the canvas
    start_x = (WIDTH - total_width) / 2

    # Center vertically -- use the font ascent/descent for precise placement
    ascent, descent = font.getmetrics()
    text_height = ascent + descent
    start_y = (HEIGHT - text_height) / 2 - BAR_Y_OFFSET / 2

    # --- Layer 1: Glow ---
    # Render the full "NeuronCite" text in purple, blur it, reduce opacity.
    # Multiple blur passes create a softer, wider glow.
    glow = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_color = (*COLOR_PURPLE, 255)
    draw_text_with_spacing(glow_draw, "NeuronCite", start_x, start_y, font, glow_color)
    for _ in range(GLOW_SPREAD):
        glow = glow.filter(ImageFilter.GaussianBlur(radius=GLOW_BLUR_RADIUS))

    # Reduce glow opacity to 40%
    glow_data = glow.load()
    for y_pos in range(HEIGHT):
        for x_pos in range(WIDTH):
            r, g, b, a = glow_data[x_pos, y_pos]
            if a > 0:
                glow_data[x_pos, y_pos] = (r, g, b, int(a * GLOW_OPACITY))

    # --- Layer 2: "Neuron" text in white ---
    neuron_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    neuron_draw = ImageDraw.Draw(neuron_layer)
    draw_text_with_spacing(
        neuron_draw, "Neuron", start_x, start_y, font, (*COLOR_WHITE, 255)
    )

    # --- Layer 3: "Cite" text with gradient ---
    cite_x = start_x + neuron_width
    cite_layer = create_gradient_mask_fast(
        "Cite", font, cite_x, start_y, WIDTH, HEIGHT
    )

    # --- Layer 4: Loading bar ---
    bar_y = int(start_y + text_height + BAR_Y_OFFSET)
    bar_layer = create_loading_bar(WIDTH, bar_y)

    # --- Composite all layers ---
    canvas = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    canvas = Image.alpha_composite(canvas, glow)
    canvas = Image.alpha_composite(canvas, neuron_layer)
    canvas = Image.alpha_composite(canvas, cite_layer)

    # Paste the loading bar at the correct vertical position
    bar_canvas = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    bar_canvas.paste(bar_layer, (0, bar_y))
    canvas = Image.alpha_composite(canvas, bar_canvas)

    canvas.save(str(output_path), "PNG", optimize=True)
    print(f"Splash PNG saved to: {output_path}")
    print(f"Dimensions: {WIDTH}x{HEIGHT} (displays at {WIDTH // 2}x{HEIGHT // 2} logical pixels)")


if __name__ == "__main__":
    main()
