"""Generates the NeuronCite application icon as a geometric "C" with the
brand's purple-to-cyan gradient on a transparent background.

The "C" is a bold arc (ring section with an opening on the right side)
with rounded end caps, matching the design language of the "Cite" gradient
used in the splash screen and web frontend. The gradient uses the brand
colors #a855f7 (purple) and #22d3ee (cyan) at 135 degrees, centered on
the C shape's bounding box so the exact midpoint color falls at the
geometric center of the icon.

Output files:
  - crates/neuroncite/assets/icon.ico          Windows executable icon (16..256px)
  - crates/neuroncite/assets/icon.icns         macOS .app bundle icon (128..1024px)
  - crates/neuroncite-web/assets/icon_256.png   tao window icon (embedded via include_bytes!)
  - crates/neuroncite-web/frontend/public/favicon.ico       Browser tab icon
  - crates/neuroncite-web/frontend/public/favicon-32x32.png Browser tab icon (PNG)

Run from the repository root:
    python tools/gen/generate_icon.py

The output files are committed to the repository. The .ico is embedded into
the Windows executable at link time via winres. The .icns is copied into the
macOS .app bundle's Resources directory by the release workflow. The 256x256
PNG is embedded into the Rust binary via include_bytes! for the tao window icon.
"""

import io
import math
import struct
import sys
from pathlib import Path

from PIL import Image, ImageDraw

# --- Brand colors from the Dark Neon Tech design system ---

COLOR_PURPLE = (168, 85, 247)   # #a855f7 -- gradient start
COLOR_CYAN = (34, 211, 238)     # #22d3ee -- gradient end

# --- Geometry parameters for the "C" shape ---
# All values are fractions of the render canvas size.

# Outer and inner radii define the stroke width of the arc.
# outer_r = 0.44 and inner_r = 0.27 give a stroke width of 0.17,
# which is bold enough to read at 16x16 while remaining balanced at 256x256.
OUTER_RADIUS_FRAC = 0.44
INNER_RADIUS_FRAC = 0.27

# Half of the angular opening on the right side of the "C", in degrees.
# 50 degrees on each side = 100 degree total opening, producing a classic
# "C" shape that is recognizable at all icon sizes.
OPENING_HALF_ANGLE_DEG = 50.0

# Render resolution for the master image. Supersampled at this resolution
# and downscaled to target sizes with LANCZOS resampling for clean edges.
RENDER_SIZE = 2048

# Target icon sizes for multi-resolution .ico and individual PNGs.
ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]

# ICNS type codes for PNG-based entries in the Apple Icon Image format.
# macOS 10.7+ accepts raw PNG data as the payload for these type codes.
# Each tuple maps a 4-byte type code to the required pixel resolution.
ICNS_TYPES = [
    (b"ic07", 128),    # 128x128
    (b"ic08", 256),    # 256x256
    (b"ic09", 512),    # 512x512
    (b"ic10", 1024),   # 1024x1024 (512x512@2x)
]


def build_c_mask(size: int) -> Image.Image:
    """Builds a grayscale mask of the "C" shape at the given pixel resolution.

    The mask is white (255) where the C shape is, and black (0) elsewhere.
    The shape consists of:
      1. A thick ring (outer circle minus inner circle)
      2. A wedge cut out on the right side for the opening
      3. Two semicircular caps at the arc endpoints for rounded ends

    Anti-aliasing comes from rendering at RENDER_SIZE and downscaling later.
    """
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)

    cx = size / 2.0
    cy = size / 2.0
    outer_r = size * OUTER_RADIUS_FRAC
    inner_r = size * INNER_RADIUS_FRAC

    # Step 1: Draw the outer filled circle
    draw.ellipse(
        [cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r],
        fill=255,
    )

    # Step 2: Cut out the inner circle to create a ring
    draw.ellipse(
        [cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r],
        fill=0,
    )

    # Step 3: Cut out the wedge on the right side.
    # The wedge is a polygon extending from the center to well beyond the
    # outer circle at angles +/- OPENING_HALF_ANGLE_DEG from horizontal right.
    angle_rad = math.radians(OPENING_HALF_ANGLE_DEG)
    far = size * 2.0  # distance beyond the circle for the wedge polygon

    # Polygon vertices: center, then sweep along the opening from top to bottom
    wedge_points = [
        (cx, cy),
        (cx + far * math.cos(angle_rad), cy - far * math.sin(angle_rad)),
        (cx + far, cy - far),   # far top-right corner
        (cx + far, cy + far),   # far bottom-right corner
        (cx + far * math.cos(angle_rad), cy + far * math.sin(angle_rad)),
    ]
    draw.polygon(wedge_points, fill=0)

    # Step 4: Add rounded end caps at the two arc endpoints.
    # Each cap is a filled circle centered on the midpoint of the stroke
    # at the end angle, with radius equal to half the stroke width.
    stroke_w = outer_r - inner_r
    mid_r = (outer_r + inner_r) / 2.0
    cap_r = stroke_w / 2.0

    # Top endpoint: at +OPENING_HALF_ANGLE_DEG from horizontal right
    cap_top_x = cx + mid_r * math.cos(angle_rad)
    cap_top_y = cy - mid_r * math.sin(angle_rad)
    draw.ellipse(
        [cap_top_x - cap_r, cap_top_y - cap_r,
         cap_top_x + cap_r, cap_top_y + cap_r],
        fill=255,
    )

    # Bottom endpoint: at -OPENING_HALF_ANGLE_DEG from horizontal right
    cap_bot_x = cx + mid_r * math.cos(angle_rad)
    cap_bot_y = cy + mid_r * math.sin(angle_rad)
    draw.ellipse(
        [cap_bot_x - cap_r, cap_bot_y - cap_r,
         cap_bot_x + cap_r, cap_bot_y + cap_r],
        fill=255,
    )

    return img


def build_gradient(size: int, mask: Image.Image) -> Image.Image:
    """Creates an RGBA gradient centered on the C shape's bounding box.

    The gradient runs at 135 degrees (top-left to bottom-right). The exact
    midpoint of the purple-to-cyan transition falls at the geometric center
    of the C shape's bounding box, so the color distribution is symmetric
    around the visual center of the icon.

    The mask parameter is used to compute the tight bounding box of the C
    shape. The gradient interpolation factor is normalized to this bounding
    box: 0.0 at the top-left corner, 1.0 at the bottom-right corner, and
    0.5 at the center -- where the exact middle color between purple and
    cyan appears.
    """
    # Compute the bounding box of the C shape from the mask. getbbox()
    # returns (left, top, right, bottom) of the non-zero region.
    bbox = mask.getbbox()
    if bbox is None:
        # Fallback: full canvas (should not happen with a valid mask)
        bbox = (0, 0, size, size)

    bb_left, bb_top, bb_right, bb_bottom = bbox
    bb_width = bb_right - bb_left
    bb_height = bb_bottom - bb_top

    gradient = Image.new("RGBA", (size, size))
    pixels = gradient.load()

    for y in range(size):
        # Normalized position within the C bounding box (0.0 at top edge,
        # 1.0 at bottom edge). Values outside the bounding box extend
        # beyond 0..1 and are clamped, pushing the endpoints further into
        # pure purple or pure cyan.
        ny = (y - bb_top) / bb_height if bb_height > 0 else 0.5
        for x in range(size):
            nx = (x - bb_left) / bb_width if bb_width > 0 else 0.5

            # 135-degree diagonal projection: equal weight of horizontal
            # and vertical position. At the center of the bounding box
            # (nx=0.5, ny=0.5), diag = 0.5 -- the exact midpoint.
            diag = nx * 0.5 + ny * 0.5
            t = max(0.0, min(1.0, diag))

            r = int(COLOR_PURPLE[0] + (COLOR_CYAN[0] - COLOR_PURPLE[0]) * t)
            g = int(COLOR_PURPLE[1] + (COLOR_CYAN[1] - COLOR_PURPLE[1]) * t)
            b = int(COLOR_PURPLE[2] + (COLOR_CYAN[2] - COLOR_PURPLE[2]) * t)
            pixels[x, y] = (r, g, b, 255)

    return gradient


def render_icon(size: int) -> Image.Image:
    """Renders the final RGBA icon at the requested pixel size.

    The icon is rendered at RENDER_SIZE resolution for quality, then
    downscaled to the target size with LANCZOS resampling. The gradient
    is applied through the C-shape mask to produce the final result.
    """
    # Build the C-shape mask at high resolution
    mask = build_c_mask(RENDER_SIZE)

    # Build the gradient centered on the C shape's bounding box
    gradient = build_gradient(RENDER_SIZE, mask)

    # Apply the mask: gradient pixels where the C is, transparent elsewhere
    result = Image.new("RGBA", (RENDER_SIZE, RENDER_SIZE), (0, 0, 0, 0))
    result.paste(gradient, mask=mask)

    # Downscale to target size with LANCZOS for clean anti-aliased edges
    if size != RENDER_SIZE:
        result = result.resize((size, size), Image.LANCZOS)

    return result


def build_icns(sized_images: dict[int, Image.Image]) -> bytes:
    """Constructs an ICNS file (Apple Icon Image format) from pre-rendered images.

    The ICNS binary format consists of a file header followed by a sequence of
    icon entries. The file header is 8 bytes: a 4-byte magic number ('icns')
    and a 4-byte big-endian integer for the total file size. Each entry has an
    8-byte header (4-byte type code + 4-byte entry size including the header)
    followed by the raw PNG data as payload. macOS 10.7+ reads PNG payloads
    directly for the ic07/ic08/ic09/ic10 type codes.
    """
    entries = b""
    for type_code, size in ICNS_TYPES:
        buf = io.BytesIO()
        sized_images[size].save(buf, format="PNG", optimize=True)
        png_data = buf.getvalue()
        # Entry: 4-byte type code + 4-byte entry size (header + payload) + PNG data
        entry_size = 8 + len(png_data)
        entries += type_code + struct.pack(">I", entry_size) + png_data

    # File header: 4-byte magic 'icns' + 4-byte total file size (header + all entries)
    total_size = 8 + len(entries)
    return b"icns" + struct.pack(">I", total_size) + entries


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent

    # Output paths
    ico_path = repo_root / "crates" / "neuroncite" / "assets" / "icon.ico"
    icns_path = repo_root / "crates" / "neuroncite" / "assets" / "icon.icns"
    png_256_path = repo_root / "crates" / "neuroncite-web" / "assets" / "icon_256.png"
    favicon_ico_path = repo_root / "crates" / "neuroncite-web" / "frontend" / "public" / "favicon.ico"
    favicon_png_path = repo_root / "crates" / "neuroncite-web" / "frontend" / "public" / "favicon-32x32.png"

    # Ensure output directories exist
    for path in [ico_path, icns_path, png_256_path, favicon_ico_path, favicon_png_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Rendering master icon at {RENDER_SIZE}x{RENDER_SIZE}...")

    # Pre-render the high-resolution master and cache it for reuse
    master_mask = build_c_mask(RENDER_SIZE)
    master_gradient = build_gradient(RENDER_SIZE, master_mask)
    master_rgba = Image.new("RGBA", (RENDER_SIZE, RENDER_SIZE), (0, 0, 0, 0))
    master_rgba.paste(master_gradient, mask=master_mask)

    # Collect all required sizes: ICO sizes + ICNS sizes (128, 256, 512, 1024)
    # + standalone PNGs (256, 32). Deduplicated and sorted.
    icns_sizes = [size for _, size in ICNS_TYPES]
    all_sizes = sorted(set(ICO_SIZES + icns_sizes + [256, 32]))

    # Generate all target sizes by downscaling the master
    sized_images = {}
    for target_size in all_sizes:
        print(f"  Downscaling to {target_size}x{target_size}...")
        sized_images[target_size] = master_rgba.resize(
            (target_size, target_size), Image.LANCZOS
        )

    # --- Save 256x256 PNG for tao window icon ---
    sized_images[256].save(str(png_256_path), "PNG", optimize=True)
    print(f"Window icon PNG saved to: {png_256_path}")

    # --- Save 32x32 PNG favicon ---
    sized_images[32].save(str(favicon_png_path), "PNG", optimize=True)
    print(f"Favicon PNG saved to: {favicon_png_path}")

    # --- Save multi-resolution .ico for Windows executable ---
    # PIL's ICO writer requires the largest image as the base, with smaller
    # sizes passed via append_images. The `sizes` parameter must NOT be used
    # together with `append_images` -- PIL ignores appended images when `sizes`
    # is present and only writes the base image resized, which produces a
    # single-entry ICO (the root cause of the blurry Windows taskbar icon).
    # By omitting `sizes` and using only `append_images`, each pre-rendered
    # downscale is written at its native resolution into the ICO directory.
    ico_base = sized_images[ICO_SIZES[-1]]  # 256px -- largest entry
    ico_rest = [sized_images[s] for s in ICO_SIZES[:-1]]  # 16..128px
    ico_base.save(
        str(ico_path),
        format="ICO",
        append_images=ico_rest,
    )
    print(f"Windows .ico saved to: {ico_path}")

    # --- Save macOS .icns for the .app bundle ---
    # The ICNS file contains PNG payloads at 128, 256, 512, and 1024 pixels.
    # It is copied into NeuronCite.app/Contents/Resources/AppIcon.icns by the
    # GitHub Actions release workflow during .app bundle assembly.
    icns_data = build_icns(sized_images)
    icns_path.write_bytes(icns_data)
    print(f"macOS .icns saved to: {icns_path}")

    # --- Save browser favicon.ico (16, 32, 48) ---
    # Same approach as the main .ico: largest image as base, smaller sizes
    # appended. No `sizes` parameter to avoid PIL silently dropping entries.
    favicon_sizes = [16, 32, 48]
    favicon_base = sized_images[favicon_sizes[-1]]  # 48px
    favicon_rest = [sized_images[s] for s in favicon_sizes[:-1]]  # 16, 32
    favicon_base.save(
        str(favicon_ico_path),
        format="ICO",
        append_images=favicon_rest,
    )
    print(f"Browser favicon.ico saved to: {favicon_ico_path}")

    print("Icon generation complete.")


if __name__ == "__main__":
    main()
