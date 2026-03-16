// NeuronCite -- local, privacy-preserving semantic document search engine.
// Copyright (C) 2026 NeuronCite Contributors
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

//! Platform-specific transparent splash screen rendering for the NeuronCite
//! startup sequence.
//!
//! Decodes an embedded PNG (960x360 RGBA, pre-rendered by tools/gen/generate_splash.py)
//! and paints it onto a tao window using native OS APIs so the desktop is visible
//! behind the logo text. The PNG contains the "NeuronCite" brand text with a
//! purple-to-cyan gradient on "Cite", gaussian blur glow, and fully transparent
//! background.
//!
//! Platform rendering:
//! - **Windows**: Win32 layered window -- `UpdateLayeredWindow` with premultiplied
//!   BGRA pixel data in a 32-bit DIB section. The window receives `WS_EX_LAYERED`
//!   after creation, and the bitmap is applied as a one-shot operation (no WM_PAINT).
//! - **macOS**: `NSImage` loaded from the raw PNG bytes, displayed in an `NSImageView`
//!   added to the `NSWindow`'s content view. The window is set to non-opaque with a
//!   clear background color. AppKit handles Retina 2x scaling automatically.
//! - **Linux**: Cairo `ImageSurface` from premultiplied ARGB data, painted in a GTK
//!   `draw` signal handler. The window uses an RGBA visual and `app_paintable` mode
//!   (both set by tao's `with_transparent(true)` + `with_transparent_draw(false)`).
//!
//! This module is compiled only when the `gui` feature flag is enabled.

use tao::window::Window;

/// Embedded splash screen PNG (960x360 pixels, RGBA, pre-rendered by
/// tools/gen/generate_splash.py). Contains the "NeuronCite" logo text with
/// purple-to-cyan gradient on "Cite", gaussian blur glow, and fully
/// transparent background. The 2x resolution provides crisp rendering
/// on HiDPI / Retina displays.
const SPLASH_PNG: &[u8] = include_bytes!("../assets/splash.png");

/// Decodes the embedded PNG to an RGBA pixel buffer using the `image` crate.
///
/// Returns a tuple of (pixels, width, height) where pixels is a Vec<u8> in
/// row-major RGBA byte order (4 bytes per pixel: R, G, B, A).
///
/// Panics if the embedded PNG is corrupted or not a valid image (this is a
/// compile-time-embedded asset, so corruption indicates a build toolchain
/// problem).
fn decode_png() -> (Vec<u8>, u32, u32) {
    let img = image::load_from_memory(SPLASH_PNG)
        .expect("embedded splash.png is a valid PNG")
        .into_rgba8();
    let (w, h) = img.dimensions();
    (img.into_raw(), w, h)
}

/// Renders the transparent splash PNG onto the given tao window using
/// platform-native APIs. After this call, the window displays the PNG
/// with per-pixel alpha transparency -- the desktop is visible behind
/// areas where the PNG alpha channel is zero.
///
/// This function runs on the main thread after the window is created and
/// before the event loop starts processing events.
///
/// Returns Ok(()) on success, or Err with a description if the platform
/// rendering fails. The caller logs a warning on error but does not abort
/// the startup sequence -- the window remains with whatever default
/// background the platform provides for transparent windows.
pub fn render_splash_on_window(window: &Window) -> Result<(), String> {
    let (rgba_pixels, width, height) = decode_png();
    render_platform(window, &rgba_pixels, width, height)
}

// ---------------------------------------------------------------------------
// Windows: Win32 layered window with UpdateLayeredWindow
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
fn render_platform(window: &Window, rgba: &[u8], width: u32, height: u32) -> Result<(), String> {
    use image::RgbaImage;
    use image::imageops::FilterType;
    use tao::platform::windows::WindowExtWindows;
    use windows::Win32::Foundation::{COLORREF, HWND, POINT, SIZE};
    use windows::Win32::Graphics::Gdi::{
        AC_SRC_ALPHA, AC_SRC_OVER, BI_RGB, BITMAPINFO, BITMAPINFOHEADER, BLENDFUNCTION,
        CreateCompatibleDC, CreateDIBSection, DIB_RGB_COLORS, DeleteDC, DeleteObject, GetDC,
        ReleaseDC, SelectObject,
    };
    use windows::Win32::UI::WindowsAndMessaging::{
        GWL_EXSTYLE, GetWindowLongPtrW, SetWindowLongPtrW, ULW_ALPHA, UpdateLayeredWindow,
        WS_EX_LAYERED,
    };

    // Get the physical pixel dimensions of the window. On standard DPI (100%),
    // the 480x180 logical window maps to 480x180 physical. On 200% DPI, it
    // maps to 960x360 physical -- matching the PNG exactly. For intermediate
    // DPI values (125%, 150%), the physical size falls between those extremes.
    let phys_size = window.inner_size();
    let target_w = phys_size.width;
    let target_h = phys_size.height;

    // Resize the decoded PNG to the physical window dimensions. The source PNG
    // is 960x360 (2x for HiDPI). UpdateLayeredWindow does not scale the bitmap --
    // it reads pixels 1:1 from the DIB. So the bitmap must match the physical
    // window size exactly, otherwise the image gets clipped (at 100% DPI the
    // window is only 480x180 physical pixels, but the PNG is 960x360).
    // CatmullRom (bicubic) provides sharp downscaling for text.
    let (final_rgba, final_w, final_h) = if target_w == width && target_h == height {
        // At 200% DPI the physical size matches the PNG exactly, no resize needed.
        (rgba.to_vec(), width, height)
    } else {
        let src = RgbaImage::from_raw(width, height, rgba.to_vec())
            .ok_or("failed to construct RgbaImage from decoded PNG")?;
        let resized = image::imageops::resize(&src, target_w, target_h, FilterType::CatmullRom);
        let (rw, rh) = resized.dimensions();
        (resized.into_raw(), rw, rh)
    };

    // Get the native HWND from the tao window via the platform extension trait.
    // The trait returns isize; HWND wraps a raw pointer.
    let hwnd_raw = window.hwnd();
    let hwnd = HWND(hwnd_raw as *mut _);

    // Convert RGBA (PNG byte order) to BGRA with premultiplied alpha.
    // Win32's UpdateLayeredWindow requires BGRA byte order with each color
    // channel pre-multiplied by the alpha value: C' = C * A / 255.
    // This is the standard ARGB32 format for GDI layered window operations.
    let pixel_count = (final_w * final_h) as usize;
    let mut bgra = Vec::with_capacity(pixel_count * 4);
    for pixel in final_rgba.chunks_exact(4) {
        let (r, g, b, a) = (
            pixel[0] as u32,
            pixel[1] as u32,
            pixel[2] as u32,
            pixel[3] as u32,
        );
        bgra.push(((b * a / 255) & 0xFF) as u8); // Blue, premultiplied
        bgra.push(((g * a / 255) & 0xFF) as u8); // Green, premultiplied
        bgra.push(((r * a / 255) & 0xFF) as u8); // Red, premultiplied
        bgra.push(a as u8); // Alpha, unchanged
    }

    unsafe {
        // Add WS_EX_LAYERED to the window's extended style flags. This
        // enables the UpdateLayeredWindow API for per-pixel alpha rendering.
        // tao's with_transparent(true) uses DWM blur on Windows, which is
        // a different transparency mechanism. WS_EX_LAYERED provides direct
        // control over the pixel buffer including per-pixel alpha blending.
        let ex_style = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
        SetWindowLongPtrW(hwnd, GWL_EXSTYLE, ex_style | WS_EX_LAYERED.0 as isize);

        // Create a memory device context compatible with the screen.
        // The memory DC holds the 32-bit DIB section that contains the
        // premultiplied BGRA pixel data for UpdateLayeredWindow.
        let screen_dc = GetDC(Some(hwnd));
        let mem_dc = CreateCompatibleDC(Some(screen_dc));

        // Create a 32-bit top-down DIB section. The negative biHeight
        // specifies top-down row order (first row at the top of the image),
        // matching the PNG's row-major pixel layout. Without the negative
        // height, the bitmap would be bottom-up and the image would be
        // vertically flipped.
        let bmi = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: final_w as i32,
                biHeight: -(final_h as i32),
                biPlanes: 1,
                biBitCount: 32,
                biCompression: BI_RGB.0,
                biSizeImage: 0,
                biXPelsPerMeter: 0,
                biYPelsPerMeter: 0,
                biClrUsed: 0,
                biClrImportant: 0,
            },
            bmiColors: [Default::default()],
        };

        let mut bits_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let dib = CreateDIBSection(Some(mem_dc), &bmi, DIB_RGB_COLORS, &mut bits_ptr, None, 0)
            .map_err(|e| format!("CreateDIBSection failed: {e}"))?;

        // Copy the premultiplied BGRA pixels into the DIB section buffer.
        // The DIB section allocated by Windows has the same layout as our
        // bgra Vec (width * height * 4 bytes, top-down row order).
        std::ptr::copy_nonoverlapping(bgra.as_ptr(), bits_ptr as *mut u8, bgra.len());

        // Select the DIB into the memory DC. SelectObject returns the
        // previously selected object which must be restored before
        // deleting the DC (GDI resource management protocol).
        let old_bitmap = SelectObject(mem_dc, dib.into());

        // Apply the per-pixel alpha bitmap to the layered window.
        // UpdateLayeredWindow is a one-shot operation -- the bitmap becomes
        // the window's visual representation. The window does not receive
        // WM_PAINT messages and requires no render loop.
        //
        // The SIZE parameter controls the visual output dimensions. We use
        // the target (physical) window size so the bitmap scales correctly
        // on all DPI settings.
        let size = SIZE {
            cx: target_w as i32,
            cy: target_h as i32,
        };
        let src_point = POINT { x: 0, y: 0 };
        let blend = BLENDFUNCTION {
            BlendOp: AC_SRC_OVER as u8,
            BlendFlags: 0,
            SourceConstantAlpha: 255,
            AlphaFormat: AC_SRC_ALPHA as u8,
        };

        UpdateLayeredWindow(
            hwnd,
            Some(screen_dc),
            None, // pptDst: keep existing window position
            Some(&size),
            Some(mem_dc),
            Some(&src_point),
            COLORREF(0), // crKey: unused with ULW_ALPHA
            Some(&blend),
            ULW_ALPHA,
        )
        .map_err(|e| format!("UpdateLayeredWindow failed: {e}"))?;

        // Clean up GDI resources. The bitmap data has been copied to the
        // window surface by UpdateLayeredWindow, so these objects are no
        // longer needed.
        SelectObject(mem_dc, old_bitmap);
        let _ = DeleteObject(dib.into());
        let _ = DeleteDC(mem_dc);
        ReleaseDC(Some(hwnd), screen_dc);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// macOS: NSWindow transparency + NSImageView with the PNG
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
fn render_platform(window: &Window, _rgba: &[u8], _width: u32, _height: u32) -> Result<(), String> {
    use objc2::rc::Retained;
    // AnyThread provides alloc() for types that can be allocated on any thread
    // (NSImage). MainThreadOnly provides alloc(mtm) for types that require
    // main-thread allocation (NSImageView). These traits were split in objc2
    // v0.6 to enforce thread-safety at the type level.
    use objc2::MainThreadMarker;
    use objc2::{AnyThread, MainThreadOnly};
    use objc2_app_kit::{NSColor, NSImage, NSImageView, NSView, NSWindow};
    use objc2_foundation::{NSData, NSSize};
    use tao::platform::macos::WindowExtMacOS;

    unsafe {
        // Get the raw NSWindow and NSView pointers from tao's platform extension.
        // These are *mut c_void that we cast to objc2 reference types for message
        // sending. The pointers remain valid for the lifetime of the tao Window.
        let ns_window_ptr = window.ns_window();
        let ns_view_ptr = window.ns_view();

        let ns_window: &NSWindow = &*(ns_window_ptr as *const NSWindow);
        let ns_view: &NSView = &*(ns_view_ptr as *const NSView);

        // Configure the NSWindow for full transparency. setOpaque(false) tells
        // the window server that this window has non-opaque regions. Setting
        // the background color to clearColor makes all undrawn areas transparent.
        // Combined, the PNG's alpha channel determines window transparency.
        ns_window.setOpaque(false);
        ns_window.setHasShadow(false);
        let clear_color = NSColor::clearColor();
        ns_window.setBackgroundColor(Some(&clear_color));

        // Create an NSImage directly from the raw PNG bytes. AppKit's PNG
        // decoder handles the RGBA data, alpha channel, and color space
        // conversion internally. This avoids manual pixel buffer manipulation.
        // NSImage implements AnyThread, so alloc() takes no marker argument.
        let png_data = NSData::with_bytes(SPLASH_PNG);
        let image: Retained<NSImage> = NSImage::initWithData(NSImage::alloc(), &png_data)
            .ok_or_else(|| "NSImage::initWithData returned nil".to_string())?;

        // Set the image size to logical points (half the pixel dimensions).
        // On Retina displays, AppKit uses the full 2x pixel data for sharp
        // rendering. On non-Retina, it downsamples to the logical size.
        image.setSize(NSSize {
            width: 480.0,
            height: 180.0,
        });

        // NSImageView implements MainThreadOnly, requiring a MainThreadMarker
        // for allocation. This code runs inside tao's event loop callback,
        // which executes on the main thread (NSApplication run loop). The
        // unchecked constructor is safe here because tao guarantees main-thread
        // execution for all event loop callbacks on macOS.
        let mtm = MainThreadMarker::new_unchecked();
        let frame = ns_view.bounds();
        let image_view: Retained<NSImageView> =
            NSImageView::initWithFrame(NSImageView::alloc(mtm), frame);
        image_view.setImage(Some(&image));
        ns_view.addSubview(&image_view);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Linux: GTK + Cairo draw handler with RGBA visual
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
fn render_platform(window: &Window, rgba: &[u8], width: u32, height: u32) -> Result<(), String> {
    use cairo::Format;
    use gtk::prelude::*;
    use tao::platform::unix::WindowExtUnix;

    // Calculate the Cairo stride for the ARgb32 format at the given width.
    // Cairo may pad rows to a 4-byte boundary (though for 32bpp at any width
    // the stride equals width * 4, the API is used for correctness).
    let stride = Format::ARgb32
        .stride_for_width(width)
        .map_err(|_| "cairo stride calculation failed".to_string())?;

    // Convert RGBA (PNG byte order) to premultiplied BGRA (Cairo ARgb32 format
    // on little-endian systems). Cairo requires premultiplied alpha for correct
    // compositing: each color channel is multiplied by the alpha value.
    // The byte order in memory for ARgb32 on little-endian is B, G, R, A.
    let mut argb_data = vec![0u8; (stride * height as i32) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let src = (y * width as usize + x) * 4;
            let dst = y * stride as usize + x * 4;
            let (r, g, b, a) = (
                rgba[src] as u32,
                rgba[src + 1] as u32,
                rgba[src + 2] as u32,
                rgba[src + 3] as u32,
            );
            argb_data[dst] = ((b * a / 255) & 0xFF) as u8;
            argb_data[dst + 1] = ((g * a / 255) & 0xFF) as u8;
            argb_data[dst + 2] = ((r * a / 255) & 0xFF) as u8;
            argb_data[dst + 3] = a as u8;
        }
    }

    // Create a Cairo ImageSurface from the premultiplied ARGB data. The surface
    // owns the pixel data and is cloned into the draw signal closure.
    let surface = cairo::ImageSurface::create_for_data(
        argb_data,
        Format::ARgb32,
        width as i32,
        height as i32,
        stride,
    )
    .map_err(|e| format!("cairo ImageSurface creation failed: {e}"))?;

    // Get the GTK ApplicationWindow from tao's platform extension and connect
    // a draw signal handler. The handler clears to transparent, then paints
    // the PNG surface scaled to fit the widget allocation.
    let gtk_window = window.gtk_window();
    let img_width = width as f64;
    let img_height = height as f64;

    gtk_window.connect_draw(move |widget, cr| {
        let alloc = widget.allocation();

        // Clear the entire surface to transparent. The Source operator
        // replaces all pixels (including alpha) rather than compositing.
        cr.set_operator(cairo::Operator::Source);
        cr.set_source_rgba(0.0, 0.0, 0.0, 0.0);
        let _ = cr.paint();

        // Scale the 2x PNG to fit the logical window dimensions.
        // On a 100% DPI display, the 480x180 widget receives a 960x360 PNG,
        // so the scale factors are 0.5. On HiDPI the factors approach 1.0.
        let sx = alloc.width() as f64 / img_width;
        let sy = alloc.height() as f64 / img_height;
        cr.scale(sx, sy);

        // Paint the PNG surface with the Over operator so the alpha channel
        // composites correctly against the transparent background.
        cr.set_operator(cairo::Operator::Over);
        let _ = cr.set_source_surface(&surface, 0.0, 0.0);
        let _ = cr.paint();

        gtk::glib::Propagation::Stop
    });

    // Trigger an initial redraw so the PNG is painted immediately after
    // the window is shown. Without this, the draw handler would fire only
    // on the next expose event.
    gtk_window.queue_draw();

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    /// Verifies that the embedded splash PNG decodes to the expected dimensions
    /// and pixel buffer size. This catches corrupted or incorrectly generated
    /// PNG files at test time rather than at runtime.
    #[test]
    fn splash_png_decodes_to_expected_dimensions() {
        let (pixels, w, h) = super::decode_png();
        assert_eq!(w, 960, "splash PNG width must be 960 pixels (2x of 480)");
        assert_eq!(h, 360, "splash PNG height must be 360 pixels (2x of 180)");
        assert_eq!(
            pixels.len(),
            (w * h * 4) as usize,
            "pixel buffer must contain width * height * 4 bytes (RGBA)"
        );
    }
}
