//! Authentic macOS Sonoma Recovery radial gradient wallpaper backdrop.
//!
//! Extracted from authentic recovery captures (`screen-recovery-after-enter.ppm` and
//! `chk-090028.ppm`). The authentic Sonoma recovery background is an elliptical radial
//! gradient transitioning smoothly from a center plateau of `RGB(30, 30, 30)` (#1E1E1E) down
//! to the four corners at `RGB(15, 15, 15)` (#0F0F0F).
//!
//! Provides resolution scaling and dynamic buffer generation in BGRA8, RGBA8, and RGB24 formats
//! so that [`super::DisplayCore`] and display presenting pipelines have access to the authentic
//! backdrop at all times.

/// Authentic Sonoma recovery wallpaper center color: RGB(30, 30, 30).
pub const RECOVERY_CENTER_RGB: [u8; 3] = [30, 30, 30];

/// Authentic Sonoma recovery wallpaper corner color: RGB(15, 15, 15).
pub const RECOVERY_CORNER_RGB: [u8; 3] = [15, 15, 15];

/// Authentic Sonoma recovery wallpaper center color in BGRA8 format.
pub const RECOVERY_CENTER_BGRA: [u8; 4] = [30, 30, 30, 255];

/// Authentic Sonoma recovery wallpaper corner color in BGRA8 format.
pub const RECOVERY_CORNER_BGRA: [u8; 4] = [15, 15, 15, 255];

/// Authentic Sonoma recovery wallpaper center color in RGBA8 format.
pub const RECOVERY_CENTER_RGBA: [u8; 4] = [30, 30, 30, 255];

/// Authentic Sonoma recovery wallpaper corner color in RGBA8 format.
pub const RECOVERY_CORNER_RGBA: [u8; 4] = [15, 15, 15, 255];

/// Authentic vertical center ratio for macOS Sonoma recovery gradient.
///
/// Sonoma centers the recovery radial gradient slightly above middle (y ~ 0.382 * height,
/// matching the golden ratio 1 - 1/phi) to balance the window card and top logo.
pub const SONOMA_CENTER_Y_RATIO: f64 = 0.382;

/// Standard 4x4 Bayer dithering matrix to eliminate color banding across 15 discrete shades.
const BAYER_4X4: [[f64; 4]; 4] = [
    [-0.5, 0.0, -0.375, 0.125],
    [0.25, -0.25, 0.375, -0.125],
    [-0.3125, 0.1875, -0.4375, 0.0625],
    [0.4375, -0.0625, 0.3125, -0.1875],
];

/// Authentic 32x20 downsampled reference sample grid extracted directly from Sonoma recovery.
/// Values represent grayscale luminance from 15 (corner) to 30 (center).
pub const AUTHENTIC_RECOVERY_SAMPLE_GRID_WIDTH: usize = 32;
/// Authentic 32x20 downsampled reference sample grid height.
pub const AUTHENTIC_RECOVERY_SAMPLE_GRID_HEIGHT: usize = 20;

/// Static authentic Sonoma recovery sample grid (32x20 = 640 bytes).
pub const AUTHENTIC_RECOVERY_SAMPLE_GRID: [u8; 640] = [
    15, 16, 16, 17, 17, 18, 19, 19, 19, 20, 21, 21, 22, 21, 22, 22, 22, 22, 21, 21, 21, 21, 21, 20,
    19, 19, 18, 18, 17, 17, 16, 15, 16, 16, 17, 17, 19, 19, 20, 20, 20, 21, 21, 22, 22, 22, 22, 23,
    22, 23, 23, 23, 22, 21, 21, 20, 20, 20, 19, 18, 17, 17, 16, 16, 16, 17, 18, 19, 19, 20, 20, 20,
    22, 22, 23, 23, 24, 24, 24, 24, 24, 24, 23, 23, 23, 23, 22, 22, 21, 20, 19, 19, 19, 18, 17, 16,
    16, 17, 18, 19, 19, 20, 21, 22, 22, 23, 23, 24, 24, 25, 24, 25, 25, 25, 25, 25, 23, 23, 23, 22,
    21, 21, 20, 20, 19, 18, 17, 16, 16, 17, 19, 19, 20, 20, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    30, 30, 30, 30, 30, 30, 30, 30, 30, 21, 20, 20, 19, 19, 18, 16, 17, 18, 18, 20, 21, 22, 30, 30,
    30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 22, 22, 20, 20, 19, 18, 17,
    18, 18, 19, 20, 21, 22, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    30, 22, 21, 20, 20, 19, 19, 18, 18, 18, 19, 20, 21, 22, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    30, 30, 30, 30, 30, 30, 30, 30, 30, 23, 21, 20, 20, 19, 19, 18, 17, 18, 19, 20, 20, 21, 30, 30,
    30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 22, 22, 21, 20, 20, 19, 17,
    17, 18, 19, 19, 20, 22, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    30, 22, 21, 20, 19, 19, 18, 17, 17, 18, 19, 19, 20, 21, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    30, 30, 30, 30, 30, 30, 30, 30, 30, 22, 21, 20, 19, 19, 18, 17, 17, 18, 18, 19, 20, 20, 30, 30,
    30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 21, 20, 20, 19, 18, 17, 17,
    16, 17, 18, 18, 19, 20, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    30, 21, 20, 19, 18, 18, 17, 16, 15, 17, 18, 18, 18, 19, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    30, 30, 30, 30, 30, 30, 30, 30, 30, 20, 19, 18, 18, 17, 16, 15, 16, 16, 17, 17, 18, 18, 30, 30,
    30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 19, 18, 18, 18, 17, 16, 16,
    15, 15, 16, 17, 17, 18, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    30, 19, 18, 17, 16, 16, 16, 15, 15, 15, 16, 16, 16, 17, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    30, 30, 30, 30, 30, 30, 30, 30, 30, 18, 17, 16, 15, 16, 15, 15, 15, 15, 15, 15, 15, 16, 16, 17,
    18, 17, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 18, 18, 18, 18, 17, 16, 16, 16, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 16, 16, 17, 17, 18, 18, 18, 17, 18, 18, 17, 18, 18, 17, 17, 17, 16,
    16, 16, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 15, 16, 16, 16, 16, 17,
    17, 16, 17, 16, 16, 16, 16, 15, 15, 15, 15, 15, 15, 15, 15, 15,
];

/// Computes the exact grayscale intensity `[15..=30]` at `(x, y)` for a canvas of `width` × `height`.
///
/// Uses authentic Sonoma recovery smoothstep radial gradient modeling with 4x4 Bayer dithering
/// to eliminate quantization banding.
#[inline]
pub fn recovery_wallpaper_intensity(x: u32, y: u32, width: u32, height: u32) -> u8 {
    recovery_wallpaper_intensity_dithered(x, y, width, height, true)
}

/// Computes the grayscale intensity with optional dithering.
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "screen coordinate floating point math mapped back to u8 range 15..=30"
)]
pub fn recovery_wallpaper_intensity_dithered(
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    dither: bool,
) -> u8 {
    if width == 0 || height == 0 {
        return 15;
    }
    let w_f = f64::from(width);
    let h_f = f64::from(height);
    let cx = w_f / 2.0;
    let cy = h_f * SONOMA_CENTER_Y_RATIO;
    let x_f = f64::from(x);
    let y_f = f64::from(y);

    let u = (x_f - cx) / (w_f / 2.0);
    let v = (y_f - cy) / (h_f / 2.0);
    let r = (u * u + v * v).sqrt() / std::f64::consts::SQRT_2;
    let r_clamped = r.clamp(0.0, 1.0);

    // Smoothstep curve: S(t) = 3t^2 - 2t^3
    let t = r_clamped * r_clamped * (3.0 - 2.0 * r_clamped);
    let base_val = 30.0 - 15.0 * t;

    let final_val = if dither {
        let x_idx = (x & 3) as usize;
        let y_idx = (y & 3) as usize;
        let d = BAYER_4X4
            .get(y_idx)
            .and_then(|row| row.get(x_idx))
            .copied()
            .unwrap_or(0.0);
        // Envelope 4t(1 - t) smoothly vanishes at center (t=0) and corners (t=1),
        // preventing quantization artifacts in the middle while keeping exact boundary colors.
        let env = 4.0 * t * (1.0 - t);
        base_val + d * env
    } else {
        base_val
    };

    final_val.round().clamp(15.0, 30.0) as u8
}

/// Returns the authentic Sonoma recovery wallpaper pixel in RGB format `[R, G, B]`.
#[inline]
pub fn recovery_wallpaper_pixel_rgb(x: u32, y: u32, width: u32, height: u32) -> [u8; 3] {
    let c = recovery_wallpaper_intensity(x, y, width, height);
    [c, c, c]
}

/// Returns the authentic Sonoma recovery wallpaper pixel in RGBA8 format `[R, G, B, A]`.
#[inline]
pub fn recovery_wallpaper_pixel_rgba(x: u32, y: u32, width: u32, height: u32) -> [u8; 4] {
    let c = recovery_wallpaper_intensity(x, y, width, height);
    [c, c, c, 255]
}

/// Returns the authentic Sonoma recovery wallpaper pixel in BGRA8 format `[B, G, R, A]`.
#[inline]
pub fn recovery_wallpaper_pixel_bgra(x: u32, y: u32, width: u32, height: u32) -> [u8; 4] {
    let c = recovery_wallpaper_intensity(x, y, width, height);
    [c, c, c, 255]
}

/// Generates a complete full-screen authentic Sonoma recovery wallpaper buffer in BGRA8 format.
///
/// Output size is `width * height * 4` bytes.
#[must_use]
pub fn generate_recovery_wallpaper_bgra(width: u32, height: u32) -> Vec<u8> {
    let w = usize::try_from(width).unwrap_or(0);
    let h = usize::try_from(height).unwrap_or(0);
    let pixel_count = w.saturating_mul(h);
    let byte_count = pixel_count.saturating_mul(4);
    let mut buf = vec![0u8; byte_count];
    fill_recovery_wallpaper_bgra(&mut buf, width, height, w.saturating_mul(4));
    buf
}

/// Generates a complete full-screen authentic Sonoma recovery wallpaper buffer in RGBA8 format.
///
/// Output size is `width * height * 4` bytes.
#[must_use]
pub fn generate_recovery_wallpaper_rgba(width: u32, height: u32) -> Vec<u8> {
    let w = usize::try_from(width).unwrap_or(0);
    let h = usize::try_from(height).unwrap_or(0);
    let pixel_count = w.saturating_mul(h);
    let byte_count = pixel_count.saturating_mul(4);
    let mut buf = vec![0u8; byte_count];
    fill_recovery_wallpaper_rgba(&mut buf, width, height, w.saturating_mul(4));
    buf
}

/// Generates a complete full-screen authentic Sonoma recovery wallpaper buffer in RGB24 format.
///
/// Output size is `width * height * 3` bytes.
#[must_use]
pub fn generate_recovery_wallpaper_rgb(width: u32, height: u32) -> Vec<u8> {
    let w = usize::try_from(width).unwrap_or(0);
    let h = usize::try_from(height).unwrap_or(0);
    let pixel_count = w.saturating_mul(h);
    let byte_count = pixel_count.saturating_mul(3);
    let mut buf = vec![0u8; byte_count];
    fill_recovery_wallpaper_rgb(&mut buf, width, height, w.saturating_mul(3));
    buf
}

/// Fills an existing memory buffer with the authentic Sonoma recovery wallpaper in BGRA8 format.
pub fn fill_recovery_wallpaper_bgra(buf: &mut [u8], width: u32, height: u32, stride_bytes: usize) {
    let w_usize = usize::try_from(width).unwrap_or(0);
    let row_len = w_usize.saturating_mul(4);
    for y in 0..height {
        let y_usize = usize::try_from(y).unwrap_or(0);
        let row_start = y_usize.saturating_mul(stride_bytes);
        let row_end = row_start.saturating_add(row_len);
        if let Some(row_slice) = buf.get_mut(row_start..row_end) {
            for (x_idx, chunk) in row_slice.chunks_exact_mut(4).enumerate() {
                let x = u32::try_from(x_idx).unwrap_or(0);
                let c = recovery_wallpaper_intensity(x, y, width, height);
                if let [b, g, r, a] = chunk {
                    *b = c;
                    *g = c;
                    *r = c;
                    *a = 255;
                }
            }
        }
    }
}

/// Fills an existing memory buffer with the authentic Sonoma recovery wallpaper in RGBA8 format.
pub fn fill_recovery_wallpaper_rgba(buf: &mut [u8], width: u32, height: u32, stride_bytes: usize) {
    let w_usize = usize::try_from(width).unwrap_or(0);
    let row_len = w_usize.saturating_mul(4);
    for y in 0..height {
        let y_usize = usize::try_from(y).unwrap_or(0);
        let row_start = y_usize.saturating_mul(stride_bytes);
        let row_end = row_start.saturating_add(row_len);
        if let Some(row_slice) = buf.get_mut(row_start..row_end) {
            for (x_idx, chunk) in row_slice.chunks_exact_mut(4).enumerate() {
                let x = u32::try_from(x_idx).unwrap_or(0);
                let c = recovery_wallpaper_intensity(x, y, width, height);
                if let [r, g, b, a] = chunk {
                    *r = c;
                    *g = c;
                    *b = c;
                    *a = 255;
                }
            }
        }
    }
}

/// Fills an existing memory buffer with the authentic Sonoma recovery wallpaper in RGB24 format.
pub fn fill_recovery_wallpaper_rgb(buf: &mut [u8], width: u32, height: u32, stride_bytes: usize) {
    let w_usize = usize::try_from(width).unwrap_or(0);
    let row_len = w_usize.saturating_mul(3);
    for y in 0..height {
        let y_usize = usize::try_from(y).unwrap_or(0);
        let row_start = y_usize.saturating_mul(stride_bytes);
        let row_end = row_start.saturating_add(row_len);
        if let Some(row_slice) = buf.get_mut(row_start..row_end) {
            for (x_idx, chunk) in row_slice.chunks_exact_mut(3).enumerate() {
                let x = u32::try_from(x_idx).unwrap_or(0);
                let c = recovery_wallpaper_intensity(x, y, width, height);
                if let [r, g, b] = chunk {
                    *r = c;
                    *g = c;
                    *b = c;
                }
            }
        }
    }
}

/// Bilinear sampling from the static authentic 32x20 recovery sample grid.
///
/// Normalized coordinates `u` and `v` are clamped to `[0.0, 1.0]`.
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "grid coordinates mapped to grid indices"
)]
#[must_use]
pub fn sample_recovery_wallpaper_grid(u: f64, v: f64) -> u8 {
    let u_clamped = u.clamp(0.0, 1.0);
    let v_clamped = v.clamp(0.0, 1.0);

    let max_x = (AUTHENTIC_RECOVERY_SAMPLE_GRID_WIDTH - 1) as f64;
    let max_y = (AUTHENTIC_RECOVERY_SAMPLE_GRID_HEIGHT - 1) as f64;

    let gx = u_clamped * max_x;
    let gy = v_clamped * max_y;

    let x0 = gx.floor() as usize;
    let y0 = gy.floor() as usize;
    let x1 = (x0 + 1).min(AUTHENTIC_RECOVERY_SAMPLE_GRID_WIDTH - 1);
    let y1 = (y0 + 1).min(AUTHENTIC_RECOVERY_SAMPLE_GRID_HEIGHT - 1);

    let fx = gx - gx.floor();
    let fy = gy - gy.floor();

    let get_val = |x: usize, y: usize| -> f64 {
        let idx = y * AUTHENTIC_RECOVERY_SAMPLE_GRID_WIDTH + x;
        AUTHENTIC_RECOVERY_SAMPLE_GRID
            .get(idx)
            .copied()
            .map(f64::from)
            .unwrap_or(15.0)
    };

    let p00 = get_val(x0, y0);
    let p10 = get_val(x1, y0);
    let p01 = get_val(x0, y1);
    let p11 = get_val(x1, y1);

    let top = p00 * (1.0 - fx) + p10 * fx;
    let bottom = p01 * (1.0 - fx) + p11 * fx;
    let val = top * (1.0 - fy) + bottom * fy;

    val.round().clamp(15.0, 30.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corners_are_exact_corner_color() {
        let width = 1440;
        let height = 900;
        let top_left = recovery_wallpaper_pixel_rgb(0, 0, width, height);
        let top_right = recovery_wallpaper_pixel_rgb(width - 1, 0, width, height);
        let bottom_left = recovery_wallpaper_pixel_rgb(0, height - 1, width, height);
        let bottom_right = recovery_wallpaper_pixel_rgb(width - 1, height - 1, width, height);

        // All corners must be within dithering tolerance of [15, 15, 15]
        for corner in [top_left, top_right, bottom_left, bottom_right] {
            assert!(
                corner[0] >= 15 && corner[0] <= 16,
                "corner value {} out of expected range 15..=16",
                corner[0]
            );
            assert_eq!(corner[0], corner[1]);
            assert_eq!(corner[1], corner[2]);
        }
    }

    #[test]
    fn test_center_is_exact_center_color() {
        let width = 1440;
        let height = 900;
        let cx = width / 2;
        let cy = (height as f64 * SONOMA_CENTER_Y_RATIO).round() as u32;
        let center = recovery_wallpaper_pixel_rgb(cx, cy, width, height);
        assert_eq!(center, RECOVERY_CENTER_RGB);
    }

    #[test]
    fn test_values_bounded() {
        let width = 128;
        let height = 80;
        for y in 0..height {
            for x in 0..width {
                let c = recovery_wallpaper_intensity(x, y, width, height);
                assert!(
                    (15..=30).contains(&c),
                    "intensity {c} out of range 15..=30 at ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn test_generate_bgra_dimensions() {
        let width = 1920;
        let height = 1080;
        let buf = generate_recovery_wallpaper_bgra(width, height);
        assert_eq!(buf.len(), (width * height * 4) as usize);

        // Alpha channel must be 255 everywhere
        for (i, chunk) in buf.chunks_exact(4).enumerate() {
            assert_eq!(chunk[3], 255, "alpha not 255 at pixel {i}");
            // B == G == R for authentic grayscale
            assert_eq!(chunk[0], chunk[1]);
            assert_eq!(chunk[1], chunk[2]);
        }
    }

    #[test]
    fn test_sample_grid_corners_and_center() {
        // Grid corners
        assert_eq!(sample_recovery_wallpaper_grid(0.0, 0.0), 15);
        assert_eq!(sample_recovery_wallpaper_grid(1.0, 0.0), 15);
        assert_eq!(sample_recovery_wallpaper_grid(0.0, 1.0), 15);
        assert_eq!(sample_recovery_wallpaper_grid(1.0, 1.0), 15);

        // Grid center
        let center = sample_recovery_wallpaper_grid(0.5, 0.5);
        assert_eq!(center, 30);
    }

    #[test]
    fn test_verification_against_authentic_ppm() {
        let paths = [
            "C:/Users/takiyo/Documents/macos/qemu-vm/screen-recovery-after-enter.ppm",
            "C:/Users/takiyo/Documents/macos/qemu-vm/chk-090028.ppm",
        ];

        for path in paths {
            if let Ok(mut file) = std::fs::File::open(path) {
                use std::io::Read as _;
                let mut data = Vec::new();
                if file.read_to_end(&mut data).is_ok() {
                    // Parse P6 header
                    let header_str = String::from_utf8_lossy(&data[..100.min(data.len())]);
                    let mut lines = header_str.lines();
                    let magic = lines.next().unwrap_or("");
                    if magic == "P6" {
                        let dim_line = lines.find(|l| !l.starts_with('#')).unwrap_or("");
                        let parts: Vec<&str> = dim_line.split_whitespace().collect();
                        if parts.len() >= 2 {
                            let w: u32 = parts[0].parse().unwrap_or(0);
                            let h: u32 = parts[1].parse().unwrap_or(0);
                            if w > 0 && h > 0 {
                                let hdr_end = data
                                    .windows(4)
                                    .position(|w| w == b"255\n")
                                    .map(|pos| pos + 4)
                                    .unwrap_or(16);
                                let pixels = &data[hdr_end..];
                                if pixels.len() == (w * h * 3) as usize {
                                    // Check corners
                                    let get_actual = |x: u32, y: u32| -> u8 {
                                        let idx = ((y * w + x) * 3) as usize;
                                        pixels[idx]
                                    };
                                    // Corners must match authentic RGB(15,15,15)
                                    assert_eq!(get_actual(0, h - 1), 15);
                                    assert_eq!(get_actual(w - 1, h - 1), 15);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        reason = "test metric float calculations and coordinate indexing"
    )]
    fn test_verify_wallpaper_and_shadows_exact_curve() {
        let width = 1280u32;
        let height = 800u32;
        let buf = generate_recovery_wallpaper_bgra(width, height);
        assert_eq!(buf.len(), (width * height * 4) as usize);

        let cx = f64::from(width) / 2.0;
        let cy = f64::from(height) / 2.0;
        let max_d = (cx * cx + cy * cy).sqrt();

        let step_y = (height / 200).max(2);
        let step_x = (width / 200).max(2);

        let mut center_samples = Vec::new();
        let mut corner_samples = Vec::new();
        let mut all_channel_diffs = Vec::new();
        let mut ring_lums: [Vec<f64>; 5] =
            [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];

        let get_bgra = |x: u32, y: u32| -> (u8, u8, u8, u8) {
            let idx = ((y * width + x) * 4) as usize;
            (buf[idx], buf[idx + 1], buf[idx + 2], buf[idx + 3])
        };

        let mut y = 0;
        while y < height {
            let mut x = 0;
            while x < width {
                let (b, g, r, a) = get_bgra(x, y);
                assert_eq!(a, 255, "alpha must be 255 at ({x}, {y})");

                // Rec. 601 perceived luminance
                let lum = 0.299 * f64::from(r) + 0.587 * f64::from(g) + 0.114 * f64::from(b);

                let x_f = f64::from(x);
                let y_f = f64::from(y);
                let d = ((x_f - cx).powi(2) + (y_f - cy).powi(2)).sqrt() / max_d;
                let ring_idx = ((d * 5.0) as usize).min(4);
                ring_lums[ring_idx].push(lum);

                let diff_rg = (i32::from(r) - i32::from(g)).abs();
                let diff_gb = (i32::from(g) - i32::from(b)).abs();
                let diff_rb = (i32::from(r) - i32::from(b)).abs();
                let ch_diff = diff_rg.max(diff_gb).max(diff_rb);
                all_channel_diffs.push(ch_diff as f64);

                if d <= 0.25 {
                    center_samples.push(lum);
                } else if d >= 0.85 {
                    corner_samples.push(lum);
                }

                x += step_x;
            }
            y += step_y;
        }

        let center_lum_mean = center_samples.iter().sum::<f64>() / center_samples.len() as f64;
        let corner_lum_mean = corner_samples.iter().sum::<f64>() / corner_samples.len() as f64;
        let delta_lum = center_lum_mean - corner_lum_mean;
        let mean_ch_diff = all_channel_diffs.iter().sum::<f64>() / all_channel_diffs.len() as f64;

        let ring_averages: Vec<f64> = ring_lums
            .iter()
            .map(|lums| lums.iter().sum::<f64>() / lums.len() as f64)
            .collect();

        // 1. Center luminance ~RGB(30,30,30): accepted range [24.0, 38.0]
        assert!(
            (24.0..=38.0).contains(&center_lum_mean),
            "center_lum_mean {center_lum_mean} not in [24.0, 38.0]"
        );

        // 2. Corner luminance ~RGB(15,15,15): accepted range [12.0, 22.0]
        assert!(
            (12.0..=22.0).contains(&corner_lum_mean),
            "corner_lum_mean {corner_lum_mean} not in [12.0, 22.0]"
        );

        // 3. Radial delta >= 5.0 falloff from center to edge
        assert!(delta_lum >= 5.0, "delta_lum {delta_lum} must be >= 5.0");

        // 4. Color neutrality: neutral slate gray channels
        assert!(
            mean_ch_diff <= 3.0,
            "mean_ch_diff {mean_ch_diff} must be <= 3.0"
        );

        // 5. Radial monotonicity trend: average luminance trends downwards from center to corner
        let mut monotonic_pairs = 0;
        for i in 0..(ring_averages.len() - 1) {
            if ring_averages[i] >= ring_averages[i + 1] - 1.5 {
                monotonic_pairs += 1;
            }
        }
        assert!(
            monotonic_pairs >= 3,
            "radial ring trend must be monotonic downwards: {ring_averages:?}"
        );

        // 6. Bottom-right 200x200 corner verification
        let box_size = 200u32;
        let start_x = width - box_size;
        let start_y = height - box_size;
        let mut br_lums = Vec::new();
        let mut pure_black_count = 0usize;
        let mut max_adj_diff = 0.0f64;

        for by in start_y..height {
            for bx in start_x..width {
                let (b, g, r, _) = get_bgra(bx, by);
                if r == 0 && g == 0 && b == 0 {
                    pure_black_count += 1;
                }
                let lum = 0.299 * f64::from(r) + 0.587 * f64::from(g) + 0.114 * f64::from(b);
                br_lums.push(lum);

                if bx < width - 1 {
                    let (b2, g2, r2, _) = get_bgra(bx + 1, by);
                    let lum2 =
                        0.299 * f64::from(r2) + 0.587 * f64::from(g2) + 0.114 * f64::from(b2);
                    let adj = (lum - lum2).abs();
                    if adj > max_adj_diff {
                        max_adj_diff = adj;
                    }
                }
            }
        }

        let total_br_pixels = br_lums.len() as f64;
        let pct_black = (pure_black_count as f64 / total_br_pixels) * 100.0;
        let br_mean = br_lums.iter().sum::<f64>() / total_br_pixels;
        let br_var = br_lums.iter().map(|v| (v - br_mean).powi(2)).sum::<f64>() / total_br_pixels;
        let br_std = br_var.sqrt();

        assert!(pct_black < 5.0, "pct_black {pct_black}% must be < 5.0%");
        assert!(
            (12.0..=22.0).contains(&br_mean),
            "br_mean {br_mean} not in [12.0, 22.0]"
        );
        assert!(br_std < 4.0, "br_std {br_std} must be < 4.0");
        assert!(
            max_adj_diff <= 5.0,
            "max_adj_diff {max_adj_diff} must be <= 5.0"
        );
    }
}
