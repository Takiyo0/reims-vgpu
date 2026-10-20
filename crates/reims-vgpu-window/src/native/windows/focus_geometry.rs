//! Checked screen-space bounds for a future cursor-confinement adapter.
//!
//! Inputs must already be client edges converted into physical screen coordinates. Negative
//! virtual-desktop origins are valid. The caller must refresh the bounds on DPI, move and
//! resize changes, and must handle focus loss and cleanup separately. No Win32 calls occur
//! here: producing these bounds neither confines the cursor nor implements cleanup.

/// Valid nonempty bounds with representable signed dimensions, in screen pixels.
/// Right and bottom are exclusive, matching a Win32 RECT.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ClipBounds {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

impl ClipBounds {
    /// Validate screen-space client edges before a caller constructs a ClipCursor RECT.
    /// Reject dimensions whose subtraction overflows rather than wrapping or clamping.
    pub(crate) fn from_client_edges(left: i32, top: i32, right: i32, bottom: i32) -> Option<Self> {
        let width = right.checked_sub(left)?;
        let height = bottom.checked_sub(top)?;
        if width <= 0 || height <= 0 {
            return None;
        }
        Some(Self {
            left,
            top,
            right,
            bottom,
        })
    }

    /// `(left, top, right, bottom)` for the future OS adapter; no coordinate scaling occurs.
    pub(crate) fn edges(self) -> (i32, i32, i32, i32) {
        (self.left, self.top, self.right, self.bottom)
    }
}

#[cfg(test)]
mod tests {
    use super::ClipBounds;

    #[test]
    fn normal_client_bounds_preserve_edges() {
        let bounds = ClipBounds::from_client_edges(100, 200, 1380, 920).unwrap();
        assert_eq!(bounds.edges(), (100, 200, 1380, 920));
    }

    #[test]
    fn negative_virtual_desktop_origin_is_valid() {
        for edges in [(-1920, -1080, 0, 0), (-800, -200, 800, 600)] {
            let (left, top, right, bottom) = edges;
            assert_eq!(
                ClipBounds::from_client_edges(left, top, right, bottom)
                    .unwrap()
                    .edges(),
                edges
            );
        }
    }

    #[test]
    fn empty_or_inverted_bounds_are_rejected() {
        for (left, top, right, bottom) in [
            (1, 2, 1, 3),
            (1, 2, 3, 2),
            (2, 2, 1, 3),
            (1, 3, 3, 2),
            (0, 0, 0, 0),
        ] {
            assert!(ClipBounds::from_client_edges(left, top, right, bottom).is_none());
        }
    }

    #[test]
    fn integer_edges_are_checked_without_clamping() {
        let lo = i32::MIN;
        let hi = i32::MAX;
        for edges in [
            (lo, lo, lo + 1, lo + 1),
            (hi - 1, hi - 1, hi, hi),
            (0, 0, hi, hi),
        ] {
            let (left, top, right, bottom) = edges;
            assert_eq!(
                ClipBounds::from_client_edges(left, top, right, bottom)
                    .unwrap()
                    .edges(),
                edges
            );
        }
        for (left, top, right, bottom) in [
            (lo, 0, hi, 1),
            (0, lo, 1, hi),
            (hi, 0, lo, 1),
            (0, hi, 1, lo),
            (-1, 0, hi, 1),
        ] {
            assert!(ClipBounds::from_client_edges(left, top, right, bottom).is_none());
        }
    }
}
