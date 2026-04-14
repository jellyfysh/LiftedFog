//! Geometry-related utilities.

use nalgebra as na;
use rand_distr::num_traits::ToPrimitive;

use crate::types::{I2, Pos};

/// 2D rectangular unit cell.
#[derive(Clone, Copy, Debug, PartialEq)]
#[readonly::make]
pub struct GridUnit {
    /// Cell size (X).
    pub dx: f64,
    /// Cell size (Y).
    pub dy: f64,
}

impl GridUnit {
    /// Creates a new [`GridUnit`] instance.
    ///
    /// # Errors
    /// If `dx` or `dy` is not positive.
    #[inline]
    pub fn new(dx: f64, dy: f64) -> anyhow::Result<Self> {
        anyhow::ensure!(dx > 0.0, "dx must be positive");
        anyhow::ensure!(dy > 0.0, "dy must be positive");
        Ok(Self { dx, dy })
    }
}

/// Number of cells.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[readonly::make]
pub struct GridSize {
    /// Number of cells (X).
    pub nx: usize,
    /// Number of cells (Y).
    pub ny: usize,
}

impl GridSize {
    /// Creates a new [`GridSize`] instance.
    ///
    /// # Errors
    /// If `nx` or `ny` is zero.
    #[inline]
    pub fn new(nx: usize, ny: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(nx > 0, "nx must be positive");
        anyhow::ensure!(ny > 0, "ny must be positive");
        Ok(Self { nx, ny })
    }

    /// Returns the neighboring 9 cells (including self) assuming PBC.
    #[inline]
    #[must_use]
    pub fn neighbors(&self, (i, j): I2) -> [I2; 9] {
        let mut ret = [I2::default(); _];
        for (ii, (ox, oy)) in
            itertools::iproduct!([self.nx - 1, 0, 1], [self.ny - 1, 0, 1]).enumerate()
        {
            let i = (i + ox) % self.nx;
            let j = (j + oy) % self.ny;
            ret[ii] = (i, j);
        }
        ret
    }
}

/// Periodic 2D grid.
#[derive(Clone, Copy, Debug, PartialEq)]
#[readonly::make]
pub struct Grid {
    /// Unit cell.
    pub unit: GridUnit,
    /// Grid size.
    pub size: GridSize,
}

impl Grid {
    /// Creates a new [`Grid`] instance.
    #[inline]
    #[must_use]
    pub const fn new(unit: GridUnit, size: GridSize) -> Self {
        Self { unit, size }
    }

    /// Returns the total system size.
    #[expect(clippy::cast_precision_loss)]
    #[inline]
    #[must_use]
    pub const fn simbox(&self) -> (f64, f64) {
        (
            self.unit.dx * self.size.nx as f64,
            self.unit.dy * self.size.ny as f64,
        )
    }

    /// Normalizes the position assuming PBC.
    #[inline]
    #[must_use]
    pub fn normalize(&self, pos: &Pos) -> Pos {
        let (lx, ly) = self.simbox();
        na::vector![pos.x.rem_euclid(lx), pos.y.rem_euclid(ly)]
    }

    /// Checks if the position is inside the box.
    #[inline]
    #[must_use]
    pub fn is_normalized(&self, pos: &Pos) -> bool {
        let (lx, ly) = self.simbox();
        (0.0..lx).contains(&pos.x) && (0.0..ly).contains(&pos.y)
    }

    /// Returns the grid index assuming PBC.
    #[expect(clippy::missing_panics_doc)]
    #[inline]
    #[must_use]
    pub fn gindex(&self, pos: &Pos) -> I2 {
        let pos = self.normalize(pos);
        let nx = (pos.x / self.unit.dx)
            .floor()
            .to_usize()
            .expect("not too large");
        debug_assert!(nx < self.size.nx);
        let ny = (pos.y / self.unit.dy)
            .floor()
            .to_usize()
            .expect("not too large");
        debug_assert!(ny < self.size.ny);
        (nx, ny)
    }

    /// Computes the PBC-corrected closest image of `r1` from `r0`.
    ///
    /// # Panics
    ///
    /// If `r0` is not normalized.
    #[inline]
    #[must_use]
    pub fn mirror(&self, r0: &Pos, r1: &Pos) -> Pos {
        assert!(self.is_normalized(r0), "base pos. (r0) must be normalized");
        let (lx, ly) = self.simbox();
        let r1 = self.normalize(r1);
        let x1_m = if r0.x <= r1.x { r1.x - lx } else { r1.x + lx };
        let x = if (r1.x - r0.x).abs() <= (x1_m - r0.x).abs() {
            r1.x
        } else {
            x1_m
        };
        let y1_m = if r0.y <= r1.y { r1.y - ly } else { r1.y + ly };
        let y = if (r1.y - r0.y).abs() <= (y1_m - r0.y).abs() {
            r1.y
        } else {
            y1_m
        };
        na::vector![x, y]
    }

    /// Returns the base position of the given grid index.
    #[expect(clippy::cast_precision_loss)]
    #[inline]
    #[must_use]
    pub fn base(&self, (i, j): I2) -> Pos {
        na::vector![i as f64 * self.unit.dx, j as f64 * self.unit.dy]
    }
}

/// A chord defined by the closest distance and length.
#[derive(Clone, Copy, Debug, PartialEq)]
#[readonly::make]
pub struct Chord {
    /// Closest distance.
    pub rmin: f64,
    /// Half length.
    pub hlen: f64,
}

impl Chord {
    /// Creates a new [`Chord`].
    ///
    /// # Errors
    /// - If `rmin` is negative.
    /// - If `hlen` is not positive.
    #[inline]
    pub fn new(rmin: f64, hlen: f64) -> anyhow::Result<Self> {
        anyhow::ensure!(rmin >= 0.0, "rmin must be non-negative");
        anyhow::ensure!(hlen > 0.0, "half length must be positive");
        Ok(Self { rmin, hlen })
    }

    /// Computes the absolute offset from the closest point for a given distance `r`.
    ///
    /// # Notes
    ///
    /// Return value can be larger than `hlen`.
    ///
    /// # Errors
    /// If `r` is too small.
    #[inline]
    pub fn abs_d(&self, r: f64) -> anyhow::Result<f64> {
        let rmin = self.rmin;
        let th = r * r - rmin * rmin;
        anyhow::ensure!(th >= 0.0, "out of range");
        Ok(th.sqrt())
    }

    /// Length of the chord.
    #[inline]
    #[must_use]
    pub fn length(&self) -> f64 {
        2.0 * self.hlen
    }

    /// Maximum distance from the center.
    #[inline]
    #[must_use]
    pub fn rmax(&self) -> f64 {
        f64::hypot(self.rmin, self.hlen)
    }

    /// Creates a point on the chord with offset `d`.
    ///
    /// # Errors
    /// See [`OnChord::new`].
    #[inline]
    pub fn at_d(&self, d: f64) -> anyhow::Result<OnChord> {
        OnChord::new(*self, d)
    }

    /// Creates a point on the chord with distance `r.abs()` and the same sign of `r`.
    ///
    /// # Errors
    /// See [`OnChord::new`].
    #[inline]
    pub fn at_r(&self, r: f64) -> anyhow::Result<OnChord> {
        let d = f64::copysign(self.abs_d(r.abs())?, r);
        OnChord::new(*self, d)
    }
}

/// A point on a chord.
#[derive(Clone, Copy, Debug, PartialEq)]
#[readonly::make]
pub struct OnChord {
    /// The chord.
    pub chord: Chord,
    /// Offset from the closest point on the chord.
    pub d: f64,
}

impl OnChord {
    /// Creates a new [`OnChord`].
    ///
    /// # Errors
    /// If `d` is not in the range of the chord.
    ///
    #[inline]
    pub fn new(chord: Chord, d: f64) -> anyhow::Result<Self> {
        anyhow::ensure!(d.abs() <= chord.hlen, "out of the range");
        Ok(Self { chord, d })
    }

    /// Normalizes the position on the chord.
    ///
    /// # Notes
    /// - If `d.abs()` is `0.0`, it is normalized to `+0.0`.
    /// - If `d.abs()` is `hlen`, it is normalized to `-hlen`.
    #[expect(clippy::float_cmp, clippy::missing_panics_doc)]
    #[inline]
    #[must_use]
    pub fn normalize(self) -> Self {
        let ad = self.d.abs();
        let d = if ad == 0.0 {
            0.0
        } else if ad == self.chord.hlen {
            -self.chord.hlen
        } else {
            self.d
        };
        Self::new(self.chord, d).unwrap()
    }

    /// Computes the distance from the center.
    #[inline]
    #[must_use]
    pub fn r(&self) -> f64 {
        f64::hypot(self.chord.rmin, self.d)
    }

    /// Creates a point on the chord with distance `r` and the same sign of `d`.
    ///
    /// # Errors
    /// If `r` is out of the range.
    #[inline]
    pub fn try_from_r(&self, r: f64) -> anyhow::Result<Self> {
        let d = f64::copysign(self.chord.abs_d(r)?, self.d);
        Self::new(self.chord, d)
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;

    use super::*;

    #[test]
    fn test_mirror() {
        let unit = GridUnit::new(1.0, 2.0).unwrap();
        let size = GridSize::new(6, 3).unwrap();
        let gs = Grid::new(unit, size);
        let r0 = na::vector![3.0, 0.0];
        let r1 = na::vector![3.01, 5.99];
        let r1_ = gs.mirror(&r0, &r1);
        assert_approx_eq!(f64, r1.x, r1_.x);
        assert_approx_eq!(f64, r1.y, r1_.y + gs.simbox().1);
    }

    #[test]
    fn test_basic() {
        let ch = Chord::new(1.0, f64::sqrt(3.0)).unwrap();
        let pos = ch.at_d(f64::sqrt(3.0)).unwrap();
        assert_approx_eq!(f64, ch.rmax(), 2.0);
        assert_approx_eq!(f64, ch.length(), 2.0 * f64::sqrt(3.0));
        assert_approx_eq!(f64, pos.r(), 2.0);
    }

    #[test]
    fn test_normalize() -> anyhow::Result<()> {
        let ch = Chord::new(1.0, 1.0)?;
        let pos = ch.at_d(1.0)?.normalize();
        assert_approx_eq!(f64, pos.d, -1.0);
        let pos = ch.at_d(-0.0)?.normalize();
        assert_approx_eq!(f64, pos.d, 0.0);
        assert!(pos.d.is_sign_positive());
        Ok(())
    }

    #[test]
    fn test_try_from_r_pos() -> anyhow::Result<()> {
        let pos = Chord::new(1.0, f64::sqrt(3.0))?
            .at_d(0.1)?
            .try_from_r(f64::sqrt(2.0))?;
        assert_approx_eq!(f64, pos.d, 1.0);
        Ok(())
    }

    #[test]
    fn test_try_from_r_neg() -> anyhow::Result<()> {
        let pos = Chord::new(1.0, f64::sqrt(3.0))?
            .at_d(-0.1)?
            .try_from_r(f64::sqrt(2.0))?;
        assert_approx_eq!(f64, pos.d, -1.0);
        Ok(())
    }
}
