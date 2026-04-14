//! Lennard-Jones potential.

use crate::{
    traits::{DualPotential, InvPotential, Potential, PotentialDiff},
    types::{RootKindLeft, RootKindRight},
};

/// Lennard-Jones potential and related functions.
#[derive(Clone, Copy, Debug, PartialEq)]
#[readonly::make]
pub struct LJ {
    /// Energy scale.
    pub eps: f64,
    /// Length scale.
    pub sig: f64,
}

impl LJ {
    /// Creates a new [`LJ`] instance.
    ///
    /// # Errors
    /// - If `eps` is not positive.
    /// - If `sig` is not positive.
    #[inline]
    pub fn new(eps: f64, sig: f64) -> anyhow::Result<Self> {
        anyhow::ensure!(eps > 0.0, "eps must be positive");
        anyhow::ensure!(sig > 0.0, "sig must be positive");
        Ok(Self { eps, sig })
    }

    /// Returns the min of the potential.
    #[inline]
    #[must_use]
    pub fn min(&self) -> f64 {
        -self.eps
    }

    /// Returns the argmin of the potential.
    #[inline]
    #[must_use]
    pub fn argmin(&self) -> f64 {
        f64::powf(2.0, 1.0 / 6.0) * self.sig
    }
}

impl Potential for LJ {
    #[inline]
    fn u(&self, r: f64) -> f64 {
        debug_assert!(r >= 0.0);
        if r == 0.0 {
            f64::INFINITY
        } else {
            let sr = self.sig / r;
            let sr2 = sr * sr;
            let sr6 = sr2 * sr2 * sr2;
            4.0 * self.eps * (sr6 * sr6 - sr6)
        }
    }
}

impl PotentialDiff for LJ {
    #[inline]
    fn du(&self, r: f64) -> f64 {
        debug_assert!(r >= 0.0);
        if r == 0.0 {
            f64::NEG_INFINITY
        } else {
            let s6 = self.sig.powi(6);
            24.0 * self.eps * (-2.0 * s6 * s6 / r.powi(13) + s6 / r.powi(7))
        }
    }
}

impl DualPotential for LJ {
    #[inline]
    fn u_p(&self, r: f64) -> f64 {
        debug_assert!(r >= 0.0);
        if r <= self.argmin() {
            self.min()
        } else {
            self.u(r)
        }
    }

    #[inline]
    fn u_n(&self, r: f64) -> f64 {
        debug_assert!(r >= 0.0);
        if r >= self.argmin() {
            0.0
        } else {
            self.u(r) - self.min()
        }
    }
}

impl InvPotential for LJ {
    #[inline]
    fn iu_p(&self, e: f64) -> RootKindRight {
        if e < self.min() {
            RootKindRight::TooSmall
        } else if 0.0 < e {
            RootKindRight::TooLarge
        } else if e == 0.0 {
            RootKindRight::Right(f64::INFINITY)
        } else {
            let co = f64::sqrt(1.0 + e / self.eps);
            let r = f64::powf(2.0 / (1.0 - co), 1.0 / 6.0) * self.sig;
            RootKindRight::Right(r)
        }
    }

    #[inline]
    fn iu_n(&self, e: f64) -> RootKindLeft {
        if e < 0.0 {
            RootKindLeft::TooSmall
        } else {
            let co = f64::sqrt(e / self.eps);
            let r = f64::powf(2.0 / (1.0 + co), 1.0 / 6.0) * self.sig;
            RootKindLeft::Left(r)
        }
    }
}

#[cfg(test)]
mod tests {
    use core::f64;

    use float_cmp::assert_approx_eq;

    use super::*;

    #[test]
    fn test_new() {
        assert!(LJ::new(1.0, 1.0).is_ok());
        assert!(LJ::new(0.0, 1.0).is_err());
        assert!(LJ::new(1.0, 0.0).is_err());
    }

    #[test]
    fn test_u() {
        let lj = LJ::new(3.1, 0.4).unwrap();
        assert_approx_eq!(f64, lj.u(lj.sig), 0.0);
        assert_approx_eq!(f64, lj.u(lj.argmin()), -lj.eps);
        assert_approx_eq!(f64, lj.u(0.0), f64::INFINITY);
        assert_approx_eq!(f64, lj.u(f64::INFINITY), 0.0);
    }

    #[test]
    fn test_du() {
        let lj = LJ::new(3.1, 0.4).unwrap();
        assert_approx_eq!(f64, lj.du(lj.argmin()), 0.0, epsilon = 1e-12);
        assert_approx_eq!(f64, lj.du(0.0), f64::NEG_INFINITY);
        assert_approx_eq!(f64, lj.du(f64::INFINITY), 0.0);
    }

    #[test]
    fn test_dual() {
        let lj = LJ::new(3.1, 0.4).unwrap();
        for rr in [0.0, 0.5, 1.0, 2.0, f64::INFINITY] {
            let r = rr * lj.argmin();
            assert_approx_eq!(f64, lj.u(r), lj.u_p(r) + lj.u_n(r));
        }
    }

    #[test]
    fn test_dual_inv() {
        let lj = LJ::new(3.1, 0.4).unwrap();
        for ee in [f64::NEG_INFINITY, -2.0, -1.0, 0.0, 1.0, 2.0, f64::INFINITY] {
            let e = ee * lj.eps;
            match lj.iu_p(e) {
                RootKindRight::Right(r) => {
                    assert_approx_eq!(f64, lj.u_p(r), e);
                }
                RootKindRight::TooSmall => {
                    assert!(e < lj.min());
                }
                RootKindRight::TooLarge => {
                    assert!(0.0 < e);
                }
            }
            match lj.iu_n(e) {
                RootKindLeft::Left(r) => {
                    assert_approx_eq!(f64, lj.u_n(r), e, ulps = 10);
                }
                RootKindLeft::TooSmall => {
                    assert!(e < 0.0);
                }
                RootKindLeft::TooLarge => {
                    unreachable!()
                }
            }
        }
    }
}
