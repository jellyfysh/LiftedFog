//! ECMC samplers based on energy difference and inverse potentials.

use crate::{
    geom::OnChord,
    traits::InvPotential,
    types::{RootKindLeft, RootKindRight},
};

/// How the sampling ended.
#[derive(Clone, Copy, Debug, PartialEq)]
enum DeResult {
    /// Reached the end.
    Continue(f64),
    /// Stopped before the end.
    Stop(f64),
}

impl From<DeResult> for f64 {
    #[inline]
    fn from(value: DeResult) -> Self {
        match value {
            DeResult::Continue(l) | DeResult::Stop(l) => l,
        }
    }
}

/// ECMC sampling based on energy difference and inverse potentials.
pub trait EDiffSampler {
    /// Samples the distance.
    ///
    /// # Arguments
    /// - `de`: Energy difference sampled in advance.
    fn sample_impl(&self, de: f64, pos: OnChord, hint: f64) -> Option<f64>;
}

/// Mutable state for [`EDiffSampler`].
#[derive(Clone, Debug, PartialEq)]
struct EDiffSamplerImpl<'a, T: InvPotential> {
    /// Underlying potential.
    inner: &'a T,
    /// Energy difference left.
    de: f64,
    /// Current position.
    pos: OnChord,
    // Cached values
    rmax_: f64,   // rmax
    up_max_: f64, // u_p @ rmax
    un_min_: f64, // u_n @ rmin
}

impl<'a, T: InvPotential> EDiffSamplerImpl<'a, T> {
    /// Initializes the state.
    fn new(inner: &'a T, de: f64, pos: OnChord) -> Self {
        assert!(de >= 0.0);
        let rmax_ = pos.chord.rmax();
        let up_max_ = inner.u_p(rmax_);
        let un_min_ = inner.u_n(pos.chord.rmin);
        Self {
            inner,
            de,
            pos,
            rmax_,
            up_max_,
            un_min_,
        }
    }

    /// Energy difference for a full cycle.
    fn de_cyc(&self) -> f64 {
        let ch = self.pos.chord;
        // TODO: Check if this is compatible with divergent potentials
        let cyc_p = self.up_max_ - self.inner.u_p(ch.rmin);
        debug_assert!(cyc_p >= 0.0);
        let cyc_n = self.un_min_ - self.inner.u_n(self.rmax_);
        debug_assert!(cyc_n >= 0.0);
        cyc_p + cyc_n
    }

    /// Takes care of full cycles.
    fn simulate_cyc(&mut self, hint: f64) -> Option<f64> {
        let de_cyc = self.de_cyc();
        if de_cyc == 0.0 {
            // Avoid NaN
            self.de = 0.0;
            return Some(f64::INFINITY);
        }
        let l_cyc = self.pos.chord.length();
        let ret = (self.de / de_cyc).floor() * l_cyc;
        (ret <= hint).then(|| {
            self.de %= de_cyc;
            ret
        })
    }

    /// Used where `val` is supposed to be non-negative.
    fn sanitize(val: f64) -> f64 {
        if val < 0.0 {
            // log::info!("negative value encountered: {val}");
            0.0
        } else {
            val
        }
    }

    /// Takes care of moves towards plus infinity.
    fn simulate_impl_plus(&mut self) -> DeResult {
        debug_assert!(self.pos.d.is_sign_positive());
        debug_assert!(self.pos.d >= 0.0);
        let ch = self.pos.chord;
        let r = self.pos.r();
        let e = self.inner.u_p(r);
        let e_ub = self.up_max_;
        let e_stop = e + self.de;
        let d = self.pos.d;
        if e_ub <= e_stop {
            // Guaranteed to reach rmax
            self.de -= e_ub - e;
            self.de = Self::sanitize(self.de);
            self.pos = ch.at_d(-ch.hlen).unwrap();
            DeResult::Continue(ch.hlen - d)
        } else {
            // Calculation ends
            self.de = 0.0;
            let RootKindRight::Right(r_stop) = self.inner.iu_p(e_stop) else {
                unreachable!("can evaluate at e_stop")
            };
            self.pos = self.pos.try_from_r(r_stop).expect("cannot reach rmax");
            DeResult::Stop(Self::sanitize(self.pos.d - d))
        }
    }

    /// Takes care of moves towards zero.
    fn simulate_impl_minus(&mut self) -> DeResult {
        debug_assert!(self.pos.d < 0.0);
        let ch = self.pos.chord;
        let r = self.pos.r();
        let e = self.inner.u_n(r);
        let e_ub = self.un_min_;
        let e_stop = e + self.de;
        let d = self.pos.d;
        if e_ub <= e_stop {
            // Guaranteed to reach rmin
            self.de -= e_ub - e;
            self.de = Self::sanitize(self.de);
            self.pos = ch.at_d(0.0).unwrap();
            DeResult::Continue(-d)
        } else {
            // Calculation ends
            self.de = 0.0;
            let RootKindLeft::Left(r_stop) = self.inner.iu_n(e_stop) else {
                unreachable!("can evaluate at e_stop")
            };
            self.pos = self.pos.try_from_r(r_stop).expect("cannot reach rmin");
            DeResult::Stop(Self::sanitize(self.pos.d - d))
        }
    }

    /// One step of the simulation.
    fn simulate_impl(&mut self) -> DeResult {
        self.pos = self.pos.normalize();
        let ret = if self.pos.d >= 0.0 {
            self.simulate_impl_plus()
        } else {
            self.simulate_impl_minus()
        };
        debug_assert!(self.de >= 0.0);
        log::debug!("move by {ret:?} (left {})", self.de);
        ret
    }

    /// Consumes self and computes the result.
    #[expect(clippy::float_cmp)]
    fn simulate(mut self, hint: f64) -> Option<f64> {
        debug_assert!(hint >= 0.0);
        let mut ret = self.simulate_cyc(hint)?;
        // At most three moves
        //   r -> rmin -> rmax -> rmin
        //   r -> rmax -> rmin -> rmax
        for _ in 0..3 {
            if ret > hint {
                log::debug!("no need to sim.: {ret}");
                return None;
            }
            debug_assert!(self.de >= 0.0);
            let res = self.simulate_impl();
            ret += f64::from(res);
            if let DeResult::Stop(_) = res {
                break;
            }
        }
        debug_assert_eq!(self.de, 0.0);
        (ret <= hint).then_some(ret)
    }
}

impl<T: InvPotential> EDiffSampler for T {
    #[inline]
    fn sample_impl(&self, de: f64, pos: OnChord, hint: f64) -> Option<f64> {
        EDiffSamplerImpl::new(self, de, pos).simulate(hint)
    }
}

#[cfg(test)]
mod tests {
    use core::f64;

    use float_cmp::assert_approx_eq;
    use rand::{Rng, RngExt};

    use super::*;
    use crate::{
        geom::Chord,
        lj::LJ,
        traits::{DualPotential, InvPotential},
    };

    /// Generates a random position on a random chord.
    fn rand_pos(rng: &mut impl Rng) -> OnChord {
        let s = 10.0;
        let ch = Chord::new(rng.random_range(0.0..=s), rng.random_range(0.0..=s)).unwrap();
        ch.at_d(rng.random_range(-ch.hlen..=ch.hlen)).unwrap()
    }

    #[test]
    fn test_trivial() {
        struct Trivial;

        impl DualPotential for Trivial {
            fn u_p(&self, _: f64) -> f64 {
                0.0
            }

            fn u_n(&self, _: f64) -> f64 {
                0.0
            }
        }

        impl InvPotential for Trivial {
            fn iu_p(&self, _: f64) -> RootKindRight {
                unimplemented!()
            }

            fn iu_n(&self, _: f64) -> RootKindLeft {
                unimplemented!()
            }
        }

        let mut rng = rand::rng();
        let smp = Trivial;
        for _ in 0..10000 {
            let pos = rand_pos(&mut rng);
            for de in [0.0, rng.random()] {
                for hint in [rng.random(), f64::INFINITY] {
                    assert_eq!(
                        smp.sample_impl(de, pos, hint),
                        hint.is_infinite().then_some(f64::INFINITY)
                    );
                }
            }
        }
    }

    #[test]
    fn test_plinear() {
        struct PLinear;

        impl DualPotential for PLinear {
            fn u_p(&self, r: f64) -> f64 {
                (r - 2.0).max(0.0)
            }

            fn u_n(&self, r: f64) -> f64 {
                (1.0 - r).max(0.0)
            }
        }

        impl InvPotential for PLinear {
            fn iu_p(&self, e: f64) -> RootKindRight {
                if e < 0.0 {
                    RootKindRight::TooSmall
                } else {
                    RootKindRight::Right(e + 2.0)
                }
            }

            fn iu_n(&self, e: f64) -> RootKindLeft {
                if e < 0.0 {
                    RootKindLeft::TooSmall
                } else if e > 1.0 {
                    RootKindLeft::TooLarge
                } else {
                    RootKindLeft::Left(1.0 - e)
                }
            }
        }

        let smp = PLinear;
        let ch = Chord::new(0.0, 3.0).unwrap();

        let pos = ch.at_r(-0.5).unwrap();
        assert_approx_eq!(f64, smp.sample_impl(0.0, pos, f64::INFINITY).unwrap(), 0.0);
        assert_approx_eq!(f64, smp.sample_impl(0.5, pos, f64::INFINITY).unwrap(), 2.5);
        assert_approx_eq!(f64, smp.sample_impl(1.0, pos, f64::INFINITY).unwrap(), 3.0);
        assert_approx_eq!(
            f64,
            smp.sample_impl(10.5, pos, f64::INFINITY).unwrap(),
            32.5
        );

        let pos = ch.at_r(0.5).unwrap();
        assert_approx_eq!(f64, smp.sample_impl(0.0, pos, f64::INFINITY).unwrap(), 1.5);
        assert_approx_eq!(f64, smp.sample_impl(0.5, pos, f64::INFINITY).unwrap(), 2.0);
        assert_approx_eq!(f64, smp.sample_impl(1.0, pos, f64::INFINITY).unwrap(), 4.5);
        assert_approx_eq!(
            f64,
            smp.sample_impl(10.5, pos, f64::INFINITY).unwrap(),
            32.0
        );

        let pos = ch.at_r(1.5).unwrap();
        assert_approx_eq!(f64, smp.sample_impl(0.0, pos, f64::INFINITY).unwrap(), 0.5);
        assert_approx_eq!(
            f64,
            smp.sample_impl(10.5, pos, f64::INFINITY).unwrap(),
            31.0
        );

        let pos = ch.at_r(-2.5).unwrap();
        assert_approx_eq!(
            f64,
            smp.sample_impl(10.5, pos, f64::INFINITY).unwrap(),
            32.0
        );

        let pos = ch.at_r(0.0).unwrap();
        assert_approx_eq!(f64, smp.sample_impl(0.5, pos, f64::INFINITY).unwrap(), 2.5);

        let pos = ch.at_r(-0.0).unwrap();
        assert_approx_eq!(f64, smp.sample_impl(0.5, pos, f64::INFINITY).unwrap(), 2.5);

        let pos = ch.at_r(ch.rmax()).unwrap();
        assert_approx_eq!(f64, smp.sample_impl(0.5, pos, f64::INFINITY).unwrap(), 2.5);
    }

    #[test]
    fn test_coulomb() {
        struct Coulomb;

        impl DualPotential for Coulomb {
            fn u_p(&self, _: f64) -> f64 {
                0.0
            }

            fn u_n(&self, r: f64) -> f64 {
                1.0 / r
            }
        }

        impl InvPotential for Coulomb {
            fn iu_p(&self, e: f64) -> RootKindRight {
                if e < 0.0 {
                    RootKindRight::TooSmall
                } else if e > 0.0 {
                    RootKindRight::TooLarge
                } else {
                    RootKindRight::Right(f64::INFINITY)
                }
            }

            fn iu_n(&self, e: f64) -> RootKindLeft {
                if e <= 0.0 {
                    RootKindLeft::TooSmall
                } else {
                    RootKindLeft::Left(1.0 / e)
                }
            }
        }

        let smp = Coulomb;
        let ch = Chord::new(1.0, f64::sqrt(3.0)).unwrap();

        let pos = ch.at_d(1.0).unwrap();
        assert_approx_eq!(
            f64,
            smp.sample_impl(0.0, pos, f64::INFINITY).unwrap(),
            f64::sqrt(3.0) - 1.0
        );
        assert_approx_eq!(
            f64,
            smp.sample_impl(0.25, pos, f64::INFINITY).unwrap(),
            2.0 * f64::sqrt(3.0) - 1.0 - f64::sqrt(7.0) / 3.0
        );
        assert_approx_eq!(
            f64,
            smp.sample_impl(0.5, pos, f64::INFINITY).unwrap(),
            3.0 * f64::sqrt(3.0) - 1.0
        );

        let pos = ch.at_d(-1.0).unwrap();
        assert_approx_eq!(f64, smp.sample_impl(0.0, pos, f64::INFINITY).unwrap(), 0.0);
        assert_approx_eq!(
            f64,
            smp.sample_impl(0.25, pos, f64::INFINITY).unwrap(),
            1.0 - f64::sqrt(95.0 - 64.0 * f64::sqrt(2.0)) / 7.0,
            ulps = 6
        );
        assert_approx_eq!(
            f64,
            smp.sample_impl(0.5, pos, f64::INFINITY).unwrap(),
            2.0 * f64::sqrt(3.0)
        );

        let ch = Chord::new(0.0, 2.0).unwrap();

        let pos = ch.at_d(1.0).unwrap();
        assert_approx_eq!(f64, smp.sample_impl(0.0, pos, f64::INFINITY).unwrap(), 1.0);
        assert_approx_eq!(
            f64,
            smp.sample_impl(0.25, pos, f64::INFINITY).unwrap(),
            5.0 / 3.0
        );
        assert_approx_eq!(f64, smp.sample_impl(0.5, pos, f64::INFINITY).unwrap(), 2.0);

        let pos = ch.at_d(-1.0).unwrap();
        assert_approx_eq!(f64, smp.sample_impl(0.0, pos, f64::INFINITY).unwrap(), 0.0);
        assert_approx_eq!(
            f64,
            smp.sample_impl(0.25, pos, f64::INFINITY).unwrap(),
            1.0 / 5.0
        );
        assert_approx_eq!(
            f64,
            smp.sample_impl(0.5, pos, f64::INFINITY).unwrap(),
            1.0 / 3.0
        );
    }

    #[test]
    fn test_lj() {
        let lj = LJ::new(1.0, 1.0).unwrap();
        let ch = Chord::new(0.5, f64::sqrt(35.0 / 4.0)).unwrap();

        let pos = ch.at_r(1.0).unwrap();
        assert_approx_eq!(
            f64,
            lj.sample_impl(0.0, pos, f64::INFINITY).unwrap(),
            f64::sqrt(f64::powf(2.0, 1.0 / 3.0) - 1.0 / 4.0) - f64::sqrt(3.0) / 2.0
        );
        assert_approx_eq!(
            f64,
            lj.sample_impl(0.5, pos, f64::INFINITY).unwrap(),
            f64::sqrt(f64::powf(4.0 + 2.0 * f64::sqrt(2.0), 1.0 / 3.0) - 1.0 / 4.0)
                - f64::sqrt(3.0) / 2.0
        );

        let pos = ch.at_r(-1.0).unwrap();
        assert_approx_eq!(f64, lj.sample_impl(0.0, pos, f64::INFINITY).unwrap(), 0.0);
    }
}
