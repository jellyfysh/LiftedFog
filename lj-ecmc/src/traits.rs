//! Traits for ECMC sampling.

use nalgebra::Vector2;
use rand::Rng;

use crate::types::{Event, MoveDirection, Pos, RootKindLeft, RootKindRight, SamplingContext};

/// Radial pair potential.
///
/// Can be inifinite at `r = 0`, but [`f64::NAN`] is anywhere forbidden. Also see [`DualPotential`].
pub trait Potential {
    /// Evaluates the pair potential at distance `r`.
    fn u(&self, r: f64) -> f64;
}

/// `r` derivative of [`Potential`].
pub trait PotentialDiff: Potential {
    /// Evaluates the derivative of `u` **WITHOUT** the negative sign.
    fn du(&self, r: f64) -> f64;
}

/// Potential split into increasing/decreasing parts.
///
/// `u_p` and `u_n` must satisfy the following conditions, even at `r = 0`:
/// - `r` derivative of `u_p` must be equal to `du(r).max(0)`.
/// - `r` derivative of `u_n` must be equal to `du(r).min(0)`.
/// - `u_p + u_n` must be equal to `u`.
pub trait DualPotential {
    /// Evaluates the increasing part of [`Potential::u`].
    fn u_p(&self, r: f64) -> f64;
    /// Evaluates the decreasing part of [`Potential::u`].
    fn u_n(&self, r: f64) -> f64;
}

/// Inverse of [`DualPotential`].
pub trait InvPotential: DualPotential {
    /// Inverse of [`DualPotential::u_p`].
    fn iu_p(&self, e: f64) -> RootKindRight;
    /// Inverse of [`DualPotential::u_n`].
    fn iu_n(&self, e: f64) -> RootKindLeft;
}

/// Interface for emitting ECMC events.
pub trait EmitEvent {
    /// Output type: either [`Event`] or [`Option<Event>`].
    type EventLike;

    /// Emits an event based on the context.
    ///
    /// # Parameters
    /// - `rng`: Random number generator.
    /// - `ctx`: Context containing sampling information.
    /// - `hint`: Duration before the current earliest event fires.
    fn emit(
        &self,
        rng: &mut impl Rng,
        ctx: SamplingContext<'_>,
        hint: Option<f64>,
    ) -> Self::EventLike;
}

/// Interface for modifying state based on events.
pub trait ApplyEvent {
    /// Applies an event to the state.
    fn apply(&mut self, ev: &Event);
}

/// Interface for putting particles into manager structs.
pub trait AddP {
    /// Adds a particle.
    #[expect(clippy::missing_errors_doc)]
    fn add(&mut self, r: &Pos) -> anyhow::Result<()>;

    /// Adds multiple particles from a slice.
    #[expect(clippy::missing_errors_doc)]
    #[inline]
    fn add_from_slice(&mut self, slice: &[Pos]) -> anyhow::Result<()> {
        slice.iter().try_for_each(|disk| self.add(disk).map(drop))
    }
}

/// Moves a particle.
pub trait MoveP {
    /// Moves a particle in the specified direction by distance `d`.
    #[must_use]
    fn move_by(&self, dir: MoveDirection, d: f64) -> Self;
}

impl MoveP for Vector2<f64> {
    #[inline]
    fn move_by(&self, dir: MoveDirection, d: f64) -> Self {
        let mut ret = *self;
        ret[dir] += d;
        ret
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;
    use nalgebra as na;

    use super::*;

    #[test]
    fn test_move_by() {
        let r = na::vector![0.0, 0.0];
        let moved = r.move_by(MoveDirection::X, 1.0);
        assert_approx_eq!(f64, moved.x, 1.0);
        assert_approx_eq!(f64, moved.y, 0.0);
        let moved = r.move_by(MoveDirection::Y, 1.0);
        assert_approx_eq!(f64, moved.x, 0.0);
        assert_approx_eq!(f64, moved.y, 1.0);
    }
}
