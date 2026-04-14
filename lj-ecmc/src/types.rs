//! Common types.

use core::ops::{Index, IndexMut};

use nalgebra::Vector2;
use rand::{Rng, RngExt};
use rand_distr::{Distribution, StandardUniform};

use crate::traits::{ApplyEvent, EmitEvent};

/// Solution of `f(x) = y` for a given `y` and a non-decreasing/continuous `f`.
#[derive(Clone, Debug, PartialEq)]
pub enum RootKindRight {
    /// Cannot find a `x` because `y` is too small.
    TooSmall,
    /// Found roots: return the largest one.
    Right(f64),
    /// Cannot find a `x` because `y` is too large.
    TooLarge,
}

/// Solution of `f(x) = y` for a given `y` and a non-increasing/continuous `f`.
#[derive(Clone, Debug, PartialEq)]
pub enum RootKindLeft {
    /// Cannot find a `x` because `y` is too small.
    TooSmall,
    /// Found roots: return the smallest one.
    Left(f64),
    /// Cannot find a `x` because `y` is too large.
    TooLarge,
}

/// Move direction.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MoveDirection {
    /// X direction.
    X,
    /// Y direction.
    Y,
}

impl MoveDirection {
    /// Returns the opposite direction.
    ///
    /// # Examples
    /// ```
    /// # use lj_ecmc::types::MoveDirection;
    /// assert_eq!(MoveDirection::X.opposite(), MoveDirection::Y);
    /// assert_eq!(MoveDirection::Y.opposite(), MoveDirection::X);
    /// ```
    #[inline]
    #[must_use]
    pub const fn opposite(self) -> Self {
        match self {
            MoveDirection::X => MoveDirection::Y,
            MoveDirection::Y => MoveDirection::X,
        }
    }
}

impl<T> Index<MoveDirection> for Vector2<T> {
    type Output = T;

    #[inline]
    fn index(&self, dir: MoveDirection) -> &Self::Output {
        match dir {
            MoveDirection::X => &self[0],
            MoveDirection::Y => &self[1],
        }
    }
}

impl<T> IndexMut<MoveDirection> for Vector2<T> {
    #[inline]
    fn index_mut(&mut self, dir: MoveDirection) -> &mut Self::Output {
        match dir {
            MoveDirection::X => &mut self[0],
            MoveDirection::Y => &mut self[1],
        }
    }
}

impl<T> Index<MoveDirection> for (T, T) {
    type Output = T;

    #[inline]
    fn index(&self, dir: MoveDirection) -> &Self::Output {
        match dir {
            MoveDirection::X => &self.0,
            MoveDirection::Y => &self.1,
        }
    }
}

impl<T> IndexMut<MoveDirection> for (T, T) {
    #[inline]
    fn index_mut(&mut self, dir: MoveDirection) -> &mut Self::Output {
        match dir {
            MoveDirection::X => &mut self.0,
            MoveDirection::Y => &mut self.1,
        }
    }
}

impl Distribution<MoveDirection> for StandardUniform {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> MoveDirection {
        match rng.random_range(0..2) {
            0 => MoveDirection::X,
            _ => MoveDirection::Y,
        }
    }
}

/// Static branch selection.
pub type BranchTag = bool;

/// Tag for X branch.
pub const BRANCH_X: BranchTag = true;
/// Tag for Y branch.
pub const BRANCH_Y: BranchTag = false;

/// 1D index.
pub type I1 = usize;

/// 2D index.
pub type I2 = (I1, I1);

/// Particle position.
pub type Pos = Vector2<f64>;
pub type I1Pos = (I1, Pos);

/// Sampling context.
#[derive(Clone, Debug, PartialEq)]
#[readonly::make]
pub struct SamplingState {
    /// Direction of particle movement.
    pub dir: MoveDirection,
    /// Id of the particle currently moving.
    pub id: I1,
    /// Remaining time until the current MCS ends.
    pub time: f64,
}

impl SamplingState {
    /// Creates a new [`SamplingState`].
    ///
    /// # Errors
    /// - If `time` is negative.
    #[inline]
    pub fn new(dir: MoveDirection, id: I1, time: f64) -> anyhow::Result<Self> {
        anyhow::ensure!(time >= 0.0, "remaining time must be non-negative");
        Ok(Self { dir, id, time })
    }
}

impl EmitEvent for SamplingState {
    type EventLike = Event;

    #[inline]
    fn emit(
        &self,
        _: &mut impl Rng,
        ctx @ (_, ss): SamplingContext<'_>,
        _: Option<f64>,
    ) -> Self::EventLike {
        debug_assert_eq!(self, ss);
        log::debug!("timeout: {}", self.time);
        Event::without_peer(ctx, EventKind::Timeout, self.time).unwrap()
    }
}

impl ApplyEvent for SamplingState {
    #[inline]
    fn apply(&mut self, ev: &Event) {
        debug_assert_eq!(self.id, ev.id);
        debug_assert_eq!(self.dir, ev.dir);
        self.id = ev.next_id();
        self.time -= ev.after;
        debug_assert!(self.time >= 0.0);
    }
}

pub type SamplingContext<'a> = (&'a [Pos], &'a SamplingState);

/// Kind of event.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum EventKind {
    /// Particle reached box boundary.
    Boundary,
    /// Need to restart cell-veto sampling.
    Grid,
    /// fMet filter resulted in a rejection.
    Stop,
    /// Current MCS has ended.
    Timeout,
}

/// ECMC event.
#[derive(Clone, Debug, PartialEq)]
#[readonly::make]
pub struct Event {
    /// Kind of event.
    pub kind: EventKind,
    /// Direction of particle movement.
    pub dir: MoveDirection,
    /// Id of the particle currently moving.
    pub id: I1,
    /// When the event occurs.
    pub after: f64,
    /// Initial position of the particle.
    pub start: Pos,
    /// Optional peer particle involved in the event.
    pub peer: Option<I1Pos>,
}

impl Event {
    /// Creates a new [`Event`] instance without peer.
    #[expect(clippy::missing_errors_doc)]
    #[inline]
    pub fn without_peer(
        ctx: SamplingContext<'_>,
        kind: EventKind,
        after: f64,
    ) -> anyhow::Result<Self> {
        let (rs, ss) = ctx;
        anyhow::ensure!(matches!(
            kind,
            EventKind::Boundary | EventKind::Grid | EventKind::Timeout
        ));
        anyhow::ensure!(after >= 0.0, "after must be non-negative");
        Ok(Self {
            kind,
            dir: ss.dir,
            id: ss.id,
            after,
            start: rs[ss.id],
            peer: None,
        })
    }

    /// Creates a new [`Event`] instance with peer.
    #[expect(clippy::missing_errors_doc)]
    #[inline]
    pub fn with_peer(
        ctx: SamplingContext<'_>,
        kind: EventKind,
        after: f64,
        peer: I1Pos,
    ) -> anyhow::Result<Self> {
        let (rs, ss) = ctx;
        anyhow::ensure!(after >= 0.0, "after must be non-negative");
        anyhow::ensure!(matches!(kind, EventKind::Stop));
        Ok(Self {
            kind,
            dir: ss.dir,
            id: ss.id,
            after,
            start: rs[ss.id],
            peer: Some(peer),
        })
    }

    /// Selects the earlier event.
    #[inline]
    #[must_use]
    pub fn earlier(ev1: Event, ev2: Event) -> Event {
        if ev1.after <= ev2.after { ev1 } else { ev2 }
    }

    /// Returns the particle position after the event.
    #[inline]
    #[must_use]
    pub fn stop(&self) -> Pos {
        let mut stop = self.start;
        stop[self.dir] += self.after;
        stop
    }

    /// Returns the final particle position after the event.
    ///
    /// # Notes
    /// This method is very similar to `stop` except it takes the PBC into account.
    #[inline]
    #[must_use]
    pub fn patch(&self) -> Pos {
        let mut ret = self.stop();
        if self.kind == EventKind::Boundary {
            ret[self.dir] = 0.0;
        }
        ret
    }

    /// Returns the particle id moving next.
    #[inline]
    #[must_use]
    pub const fn next_id(&self) -> I1 {
        if let Some((id, _)) = self.peer {
            id
        } else {
            self.id
        }
    }

    /// Returns the extra distance earned.
    #[inline]
    #[must_use]
    pub fn extra(&self) -> f64 {
        if let Some((_, peer)) = self.peer {
            peer[self.dir] - self.stop()[self.dir]
        } else {
            0.0
        }
    }

    /// Returns the PBC-corrected pointer displacement along the vertical direction.
    #[inline]
    #[must_use]
    pub fn drift(&self) -> f64 {
        if let Some((_, peer)) = self.peer {
            let idir = self.dir.opposite();
            peer[idir] - self.start[idir]
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use float_cmp::assert_approx_eq;
    use nalgebra as na;

    use super::*;

    #[test]
    fn test_index_tuple() {
        let mut v = (1, 2);
        assert_eq!(v[MoveDirection::X], 1);
        assert_eq!(v[MoveDirection::Y], 2);
        v[MoveDirection::X] = 3;
        v[MoveDirection::Y] = 4;
        assert_eq!(v[MoveDirection::X], 3);
        assert_eq!(v[MoveDirection::Y], 4);
    }

    #[test]
    fn test_index_vector() {
        let mut v = na::vector![1, 2];
        assert_eq!(v[MoveDirection::X], 1);
        assert_eq!(v[MoveDirection::Y], 2);
        v[MoveDirection::X] = 3;
        v[MoveDirection::Y] = 4;
        assert_eq!(v[MoveDirection::X], 3);
        assert_eq!(v[MoveDirection::Y], 4);
    }

    #[test]
    fn test_dir_sample() {
        let mut rng = rand::rng();
        let mut cnt =
            HashMap::<_, usize>::from_iter([(MoveDirection::X, 0), (MoveDirection::Y, 0)]);
        for _ in 0..100 {
            let dir: MoveDirection = rng.random();
            *cnt.get_mut(&dir).unwrap() += 1;
        }
        assert!(cnt[&MoveDirection::X] > 0);
        assert!(cnt[&MoveDirection::Y] > 0);
    }

    #[test]
    fn test_patch_normal() -> anyhow::Result<()> {
        let rs = [na::vector![0.0, 0.0]];
        let ss = SamplingState::new(MoveDirection::X, 0, 10.0)?;
        let ev = Event::without_peer((&rs, &ss), EventKind::Timeout, 0.1)?;
        let patch = ev.patch();
        assert_approx_eq!(f64, patch.x, ev.after);
        Ok(())
    }

    #[test]
    fn test_patch_bound() -> anyhow::Result<()> {
        let rs = [na::vector![0.0, 0.0]];
        let ss = SamplingState::new(MoveDirection::Y, 0, 10.0)?;
        // Invalid event for testing (box)
        let ev = Event::without_peer((&rs, &ss), EventKind::Boundary, 1E-10)?;
        let patch = ev.patch();
        // PBC applied
        assert_approx_eq!(f64, patch.y, 0.0);
        Ok(())
    }
}
