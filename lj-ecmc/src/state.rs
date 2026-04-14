//! Maintains state.

use core::ops::Deref;

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    geom::Grid,
    traits::{AddP, ApplyEvent, EmitEvent},
    types::{Event, EventKind, I1, Pos, SamplingContext},
};

/// Simulation record.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[readonly::make]
pub struct Record {
    /// Simulation box.
    pub simbox: (f64, f64),
    /// Particle positions.
    pub data: Vec<Pos>,
}

/// Particle state.
#[derive(Clone, Debug, PartialEq)]
pub struct State {
    /// Grid spec.
    gs: Grid,
    /// Particle positions.
    data: Vec<Pos>,
}

impl State {
    /// Creates an empty [`State`].
    #[inline]
    #[must_use]
    pub const fn new(gs: Grid) -> Self {
        Self {
            gs,
            data: Vec::new(),
        }
    }

    #[inline]
    fn update(&mut self, i: I1, r: &Pos) -> anyhow::Result<()> {
        anyhow::ensure!(i < self.data.len(), "index out of bounds");
        anyhow::ensure!(self.gs.is_normalized(r), "r out of bounds");
        log::debug!("update {i}: {:?} -> {r:?}", self.data[i]);
        self.data[i] = *r;
        Ok(())
    }

    #[inline]
    #[must_use]
    pub fn dump(&self) -> Record {
        Record {
            simbox: self.gs.simbox(),
            data: self.data.clone(),
        }
    }
}

impl AddP for State {
    #[inline]
    fn add(&mut self, r: &Pos) -> anyhow::Result<()> {
        anyhow::ensure!(self.gs.is_normalized(r), "r out of bounds");
        self.data.push(*r);
        Ok(())
    }
}

impl EmitEvent for State {
    type EventLike = Event;

    #[inline]
    fn emit(
        &self,
        _: &mut impl Rng,
        ctx @ (_, ss): SamplingContext<'_>,
        _: Option<f64>,
    ) -> Self::EventLike {
        let r = self.data[ss.id];
        let time = self.gs.simbox()[ss.dir] - r[ss.dir];
        log::debug!("bound: {time}");
        Event::without_peer(ctx, EventKind::Boundary, time).unwrap()
    }
}

impl ApplyEvent for State {
    #[inline]
    fn apply(&mut self, ev: &Event) {
        let patch = ev.patch();
        debug_assert!(self.gs.is_normalized(&patch));
        self.update(ev.id, &patch).unwrap();
    }
}

impl Deref for State {
    type Target = [Pos];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;
    use nalgebra as na;

    use super::*;
    use crate::{
        geom::{GridSize, GridUnit},
        types::{MoveDirection, SamplingState},
    };

    #[test]
    fn test_add() -> anyhow::Result<()> {
        let gs = Grid::new(GridUnit::new(1.0, 1.0)?, GridSize::new(4, 4)?);
        let mut state = State::new(gs);
        assert!(state.data.is_empty());
        let r = na::vector![0.5, 0.5];
        state.add(&r)?;
        assert_eq!(state.data[0], r);
        Ok(())
    }

    #[test]
    fn test_update() -> anyhow::Result<()> {
        let gs = Grid::new(GridUnit::new(1.0, 1.0)?, GridSize::new(4, 4)?);
        let mut state = State::new(gs);
        state.add(&na::vector![0.5, 0.5])?;
        state.add(&na::vector![0.0, 0.0])?;
        let r0 = na::vector![1.5, 1.5];
        state.update(0, &r0)?;
        assert_eq!(state.data[0], r0);
        Ok(())
    }

    #[test]
    fn test_emit() -> anyhow::Result<()> {
        let gs = Grid::new(GridUnit::new(1.0, 1.0)?, GridSize::new(4, 4)?);
        let mut state = State::new(gs);
        state.add(&na::vector![0.3, 0.7])?;
        let mut rng = rand::rng();
        let ss = SamplingState::new(MoveDirection::X, 0, 100.0)?;
        let ev = state.emit(&mut rng, (&state.data, &ss), None);
        assert_approx_eq!(f64, ev.after, 3.7);
        let ss = SamplingState::new(MoveDirection::Y, 0, 100.0)?;
        let ev = state.emit(&mut rng, (&state.data, &ss), None);
        assert_approx_eq!(f64, ev.after, 3.3);
        Ok(())
    }
}
