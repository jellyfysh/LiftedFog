//! ECMC sampler.

use std::collections::BTreeSet;

use nalgebra::DMatrix;
use rand::{Rng, RngExt};
use rand_distr::{Distribution, Exp1};

use crate::{
    cv_sampler::{CellVetoSampler, CellVetoTable, MARGIN_B},
    de_sampler::EDiffSampler,
    geom::{Chord, Grid, OnChord},
    lj::LJ,
    traits::{AddP, ApplyEvent, EmitEvent, InvPotential, MoveP, PotentialDiff},
    types::{
        BRANCH_X, BRANCH_Y, BranchTag, Event, EventKind, I1, I2, MoveDirection, Pos,
        SamplingContext,
    },
};

/// Grid state.
#[derive(Clone, Debug, PartialEq)]
struct GState {
    /// Grid spec.
    gs: Grid,
    /// Particle id to grid.
    i2g: Vec<I2>,
    /// Grid to particle ids.
    g2i: DMatrix<Vec<I1>>,
    /// Grids having surplus particles.
    sur: BTreeSet<I2>, // MEMO: Must have small capacity
}

impl GState {
    /// Creates an empty [`GState`].
    #[inline]
    #[must_use]
    fn new(gs: Grid) -> Self {
        Self {
            gs,
            i2g: Vec::new(),
            g2i: DMatrix::from_fn(gs.size.nx, gs.size.ny, |_, _| Vec::new()),
            sur: BTreeSet::new(),
        }
    }

    /// Updates the surplus set for grid `ij`.
    fn update_sur(&mut self, ij: I2) {
        match self.g2i[ij].len() {
            0 | 1 => {
                self.sur.remove(&ij);
            }
            _ => {
                self.sur.insert(ij);
            }
        }
    }

    /// Runs sanity checks.
    fn sanitize(&self, i: I1, r: &Pos) -> anyhow::Result<()> {
        let g = self.i2g[i];
        anyhow::ensure!(g == self.gs.gindex(r));
        anyhow::ensure!(self.g2i[g].contains(&i));
        anyhow::ensure!(self.sur.contains(&g) == (self.g2i[g].len() > 1));
        Ok(())
    }

    /// Updates the position of particle `i`.
    #[inline]
    fn update(&mut self, i: I1, r: &Pos) -> anyhow::Result<()> {
        anyhow::ensure!(i < self.i2g.len(), "index out of bounds");
        anyhow::ensure!(self.gs.is_normalized(r), "r out of bounds");
        let ij_old = self.i2g[i];
        let ij_new = self.gs.gindex(r);
        if ij_old == ij_new {
            return Ok(());
        }
        self.i2g[i] = ij_new;
        let src = &mut self.g2i[ij_old];
        let ii = src
            .iter()
            .position(|&id| id == i)
            .expect("maintains the id");
        src.swap_remove(ii);
        self.update_sur(ij_old);
        self.g2i[ij_new].push(i);
        self.update_sur(ij_new);
        debug_assert!(self.sanitize(i, r).is_ok());
        Ok(())
    }

    /// Clears all particles.
    pub fn clear(&mut self) {
        self.i2g.clear();
        self.g2i.iter_mut().for_each(Vec::clear);
        self.sur.clear();
    }
}

impl AddP for GState {
    #[inline]
    fn add(&mut self, r: &Pos) -> anyhow::Result<()> {
        anyhow::ensure!(self.gs.is_normalized(r), "r out of bounds");
        let ij = self.gs.gindex(r);
        let id = self.i2g.len();
        self.g2i[ij].push(id);
        self.i2g.push(ij);
        self.update_sur(ij);
        debug_assert!(self.sanitize(id, r).is_ok());
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Sampler<U> {
    /// Potential.
    u: U,
    cv_x: CellVetoSampler,
    cv_y: CellVetoSampler,
    inner: GState,
}

impl<U: InvPotential + PotentialDiff + Clone> Sampler<U> {
    /// Creates an empty [`Sampler`].
    ///
    /// # Errors
    /// - If failed to create tables from `u`.
    /// - If `th` is non-positive.
    #[inline]
    pub fn new(u: U, gs: Grid, th: f64) -> anyhow::Result<Self> {
        // TODO: Add manual neighbor selection
        anyhow::ensure!(th > 0.0, "threshold must be positive");
        let (tab_x, tab_y) = u.tab(gs);
        let tab_x = LJ::cutoff(&tab_x, th);
        let tab_y = LJ::cutoff(&tab_y, th);
        Ok(Self {
            u,
            cv_x: CellVetoSampler::new(&tab_x)?,
            cv_y: CellVetoSampler::new(&tab_y)?,
            inner: GState::new(gs),
        })
    }

    /// Computes the distance to go.
    #[expect(clippy::match_bool)]
    fn sample_direct_impl<const K: BranchTag>(
        &self,
        rng: &mut impl Rng,
        mov: &Pos,
        peer: &Pos,
        hint: f64,
    ) -> Option<f64> {
        let peer = self.inner.gs.mirror(mov, peer);
        let dx = mov.x - peer.x;
        let dy = mov.y - peer.y;
        let (lx, ly) = self.inner.gs.simbox();
        let pos = match K {
            BRANCH_X => {
                let ch = Chord::new(dy.abs(), 0.5 * lx).unwrap();
                OnChord::new(ch, dx).expect("within bounds")
            }
            BRANCH_Y => {
                let ch = Chord::new(dx.abs(), 0.5 * ly).unwrap();
                OnChord::new(ch, dy).expect("within bounds")
            }
        };
        self.u.sample_impl(Exp1.sample(rng), pos, hint)
    }

    /// Takes care of all the events that are not thinned.
    ///
    /// # Notes
    /// `after` can be zero due to numerical errors.
    #[expect(clippy::match_bool)]
    fn sample_direct<const K: BranchTag>(
        &self,
        rng: &mut impl Rng,
        ctx @ (rs, ss): SamplingContext<'_>,
        hint: f64,
    ) -> Option<Event> {
        let mov = rs[ss.id];
        let mut opt: Option<Event> = None;
        // Surplus ids
        let it_sur = self
            .inner
            .sur
            .iter()
            .flat_map(|&ij| self.inner.g2i[ij].iter().skip(1))
            .inspect(|&id| log::debug!("surplus: {} <-> {id}", ss.id));
        // Neighbor ids
        let it_nb = match K {
            BRANCH_X => &self.cv_x,
            BRANCH_Y => &self.cv_y,
        }
        .direct(self.inner.gs.gindex(&mov))
        .flat_map(|ij| self.inner.g2i[ij].iter().take(1))
        .inspect(|&id| log::debug!("neighbor: {} <-> {id}", ss.id));
        for &id_ in it_sur.chain(it_nb) {
            if ss.id == id_ {
                continue;
            }
            let peer = rs[id_];
            let Some(time) = self.sample_direct_impl::<K>(rng, &mov, &peer, hint) else {
                continue;
            };
            log::debug!("direct: {time}");
            if let Some(ref ev) = opt
                && time >= ev.after
            {
                continue;
            }
            let peer = self.inner.gs.mirror(&mov.move_by(ss.dir, time), &peer);
            opt = Some(Event::with_peer(ctx, EventKind::Stop, time, (id_, peer)).unwrap());
        }
        opt
    }

    /// Computes the maximum distance to go.
    ///
    /// # Notes
    /// Pointer can go beyond the cell boundary due to `MARGIN_B`.
    /// This is to avoid subtle issues when particles cross cell boundaries.
    #[expect(clippy::match_bool)]
    fn cv_ub<const K: BranchTag>(&self, r: &Pos) -> f64 {
        let gs = &self.inner.gs;
        let rb = gs.base(gs.gindex(r));
        match K {
            BRANCH_X => rb.x + MARGIN_B * gs.unit.dx - r.x,
            BRANCH_Y => rb.y + MARGIN_B * gs.unit.dy - r.y,
        }
    }

    /// Samples an event using the cell-veto algorithm.
    #[expect(clippy::match_bool)]
    fn sample_cv<const K: BranchTag>(
        &self,
        rng: &mut impl Rng,
        ctx @ (rs, ss): SamplingContext<'_>,
        hint: f64,
    ) -> Option<Event> {
        let init = rs[ss.id];
        let ub = self.cv_ub::<K>(&init);
        let base = self.inner.gs.gindex(&init);
        let mut time = 0.0;
        loop {
            let smp = match K {
                BRANCH_X => self.cv_x.thinned(rng, base, hint),
                BRANCH_Y => self.cv_y.thinned(rng, base, hint),
            }?;
            time += smp.time;
            if time > hint {
                log::debug!("cell-veto skipped");
                break None;
            }
            if time > ub {
                log::debug!("grid stop");
                break Some(Event::without_peer(ctx, EventKind::Grid, ub).unwrap());
            }
            let mov = init.move_by(ss.dir, time);
            let Some(&id_) = self.inner.g2i[smp.gid].first() else {
                log::debug!("cell-veto (empty): {time}");
                continue;
            };
            debug_assert_ne!(ss.id, id_);
            let peer = self.inner.gs.mirror(&mov, &rs[id_]);
            let d = mov - peer;
            let r = d.norm();
            let co = match K {
                BRANCH_X => d.x / r,
                BRANCH_Y => d.y / r,
            };
            log::debug!("weight: {}", smp.weight);
            let pp = (self.u.du(r) * co).max(0.0) / smp.weight;
            log::debug!("post prob.: {pp}");
            assert!((0.0..=1.0).contains(&pp));
            if rng.random::<f64>() < pp {
                log::debug!("cell-veto (accepted): {time}");
                break Some(Event::with_peer(ctx, EventKind::Stop, time, (id_, peer)).unwrap());
            }
            log::debug!("cell-veto (rejected): {time}");
        }
    }

    /// Number of surplus particles.
    #[inline]
    pub fn n_sur(&self) -> usize {
        self.inner
            .sur
            .iter()
            .map(|&ij| self.inner.g2i[ij].len())
            .sum()
    }

    /// Clears all particles.
    #[inline]
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

impl<U> AddP for Sampler<U> {
    #[inline]
    fn add(&mut self, r: &Pos) -> anyhow::Result<()> {
        self.inner.add(r)
    }
}

impl<U: InvPotential + PotentialDiff + Clone> EmitEvent for Sampler<U> {
    type EventLike = Option<Event>;

    #[inline]
    fn emit(
        &self,
        rng: &mut impl Rng,
        ctx @ (_, ss): SamplingContext<'_>,
        hint: Option<f64>,
    ) -> Self::EventLike {
        let mut hint = hint.expect("always needed for better performance");
        let ev_d = match ss.dir {
            MoveDirection::X => self.sample_direct::<BRANCH_X>(rng, ctx, hint),
            MoveDirection::Y => self.sample_direct::<BRANCH_Y>(rng, ctx, hint),
        };
        if let Some(ref ev_d) = ev_d {
            hint = hint.min(ev_d.after);
        }
        let ev_cv = match ss.dir {
            MoveDirection::X => self.sample_cv::<BRANCH_X>(rng, ctx, hint),
            MoveDirection::Y => self.sample_cv::<BRANCH_Y>(rng, ctx, hint),
        };
        match (ev_cv, ev_d) {
            (Some(ev_cv), Some(ev_d)) => Some(Event::earlier(ev_cv, ev_d)),
            (ev_cv, ev_d) => ev_cv.or(ev_d),
        }
    }
}

impl<U> ApplyEvent for Sampler<U> {
    #[inline]
    fn apply(&mut self, ev: &Event) {
        self.inner.update(ev.id, &ev.patch()).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use nalgebra as na;

    use super::*;
    use crate::geom::{GridSize, GridUnit};

    #[test]
    fn test_add() -> anyhow::Result<()> {
        let gs = Grid::new(GridUnit::new(1.0, 1.0)?, GridSize::new(4, 4)?);
        let mut gstate = GState::new(gs);
        assert!(gstate.i2g.is_empty());
        assert!(gstate.g2i.iter().all(Vec::is_empty));
        assert!(gstate.sur.is_empty());
        gstate.add(&na::vector![0.5, 0.5])?;
        assert_eq!(gstate.i2g[0], (0, 0));
        assert_eq!(gstate.g2i[(0, 0)], [0]);
        assert!(gstate.sur.is_empty());
        Ok(())
    }

    #[test]
    fn test_update() -> anyhow::Result<()> {
        let gs = Grid::new(GridUnit::new(1.0, 1.0)?, GridSize::new(4, 4)?);
        let mut gstate = GState::new(gs);
        gstate.add(&na::vector![0.5, 0.5])?;
        gstate.add(&na::vector![0.0, 0.0])?;
        assert!(gstate.sur.contains(&(0, 0)));
        gstate.update(0, &na::vector![1.5, 1.5])?;
        assert_eq!(gstate.i2g[0], (1, 1));
        assert_eq!(gstate.g2i[(0, 0)], [1]);
        assert_eq!(gstate.g2i[(1, 1)], [0]);
        assert!(gstate.sur.is_empty());
        Ok(())
    }

    #[test]
    fn test_update_same() -> anyhow::Result<()> {
        let gs = Grid::new(GridUnit::new(1.0, 1.0)?, GridSize::new(4, 4)?);
        let mut gstate = GState::new(gs);
        gstate.add(&na::vector![0.0, 0.0])?;
        assert_eq!(gstate.i2g[0], (0, 0));
        assert_eq!(gstate.g2i[(0, 0)], [0]);
        assert!(gstate.sur.is_empty());
        gstate.update(0, &na::vector![0.5, 0.5])?;
        assert_eq!(gstate.i2g[0], (0, 0));
        assert_eq!(gstate.g2i[(0, 0)], [0]);
        assert!(gstate.sur.is_empty());
        Ok(())
    }
}
