//! Simulator.

use core::mem;

use rand::Rng;

use crate::{
    geom::Grid,
    lj::LJ,
    sampler::Sampler,
    state::State,
    traits::{AddP, ApplyEvent, EmitEvent},
    types::{Event, EventKind, I1, MoveDirection, Pos, SamplingState},
};

#[derive(Clone, Copy, Debug, PartialEq)]
#[readonly::make]
pub struct Config {
    /// Lennard-Jones model.
    pub model: LJ,
    /// Cell-veto grid system.
    pub gs: Grid,
    /// Chain length.
    pub time: f64,
}

impl Config {
    /// Creates a new [`Config`] instance.
    ///
    /// # Errors
    /// - If `time` is non-positive.
    #[inline]
    pub fn new(model: LJ, gs: Grid, time: f64) -> anyhow::Result<Self> {
        anyhow::ensure!(time > 0.0, "time must be positive");
        Ok(Self { model, gs, time })
    }
}

/// Simulation statistics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[readonly::make]
pub struct Stat {
    /// ID of the last moved particle.
    pub last: I1,
    /// Number of boundary events.
    pub n_bound: usize,
    /// Number of stop events.
    pub n_stop: usize,
    /// Cumulative distance earned by pair-wise events.
    pub move_extra: f64,
    /// Pointer displacement along the vertical direction.
    pub drift: f64,
}

impl ApplyEvent for Stat {
    #[inline]
    fn apply(&mut self, ev: &Event) {
        self.last = ev.id;
        match ev.kind {
            EventKind::Timeout | EventKind::Grid => {}
            EventKind::Boundary => {
                self.n_bound += 1;
            }
            EventKind::Stop => {
                self.n_stop += 1;
                self.move_extra += ev.extra();
                self.drift += ev.drift();
            }
        }
    }
}

/// Wrapper struct for simulation.
#[derive(Clone, Debug)]
#[readonly::make]
pub struct Sim {
    /// Simulation configuration.
    pub cfg: Config,
    /// Particle state.
    pub state: State,
    /// ECMC sampler.
    pub smp: Sampler<LJ>,
}

impl Sim {
    /// Creates a new [`Sim`] instance with threshold `th`.
    ///
    /// # Errors
    /// See [`State::new`] and [`Sampler::new`] for possible errors.
    #[inline]
    pub fn with_th(cfg: Config, th: f64) -> anyhow::Result<Self> {
        let state = State::new(cfg.gs);
        let smp = Sampler::new(cfg.model, cfg.gs, th)?;
        Ok(Self { cfg, state, smp })
    }

    /// Creates a new [`Sim`] instance.
    ///
    /// # Errors
    /// See [`Self::with_th`].
    #[inline]
    pub fn new(cfg: Config) -> anyhow::Result<Self> {
        Self::with_th(cfg, f64::INFINITY)
    }

    /// Clears all particles.
    #[inline]
    pub fn clear(&mut self) {
        self.smp.clear();
        let _ = mem::replace(&mut self.state, State::new(self.cfg.gs));
    }

    /// Performs a single ECMC step.
    #[expect(clippy::missing_panics_doc)]
    #[inline]
    pub fn step(&mut self, rng: &mut impl Rng, id: I1, dir: MoveDirection) -> Stat {
        // TODO: Return simulation statistics
        let mut stat = Stat::default();
        let mut ss = SamplingState::new(dir, id, self.cfg.time).expect("validated in advance");
        while ss.time > 0.0 {
            log::debug!("move {}: {:?}", ss.id, self.state[ss.id]);
            let ctx = (&self.state[..], &ss);
            let mut ev = Event::earlier(self.state.emit(rng, ctx, None), ss.emit(rng, ctx, None));
            let hint = Some(ev.after);
            let ret_s = self.smp.emit(rng, ctx, hint);
            if let Some(ev_s) = ret_s {
                ev = Event::earlier(ev, ev_s);
            }
            log::debug!("event: {ev:?}");
            self.state.apply(&ev);
            self.smp.apply(&ev);
            ss.apply(&ev);
            stat.apply(&ev);
        }
        stat
    }
}

impl AddP for Sim {
    #[inline]
    fn add(&mut self, r: &Pos) -> anyhow::Result<()> {
        self.state.add(r)?;
        self.smp.add(r)?;
        Ok(())
    }
}
