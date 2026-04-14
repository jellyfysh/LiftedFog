use core::f64::consts;
use std::{fs::File, sync::LazyLock, thread};

use lj_ecmc::{
    geom::{Grid, GridSize, GridUnit},
    lj::LJ,
    sim::{Config, Sim},
    state::Record,
    traits::AddP,
    types::{MoveDirection, Pos},
};
use nalgebra::{self as na, Rotation2};
use rand::{Rng, RngExt};

static RECORD: LazyLock<Record> = LazyLock::new(|| {
    let ifile = File::open("liquid.json").unwrap();
    serde_json::from_reader(&ifile).unwrap()
});

#[derive(Debug, Clone, Copy, PartialEq)]
struct SimGeom {
    /// Simulation box length.
    l: f64,
    /// Bubble radius.
    rad: f64,
    /// Bubble distance relative to `rad`.
    rel_d: f64,
}

impl SimGeom {
    fn rc1(&self) -> Pos {
        na::vector![0.25 * self.l, 0.5 * self.l - 0.5 * self.rel_d * self.rad]
    }

    fn rc2(&self) -> Pos {
        na::vector![0.75 * self.l, 0.5 * self.l + 0.5 * self.rel_d * self.rad]
    }

    fn lb(&self) -> f64 {
        let ret = self.rc1().y - self.rad;
        assert!((0.0..self.l).contains(&ret));
        ret
    }

    fn ub(&self) -> f64 {
        let ret = self.rc2().y + self.rad;
        assert!((0.0..self.l).contains(&ret));
        ret
    }

    fn init(&self, rng: &mut impl Rng) -> Vec<Pos> {
        let (l, _) = RECORD.simbox;
        let dx = rng.random_range(0.0..l);
        let dy = rng.random_range(0.0..l);
        let rad = self.rad;
        let rc1 = self.rc1();
        let rc2 = self.rc2();
        let rot1 = Rotation2::new(rng.random_range(0.0..consts::TAU));
        let rot2 = Rotation2::new(rng.random_range(0.0..consts::TAU));
        RECORD
            .data
            .iter()
            .filter_map(|r| {
                let d = na::vector![(r.x + dx) % l, (r.y + dy) % l];
                let d1 = rot1 * (d - rc1);
                let d2 = rot2 * (d - rc2);
                match (
                    d1.norm_squared() < rad.powi(2),
                    d2.norm_squared() < rad.powi(2),
                ) {
                    (true, true) => panic!(),
                    (true, false) => Some(rc1 + d1),
                    (false, true) => Some(rc2 + d2),
                    (false, false) => None,
                }
            })
            .collect()
    }

    fn cent_impl(&self, pos: &[Pos], rc: Pos) -> Pos {
        let mut ret = na::vector![0.0, 0.0];
        let mut cnt = 0;
        for &p in pos {
            if (p - rc).norm_squared() < self.rad.powi(2) {
                ret += p;
                cnt += 1;
            }
        }
        ret / f64::from(cnt)
    }

    fn cent_1(&self, pos: &[Pos]) -> Pos {
        self.cent_impl(pos, self.rc1())
    }

    fn cent_2(&self, pos: &[Pos]) -> Pos {
        self.cent_impl(pos, self.rc2())
    }

    fn cent_diff(&self, pos: &[Pos]) -> Pos {
        self.cent_2(pos) - self.cent_1(pos)
    }
}

const EPS: f64 = 1.0 / 0.46;
const SIG: f64 = 1.0;
const DX: f64 = 0.5 * SIG;
const R: f64 = 0.2;
const D: f64 = 0.5;

fn main() -> anyhow::Result<()> {
    let gs = Grid::new(GridUnit::new(DX, DX)?, GridSize::new(1000, 1000)?);
    let (l, _) = gs.simbox();
    let lj = LJ::new(EPS, SIG)?;
    let cfg = Config::new(lj, gs, 5E-3 * l)?;
    let sim = Sim::with_th(cfg, 5.0)?;
    let geom = SimGeom {
        l,
        rad: R * l,
        rel_d: D,
    };
    let lb = geom.lb();
    let ub = geom.ub();
    let f = || {
        let mut sim = sim.clone();
        let mut rng = rand::rng();
        for _ in 0..40000 {
            let mut rs = geom.init(&mut rng);
            let y0 = rng.random_range(lb..ub);
            rs.push(na::vector![0.0, y0]);
            sim.add_from_slice(&rs).unwrap();
            let mut id = rs.len() - 1;
            let mut diff = -geom.cent_diff(&sim.state);
            while sim.state[id].x <= 0.99 * l {
                let stat = sim.step(&mut rng, id, MoveDirection::X);
                id = stat.last;
            }
            diff += geom.cent_diff(&sim.state);
            println!("{y0} {} {}", sim.state[id].y, diff.x);
            sim.clear();
        }
    };
    thread::scope(|s| {
        for _ in 0..14 {
            s.spawn(f);
        }
    });
    Ok(())
}
