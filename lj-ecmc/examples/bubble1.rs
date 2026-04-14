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

fn circ_init(rng: &mut impl Rng, rel_rad: f64) -> Vec<Pos> {
    let (l, _) = RECORD.simbox;
    let rad = rel_rad * l;
    let dx = rng.random_range(0.0..l);
    let dy = rng.random_range(0.0..l);
    let rc = na::vector![0.5 * l, 0.5 * l];
    let theta = rng.random_range(0.0..consts::TAU);
    let rot = Rotation2::new(theta);
    RECORD
        .data
        .iter()
        .filter_map(|r| {
            let d = na::vector![(r.x + dx) % l, (r.y + dy) % l] - rc;
            if d.norm_squared() < rad.powi(2) {
                Some(rc + rot * d)
            } else {
                None
            }
        })
        .collect()
}

const EPS: f64 = 1.0 / 0.46;
const SIG: f64 = 1.0;
const DX: f64 = 0.5 * SIG;
const R: f64 = 0.4;

fn main() -> anyhow::Result<()> {
    let gs = Grid::new(GridUnit::new(DX, DX)?, GridSize::new(1000, 1000)?);
    let (l, _) = gs.simbox();
    let lj = LJ::new(EPS, SIG)?;
    let cfg = Config::new(lj, gs, 5E-3 * l)?;
    let sim = Sim::with_th(cfg, 5.0)?;
    let lb = (0.5 - R) * l;
    assert!((0.0..l).contains(&lb));
    let ub = (0.5 + R) * l;
    assert!((0.0..l).contains(&ub));
    let f = || {
        let mut sim = sim.clone();
        let mut rng = rand::rng();
        for _ in 0..62500 {
            let mut rs = circ_init(&mut rng, R);
            let y0 = rng.random_range(lb..ub);
            rs.push(na::vector![0.0, y0]);
            sim.add_from_slice(&rs).unwrap();
            let mut id = rs.len() - 1;
            while sim.state[id].x <= 0.99 * l {
                let stat = sim.step(&mut rng, id, MoveDirection::X);
                id = stat.last;
            }
            println!("{y0} {}", sim.state[id].y);
            sim.clear();
        }
    };
    thread::scope(|s| {
        for _ in 0..16 {
            s.spawn(f);
        }
    });
    Ok(())
}
