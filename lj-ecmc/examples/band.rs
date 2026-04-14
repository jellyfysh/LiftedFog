use std::{fs::File, sync::LazyLock};

use lj_ecmc::{
    geom::{Grid, GridSize, GridUnit},
    lj::LJ,
    sim::{Config, Sim},
    state::Record,
    traits::AddP,
    types::{MoveDirection, Pos},
};
use nalgebra as na;
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};

const EPS: f64 = 1.0 / 0.46;
const SIG: f64 = 1.0;
const DX: f64 = 0.5 * SIG;

static RECORD: LazyLock<Record> = LazyLock::new(|| {
    let ifile = File::open("liquid.json").unwrap();
    serde_json::from_reader(&ifile).unwrap()
});

fn band_init(rng: &mut impl Rng, ratio: f64) -> Vec<Pos> {
    let (l, _) = RECORD.simbox;
    let dx = rng.random_range(0.0..l);
    let dy = rng.random_range(0.0..l);
    let sx = rng.random_bool(0.5);
    let sy = rng.random_bool(0.5);
    RECORD
        .data
        .iter()
        .filter_map(|r| {
            let mut r = na::vector![(r.x + dx) % l, (r.y + dy) % l];
            if sx {
                r.x = l - r.x;
            }
            if sy {
                r.y = l - r.y;
            }
            let s = r.x + r.y;
            if s < ratio * l || (l < s && s < (1.0 + ratio) * l) {
                Some(r)
            } else {
                None
            }
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    let mut rng = StdRng::seed_from_u64(42);
    let gs = Grid::new(GridUnit::new(DX, DX)?, GridSize::new(1000, 1000)?);
    let (l, _) = gs.simbox();
    let lj = LJ::new(EPS, SIG)?;
    let time = 5E-4 * l;
    let cfg = Config::new(lj, gs, time)?;
    let mut sim = Sim::with_th(cfg, 5.0)?;
    for _ in 0..25 {
        let rs = band_init(&mut rng, 0.9);
        sim.add_from_slice(&rs)?;
        eprintln!("#SUR: {}", sim.smp.n_sur());
        let mut id = rng.random_range(0..rs.len());
        let mut cum = 0.0;
        while f64::abs(cum) <= 2.0 * l {
            let init = sim.state[id];
            let stat = sim.step(&mut rng, id, MoveDirection::X);
            id = stat.last;
            if stat.drift != 0.0 {
                println!("{} {} {}", init.x, init.y, stat.drift);
            }
            cum += stat.drift;
            cum += stat.move_extra + time;
        }
        sim.clear();
    }
    Ok(())
}
