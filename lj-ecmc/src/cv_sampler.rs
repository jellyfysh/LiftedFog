//! Cell-veto thinning sampling.

use argmin::{
    core::{CostFunction, Executor, IterState, Solver},
    solver::brent::BrentOpt,
};
use nalgebra as na;
use nalgebra::{DMatrix, Dim, Matrix, RawStorage};
use rand::Rng;
use rand_distr::{Distribution, Exp, weighted::WeightedAliasIndex};

use crate::{
    geom::{Grid, GridSize},
    traits::PotentialDiff,
    types::{I2, Pos},
};

/// Two-dimensional Walker sampler.
#[derive(Clone, Debug)]
#[readonly::make]
pub struct GridSampler {
    /// Two-dimensional weights.
    pub weights: DMatrix<f64>,
    inner: WeightedAliasIndex<f64>,
}

impl GridSampler {
    /// Creates a new [`GridSampler`].
    ///
    /// # Errors
    /// If any weight is negative.
    #[inline]
    pub fn new<R: Dim, C: Dim, S: RawStorage<f64, R, C>>(
        weights: &Matrix<f64, R, C, S>,
    ) -> anyhow::Result<Self> {
        let r = weights.nrows();
        let c = weights.ncols();
        let weights = DMatrix::from_iterator(r, c, weights.iter().copied());
        anyhow::ensure!(
            weights.iter().all(|&w| w >= 0.0),
            "weights must be non-negative"
        );
        // Column-major order
        let v = weights.iter().copied().collect();
        let inner = WeightedAliasIndex::new(v)?;
        Ok(Self { weights, inner })
    }
}

impl Distribution<I2> for GridSampler {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> I2 {
        let ii = self.inner.sample(rng);
        let r = self.weights.nrows();
        (ii % r, ii / r)
    }
}

/// Computes four vertices of the cell having `base` as the bottom-left corner.
fn edge_nodes(gs: Grid, base: Pos) -> [Pos; 4] {
    let dx = gs.unit.dx;
    let dy = gs.unit.dy;
    [
        base,
        base + na::vector![dx, 0.0],
        base + na::vector![0.0, dy],
        base + na::vector![dx, dy],
    ]
}

/// Checks if the given cell is a PBC neighbor of `(0, 0)`.
fn is_00_neighbor(gs: Grid, ij: I2) -> bool {
    gs.size.neighbors((0, 0)).contains(&ij)
}

/// Weight margin factor to avoid numerical issues.
const MARGIN_W: f64 = 1.01;
const _: () = const { assert!(MARGIN_W > 1.0) };

/// Wraps solver execution.
fn solve_wrap<O>(problem: O, rmin: f64, rmax: f64) -> anyhow::Result<f64>
where
    BrentOpt<f64>: Solver<O, IterState<f64, (), (), (), (), f64>>,
    O: Clone,
{
    let sol = BrentOpt::new(rmin, rmax).set_tolerance(f64::EPSILON.sqrt(), 1E-3);
    let ret = Executor::new(problem.clone(), sol)
        .configure(|state| state.param(rmin.midpoint(rmax)).max_iters(100))
        .run()?
        .state()
        .best_cost
        .abs();
    Ok(ret)
}

/// Increase cell boundary sizes by this factor to create grid overlaps.
pub const MARGIN_B: f64 = 1.001;
const _: () = const { assert!(1.0 < MARGIN_B && MARGIN_B < 2.0) };

#[derive(Clone, Copy, Debug, PartialEq)]
#[readonly::make]
struct ReduceImpl {
    /// r extrema.
    pub r: [f64; 2],
    /// cos extrema.
    pub co_x: [f64; 2],
    /// sin extrema.
    pub co_y: [f64; 2],
}

impl ReduceImpl {
    /// Creates a new [`ReduceImpl`] instance.
    #[inline]
    const fn new() -> Self {
        Self {
            r: [f64::INFINITY, f64::NEG_INFINITY],
            co_x: [f64::INFINITY, f64::NEG_INFINITY],
            co_y: [f64::INFINITY, f64::NEG_INFINITY],
        }
    }

    /// Registers a new displacement vector.
    #[inline]
    fn register(&mut self, d: Pos) {
        let r = d.norm();
        self.r[0] = self.r[0].min(r);
        self.r[1] = self.r[1].max(r);
        let co_x = d.x / r;
        self.co_x[0] = self.co_x[0].min(co_x);
        self.co_x[1] = self.co_x[1].max(co_x);
        let co_y = d.y / r;
        self.co_y[0] = self.co_y[0].min(co_y);
        self.co_y[1] = self.co_y[1].max(co_y);
    }
}

/// Computes bounding weights.
fn tab_impl_ij<U: PotentialDiff + Clone>(du: &U, gs: Grid, ij: I2) -> anyhow::Result<(f64, f64)> {
    use argmin::core::Error as MError;

    #[derive(Clone)]
    struct CostWrapper<'a, V: Clone, const C: i8>(&'a V);

    impl<V: PotentialDiff + Clone, const C: i8> CostFunction for CostWrapper<'_, V, C> {
        type Param = f64;
        type Output = f64;

        fn cost(&self, &r: &Self::Param) -> Result<Self::Output, MError> {
            Ok(f64::from(C) * self.0.du(r))
        }
    }

    log::debug!("bounding cell {ij:?}");
    debug_assert!(!is_00_neighbor(gs, ij));
    let base = gs.base(ij);
    let (lx, ly) = gs.simbox();
    let mut ans_x = 0.0;
    let mut ans_y = 0.0;
    let it0 = edge_nodes(gs, Pos::zeros());
    for it1 in [
        edge_nodes(gs, base),
        edge_nodes(gs, base - na::vector![lx, 0.0]),
        edge_nodes(gs, base - na::vector![lx, ly]),
        edge_nodes(gs, base - na::vector![0.0, ly]),
    ] {
        let mut red = ReduceImpl::new();
        for (r0, r1) in itertools::iproduct!(it0, it1) {
            // Allow r0 to be slightly outside the box
            // Safe because neighbors are excluded
            let r0 = MARGIN_B * r0;
            // PBC already taken into account
            red.register(r1 - r0);
        }
        let [rmin, rmax] = red.r;
        debug_assert!(rmin > 0.0);
        let mut loc = f64::max(du.du(rmin).abs(), du.du(rmax).abs());
        loc = loc.max(solve_wrap(CostWrapper::<_, 1>(du), rmin, rmax)?);
        loc = loc.max(solve_wrap(CostWrapper::<_, -1>(du), rmin, rmax)?);
        let [co_min, co_max] = red.co_x;
        let co_x = f64::max(co_min.abs(), co_max.abs());
        let [co_min, co_max] = red.co_y;
        let co_y = f64::max(co_min.abs(), co_max.abs());
        log::debug!("(du, cos, sin): ({loc}, {co_x}, {co_y})");
        ans_x = (loc * co_x).max(ans_x);
        ans_y = (loc * co_y).max(ans_y);
    }
    Ok((MARGIN_W * ans_x, MARGIN_W * ans_y))
}

/// Computes cell-veto weight tables.
///
/// Will skip corresponding cells if their weights are [`f64::INFINITY`].
pub trait CellVetoTable {
    /// Tables for X/Y direction.
    fn tab(&self, gs: Grid) -> (DMatrix<f64>, DMatrix<f64>);

    /// Replaces weights above the threshold with [`f64::INFINITY`].
    #[inline]
    fn cutoff<R: Dim, C: Dim, S: RawStorage<f64, R, C>>(
        tab: &Matrix<f64, R, C, S>,
        th: f64,
    ) -> DMatrix<f64> {
        assert!(th >= 0.0, "th must be non-negative");
        let it = tab.iter().map(|&w| if w > th { f64::INFINITY } else { w });
        DMatrix::from_iterator(tab.nrows(), tab.ncols(), it)
    }
}

impl<U: PotentialDiff + Clone> CellVetoTable for U {
    #[inline]
    fn tab(&self, gs: Grid) -> (DMatrix<f64>, DMatrix<f64>) {
        let nx = gs.size.nx;
        let ny = gs.size.ny;
        let mut ret_x = DMatrix::from_element(nx, ny, f64::INFINITY);
        let mut ret_y = DMatrix::from_element(nx, ny, f64::INFINITY);
        for ij in itertools::iproduct!(0..nx, 0..ny) {
            if is_00_neighbor(gs, ij) {
                continue;
            }
            match tab_impl_ij(self, gs, ij) {
                Ok((tx, ty)) => {
                    ret_x[ij] = tx;
                    ret_y[ij] = ty;
                }
                Err(e) => panic!("failed to find extrema: {e}"),
            }
        }
        (ret_x, ret_y)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThinnedSample {
    /// Duration until candidate event.
    pub time: f64,
    /// Corresponding weight.
    pub weight: f64,
    /// Grid id to check.
    pub gid: I2,
}

#[derive(Clone, Debug)]
pub struct CellVetoSampler {
    /// Cells that are not included in the thinned sampling.
    nb: Vec<I2>,
    /// Exponential distribution for sampling the first event time.
    smp_t: Exp<f64>,
    /// Grid sampler for event distribution.
    smp_g: GridSampler,
}

impl CellVetoSampler {
    /// Creates a new [`CellVetoSampler`].
    ///
    /// # Arguments
    /// - `weights`: Weight table. [`f64::INFINITY`] can be used to exclude cells.
    ///
    /// # Errors
    /// - If total weight sum is zero.
    /// - If any weight is negative.
    /// - If any neighbor cell of `(0, 0)` has finite weight.
    #[inline]
    pub fn new<R: Dim, C: Dim, S: RawStorage<f64, R, C>>(
        weights: &Matrix<f64, R, C, S>,
    ) -> anyhow::Result<Self> {
        let r = weights.nrows();
        let c = weights.ncols();
        let mut tab = DMatrix::from_iterator(r, c, weights.iter().copied());
        let mut nb = Vec::new();
        for ij in GridSize::new(r, c)?.neighbors((0, 0)) {
            anyhow::ensure!(tab[ij] == f64::INFINITY, "neighbor cells cannot be thinned");
        }
        for ij in itertools::iproduct!(0..r, 0..c) {
            let w = tab[ij];
            anyhow::ensure!(w >= 0.0, "weights must be non-negative");
            if w == f64::INFINITY {
                nb.push(ij);
                tab[ij] = 0.0;
            }
        }
        let tot = tab.iter().sum();
        anyhow::ensure!(tot > 0.0, "no events to sample");
        Ok(Self {
            nb,
            smp_t: Exp::new(tot)?,
            smp_g: GridSampler::new(&tab)?,
        })
    }

    /// Applies `diff` to `base` with PBC.
    fn shift(&self, base: I2, diff: I2) -> I2 {
        let nx = self.smp_g.weights.nrows();
        let ny = self.smp_g.weights.ncols();
        ((base.0 + diff.0) % nx, (base.1 + diff.1) % ny)
    }

    /// Samples the cell-veto candidate event.
    #[inline]
    pub fn thinned(&self, rng: &mut impl Rng, base: I2, hint: f64) -> Option<ThinnedSample> {
        let time = self.smp_t.sample(rng);
        log::debug!("pre: {time}");
        if time > hint {
            return None;
        }
        let diff = self.smp_g.sample(rng);
        debug_assert!(!self.nb.contains(&diff));
        log::debug!("weight distributed to +{diff:?}");
        Some(ThinnedSample {
            time,
            gid: self.shift(base, diff),
            weight: self.smp_g.weights[diff],
        })
    }

    /// Iterates over cells that are not thinned.
    #[inline]
    pub fn direct(&self, base: I2) -> impl Iterator<Item = I2> {
        self.nb.iter().map(move |&ij| self.shift(base, ij))
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;
    use nalgebra::{self as na, Matrix2};
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    use super::*;
    use crate::{
        geom::{GridSize, GridUnit},
        lj::LJ,
    };

    #[test]
    fn test_grid_sampler() {
        const N: u32 = 100_000;
        let pref = na::matrix![0.0, 1.0 / 6.0; 2.0 / 6.0, 3.0 / 6.0];
        let smp = GridSampler::new(&pref).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let mut pcmp = Matrix2::<f64>::zeros();
        for _ in 0..N {
            let (i, j) = smp.sample(&mut rng);
            pcmp[(i, j)] += 1.0 / f64::from(N);
        }
        assert_approx_eq!(f64, pcmp[(0, 0)], pref[(0, 0)], epsilon = 0.002);
        assert_approx_eq!(f64, pcmp[(0, 1)], pref[(0, 1)], epsilon = 0.002);
        assert_approx_eq!(f64, pcmp[(1, 0)], pref[(1, 0)], epsilon = 0.002);
        assert_approx_eq!(f64, pcmp[(1, 1)], pref[(1, 1)], epsilon = 0.002);
    }

    #[test]
    fn test_tab_x() {
        let lj = LJ::new(1.0, 1.0).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let gs = Grid::new(
            GridUnit::new(0.5, 0.5).unwrap(),
            GridSize::new(50, 50).unwrap(),
        );
        let (lx, ly) = gs.simbox();
        let (tab, _) = lj.tab(gs);
        for _ in 0..100_000 {
            let r0 = na::vector![
                rng.random_range(0.0..gs.unit.dx),
                rng.random_range(0.0..gs.unit.dy)
            ];
            let r1 = na::vector![rng.random_range(0.0..lx), rng.random_range(0.0..ly)];
            let d = gs.mirror(&r0, &r1) - r0;
            let r = d.norm();
            assert!(lj.du(r) * d.x / r < tab[gs.gindex(&r1)]);
        }
    }

    #[test]
    fn test_tab_y() {
        let lj = LJ::new(1.0, 1.0).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let gs = Grid::new(
            GridUnit::new(0.5, 0.5).unwrap(),
            GridSize::new(50, 50).unwrap(),
        );
        let (lx, ly) = gs.simbox();
        let (_, tab) = lj.tab(gs);
        for _ in 0..100_000 {
            let r0 = na::vector![
                rng.random_range(0.0..gs.unit.dx),
                rng.random_range(0.0..gs.unit.dy)
            ];
            let r1 = na::vector![rng.random_range(0.0..lx), rng.random_range(0.0..ly)];
            let d = gs.mirror(&r0, &r1) - r0;
            let r = d.norm();
            assert!(lj.du(r) * d.y / r < tab[gs.gindex(&r1)]);
        }
    }

    #[test]
    fn test_cutoff() {
        let raw = na::matrix![0.0, 1.0; 2.0, 3.0];
        let tab = LJ::cutoff(&raw, 1.5);
        assert_eq!(tab, na::matrix![0.0, 1.0; f64::INFINITY, f64::INFINITY]);
    }

    #[test]
    fn test_cv() {
        let tab = na::matrix![
            f64::INFINITY,
            f64::INFINITY,
            1.0,
            f64::INFINITY,
            f64::INFINITY
        ];
        let cv = CellVetoSampler::new(&tab).unwrap();
        assert_eq!(cv.nb, [(0, 0), (0, 1), (0, 3), (0, 4)]);
        assert_eq!(cv.smp_t, Exp::new(1.0).unwrap());
        assert_eq!(cv.smp_g.weights, na::matrix![0.0, 0.0, 1.0, 0.0, 0.0]);

        let mut rng = StdRng::seed_from_u64(42);
        let ev_t = cv.thinned(&mut rng, (0, 3), f64::INFINITY).unwrap();
        assert_approx_eq!(f64, ev_t.weight, 1.0);
        assert_eq!(ev_t.gid, (0, 0));
        assert_eq!(
            cv.direct((0, 3)).collect::<Vec<_>>(),
            [(0, 3), (0, 4), (0, 1), (0, 2)]
        );
    }
}
