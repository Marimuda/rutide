//! Initial exact-correction constituent catalog.
//!
//! The astronomical and satellite constants are the subset required by the
//! benchmark profile, transcribed from `utide/data/ut_constants.npz` at the
//! pinned Python `UTide` oracle revision. None of these five constituents is
//! shallow-water derived.

/// A tidal constituent whose astronomical and satellite metadata is built in.
///
/// This initial catalog intentionally contains the five constituents frozen in
/// the FVCOM benchmark. More constituents can be added without changing the
/// corrected solver API.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum TidalConstituent {
    /// Principal lunar semidiurnal constituent.
    M2,
    /// Principal solar semidiurnal constituent.
    S2,
    /// Larger lunar elliptic semidiurnal constituent.
    N2,
    /// Lunar-solar declinational diurnal constituent.
    K1,
    /// Principal lunar declinational diurnal constituent.
    O1,
}

impl TidalConstituent {
    /// Return the conventional constituent name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::M2 => "M2",
            Self::S2 => "S2",
            Self::N2 => "N2",
            Self::K1 => "K1",
            Self::O1 => "O1",
        }
    }

    pub(crate) const fn metadata(self) -> Metadata {
        match self {
            Self::M2 => Metadata {
                doodson: [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                semi: 0.0,
                satellites: &M2_SATELLITES,
            },
            Self::S2 => Metadata {
                doodson: [2.0, 2.0, -2.0, 0.0, 0.0, 0.0],
                semi: 0.0,
                satellites: &S2_SATELLITES,
            },
            Self::N2 => Metadata {
                doodson: [2.0, -1.0, 0.0, 1.0, 0.0, 0.0],
                semi: 0.0,
                satellites: &N2_SATELLITES,
            },
            Self::K1 => Metadata {
                doodson: [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                semi: -0.75,
                satellites: &K1_SATELLITES,
            },
            Self::O1 => Metadata {
                doodson: [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                semi: -0.25,
                satellites: &O1_SATELLITES,
            },
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Metadata {
    pub(crate) doodson: [f64; 6],
    pub(crate) semi: f64,
    pub(crate) satellites: &'static [Satellite],
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Satellite {
    pub(crate) amplitude_ratio: f64,
    pub(crate) latitude_factor: u8,
    pub(crate) delta_doodson: [i8; 3],
    pub(crate) phase_correction: f64,
}

const fn satellite(
    amplitude_ratio: f64,
    latitude_factor: u8,
    delta_doodson: [i8; 3],
    phase_correction: f64,
) -> Satellite {
    Satellite {
        amplitude_ratio,
        latitude_factor,
        delta_doodson,
        phase_correction,
    }
}

const M2_SATELLITES: [Satellite; 9] = [
    satellite(0.0001, 2, [-1, -1, 0], 0.75),
    satellite(0.0004, 2, [-1, 0, 0], 0.75),
    satellite(0.0005, 0, [0, -2, 0], 0.0),
    satellite(0.0373, 0, [0, -1, 0], 0.5),
    satellite(0.0001, 2, [1, -1, 0], 0.25),
    satellite(0.0009, 2, [1, 0, 0], 0.75),
    satellite(0.0002, 2, [1, 1, 0], 0.75),
    satellite(0.0006, 0, [2, 0, 0], 0.0),
    satellite(0.0002, 0, [2, 1, 0], 0.0),
];

const S2_SATELLITES: [Satellite; 3] = [
    satellite(0.0022, 0, [0, -1, 0], 0.0),
    satellite(0.0001, 2, [1, 0, 0], 0.75),
    satellite(0.0001, 0, [2, 0, 0], 0.0),
];

const N2_SATELLITES: [Satellite; 4] = [
    satellite(0.0039, 0, [-2, -2, 0], 0.5),
    satellite(0.0008, 0, [-1, 0, 1], 0.0),
    satellite(0.0005, 0, [0, -2, 0], 0.0),
    satellite(0.0373, 0, [0, -1, 0], 0.5),
];

const K1_SATELLITES: [Satellite; 10] = [
    satellite(0.0002, 0, [-2, -1, 0], 0.0),
    satellite(0.0001, 1, [-1, -1, 0], 0.75),
    satellite(0.0007, 1, [-1, 0, 0], 0.25),
    satellite(0.0001, 1, [-1, 1, 0], 0.75),
    satellite(0.0001, 0, [0, -2, 0], 0.0),
    satellite(0.0198, 0, [0, -1, 0], 0.5),
    satellite(0.1356, 0, [0, 1, 0], 0.0),
    satellite(0.0029, 0, [0, 2, 0], 0.5),
    satellite(0.0002, 1, [1, 0, 0], 0.25),
    satellite(0.0001, 1, [1, 1, 0], 0.25),
];

const O1_SATELLITES: [Satellite; 8] = [
    satellite(0.0003, 1, [-1, 0, 0], 0.25),
    satellite(0.0058, 0, [0, -2, 0], 0.5),
    satellite(0.1885, 0, [0, -1, 0], 0.0),
    satellite(0.0004, 1, [1, -1, 0], 0.25),
    satellite(0.0029, 1, [1, 0, 0], 0.75),
    satellite(0.0004, 1, [1, 1, 0], 0.25),
    satellite(0.0064, 0, [2, 0, 0], 0.5),
    satellite(0.0010, 0, [2, 1, 0], 0.5),
];
