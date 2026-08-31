//! Complete constituent metadata pinned to the Python `UTide` oracle.

use std::{error::Error, fmt, str::FromStr};

/// Git revision of the Python `UTide` checkout used to generate the catalog.
pub const CATALOG_ORACLE_REVISION: &str = generated::ORACLE_REVISION;

/// SHA-256 of the pinned oracle's `ut_constants.npz` source.
pub const CATALOG_SOURCE_SHA256: &str = generated::SOURCE_SHA256;

/// Number of named constituents in the pinned catalog.
pub const CONSTITUENT_COUNT: usize = 146;

/// A stable identifier into the pinned tidal constituent catalog.
///
/// Names are parsed with [`FromStr`] or [`Self::from_name`]. The identifier is
/// compact and copyable while supporting names that cannot be Rust enum variants,
/// such as `2Q1` and `3MS8`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TidalConstituent(u8);

impl TidalConstituent {
    /// Principal lunar semidiurnal constituent.
    pub const M2: Self = Self(47);
    /// Principal solar semidiurnal constituent.
    pub const S2: Self = Self(56);
    /// Larger lunar elliptic semidiurnal constituent.
    pub const N2: Self = Self(41);
    /// Lunar-solar declinational diurnal constituent.
    pub const K1: Self = Self(20);
    /// Principal lunar declinational diurnal constituent.
    pub const O1: Self = Self(12);

    /// Look up an exact, case-sensitive conventional constituent name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        generated::CATALOG
            .iter()
            .position(|entry| entry.name == name)
            .and_then(Self::from_index)
    }

    /// Construct an identifier from its stable zero-based catalog index.
    #[must_use]
    pub fn from_index(index: usize) -> Option<Self> {
        match u8::try_from(index) {
            Ok(index) if usize::from(index) < CONSTITUENT_COUNT => Some(Self(index)),
            _ => None,
        }
    }

    /// Iterate over every constituent in stable oracle catalog order.
    pub fn all() -> impl ExactSizeIterator<Item = Self> + DoubleEndedIterator {
        (0_u8..146).map(Self)
    }

    /// Return the stable zero-based oracle catalog index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Return the conventional constituent name.
    #[must_use]
    pub fn name(self) -> &'static str {
        self.entry().name
    }

    /// Return the catalog's conventional Rayleigh separation in cycles/hour.
    #[must_use]
    pub fn rayleigh_separation_cph(self) -> f64 {
        self.entry().rayleigh_cph
    }

    /// Return the catalog's static reference frequency in cycles/hour.
    ///
    /// Exact analysis recomputes the frequency at the record reference epoch.
    #[must_use]
    pub fn catalog_frequency_cph(self) -> f64 {
        self.entry().reference_frequency_cph
    }

    /// Return whether the constituent is derived from shallow-water parents.
    #[must_use]
    pub fn is_shallow(self) -> bool {
        self.entry().shallow_len != 0
    }

    pub(crate) fn metadata(self) -> Metadata {
        let entry = self.entry();
        Metadata {
            doodson: entry.doodson,
            semi: entry.semi,
            satellites: &generated::SATELLITES
                [entry.satellite_start..entry.satellite_start + entry.satellite_len],
            shallow_terms: &generated::SHALLOW_TERMS
                [entry.shallow_start..entry.shallow_start + entry.shallow_len],
        }
    }

    fn entry(self) -> &'static CatalogEntry {
        &generated::CATALOG[self.index()]
    }
}

impl fmt::Display for TidalConstituent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

impl FromStr for TidalConstituent {
    type Err = UnknownTidalConstituent;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Self::from_name(name).ok_or_else(|| UnknownTidalConstituent {
            name: name.to_owned(),
        })
    }
}

/// Error returned when a name is absent from the pinned constituent catalog.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnknownTidalConstituent {
    name: String,
}

impl UnknownTidalConstituent {
    /// Return the unrecognized input name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl fmt::Display for UnknownTidalConstituent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "unknown tidal constituent {:?}", self.name)
    }
}

impl Error for UnknownTidalConstituent {}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Metadata {
    pub(crate) doodson: Option<[i8; 6]>,
    pub(crate) semi: Option<f64>,
    pub(crate) satellites: &'static [Satellite],
    pub(crate) shallow_terms: &'static [ShallowTerm],
}

#[derive(Clone, Copy, Debug)]
struct CatalogEntry {
    name: &'static str,
    doodson: Option<[i8; 6]>,
    semi: Option<f64>,
    rayleigh_cph: f64,
    reference_frequency_cph: f64,
    satellite_start: usize,
    satellite_len: usize,
    shallow_start: usize,
    shallow_len: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Satellite {
    pub(crate) delta_doodson: [i8; 3],
    pub(crate) phase_correction: f64,
    pub(crate) amplitude_ratio: f64,
    pub(crate) latitude_factor: u8,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ShallowTerm {
    pub(crate) parent_index: usize,
    pub(crate) coefficient: f64,
}

#[allow(
    clippy::unreadable_literal,
    reason = "generated decimal spellings preserve Python float representations"
)]
mod generated {
    use super::{CatalogEntry, Satellite, ShallowTerm};

    include!("catalog_generated.rs");
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{CONSTITUENT_COUNT, TidalConstituent, generated};

    #[test]
    fn generated_catalog_has_expected_dimensions_and_unique_names() {
        assert_eq!(generated::CATALOG.len(), CONSTITUENT_COUNT);
        assert_eq!(generated::SATELLITES.len(), 162);
        assert_eq!(generated::SHALLOW_TERMS.len(), 251);
        let names = TidalConstituent::all()
            .map(TidalConstituent::name)
            .collect::<HashSet<_>>();
        assert_eq!(names.len(), CONSTITUENT_COUNT);
    }

    #[test]
    fn generated_ranges_and_relationships_are_consistent() {
        for constituent in TidalConstituent::all() {
            let metadata = constituent.metadata();
            assert!(
                metadata
                    .satellites
                    .iter()
                    .all(|satellite| satellite.latitude_factor <= 2)
            );
            assert!(metadata.shallow_terms.iter().all(|term| {
                term.parent_index < CONSTITUENT_COUNT
                    && !TidalConstituent::from_index(term.parent_index)
                        .expect("generated parent index is valid")
                        .is_shallow()
            }));
            assert_eq!(metadata.doodson.is_none(), constituent.is_shallow());
            assert_eq!(metadata.semi.is_none(), constituent.is_shallow());
        }
    }

    #[test]
    fn conventional_names_recover_compatibility_constants() {
        for constituent in [
            TidalConstituent::M2,
            TidalConstituent::S2,
            TidalConstituent::N2,
            TidalConstituent::K1,
            TidalConstituent::O1,
        ] {
            assert_eq!(constituent.name().parse(), Ok(constituent));
        }
        assert_eq!(TidalConstituent::M2.index(), 47);
        assert_eq!(TidalConstituent::from_name("m2"), None);
    }
}
