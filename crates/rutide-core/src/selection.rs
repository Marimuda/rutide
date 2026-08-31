//! Record-length-based constituent selection.

use crate::{AnalysisError, TidalConstituent, scalar::validate_time};

/// Auditable result of UTide-compatible Rayleigh constituent selection.
#[derive(Clone, Debug, PartialEq)]
pub struct RayleighSelection {
    /// Dimensionless minimum Rayleigh criterion supplied by the caller.
    pub rayleigh_min: f64,
    /// Difference between the final and first timestamp, in days.
    pub record_span_days: f64,
    /// Minimum accepted catalog separation, in cycles per hour.
    pub minimum_separation_cph: f64,
    /// Selected constituents in stable oracle catalog order.
    pub constituents: Vec<TidalConstituent>,
}

/// Select the catalog entries resolved by a record's Rayleigh criterion.
///
/// This matches Python `UTide`'s automatic selection rule:
/// `minimum_separation_cph = rayleigh_min / (24 * record_span_days)`, followed
/// by `catalog.df >= minimum_separation_cph` in catalog order.
///
/// # Errors
///
/// Returns [`AnalysisError`] for invalid timestamps, a non-finite or non-positive
/// criterion, or a criterion that selects no constituents.
pub fn select_constituents_by_rayleigh(
    modified_julian_days: &[f64],
    rayleigh_min: f64,
) -> Result<RayleighSelection, AnalysisError> {
    if !rayleigh_min.is_finite() || rayleigh_min <= 0.0 {
        return Err(AnalysisError::InvalidRayleighMinimum);
    }
    let (_, record_span_days) = validate_time(modified_julian_days, 0)?;
    let minimum_separation_cph = rayleigh_min / (24.0 * record_span_days);
    let constituents = TidalConstituent::all()
        .filter(|constituent| constituent.rayleigh_separation_cph() >= minimum_separation_cph)
        .collect::<Vec<_>>();
    if constituents.is_empty() {
        return Err(AnalysisError::EmptyConstituents);
    }
    Ok(RayleighSelection {
        rayleigh_min,
        record_span_days,
        minimum_separation_cph,
        constituents,
    })
}

#[cfg(test)]
mod tests {
    use super::select_constituents_by_rayleigh;
    use crate::AnalysisError;

    fn fixture_times() -> Vec<f64> {
        (0_u32..745)
            .map(|index| 58_113.0 + f64::from(index) / 24.0)
            .collect()
    }

    #[test]
    fn matches_pinned_utide_selection_for_fixture_span() {
        let selection =
            select_constituents_by_rayleigh(&fixture_times(), 1.0).expect("valid selection");
        let names = selection
            .constituents
            .iter()
            .map(|constituent| constituent.name())
            .collect::<Vec<_>>();
        assert_eq!(selection.record_span_days.to_bits(), 31.0_f64.to_bits());
        assert_eq!(
            selection.minimum_separation_cph.to_bits(),
            (1.0_f64 / (24.0 * 31.0)).to_bits()
        );
        assert_eq!(
            names,
            [
                "MSF", "2Q1", "Q1", "O1", "NO1", "K1", "J1", "OO1", "UPS1", "N2", "M2", "S2",
                "ETA2", "MO3", "M3", "MK3", "SK3", "MN4", "M4", "MS4", "S4", "2MK5", "2SK5",
                "2MN6", "M6", "2MS6", "2SM6", "3MK7", "M8",
            ]
        );
    }

    #[test]
    fn rejects_invalid_or_unresolvable_criteria() {
        for criterion in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                select_constituents_by_rayleigh(&fixture_times(), criterion),
                Err(AnalysisError::InvalidRayleighMinimum)
            );
        }
        assert_eq!(
            select_constituents_by_rayleigh(&fixture_times(), 1_000.0),
            Err(AnalysisError::EmptyConstituents)
        );
    }
}
