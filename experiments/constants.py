VALIDATION_STATIONS = [
    "fowd_cdip_163p1",
    "fowd_cdip_168p1",
    "fowd_cdip_181p1",
    "fowd_cdip_185p1",
    "fowd_cdip_187p1",
    "fowd_cdip_188p1",
    "fowd_cdip_189p1",
    "fowd_cdip_194p1",
    "fowd_cdip_196p1",
    "fowd_cdip_197p1",
    "fowd_cdip_198p1",
    "fowd_cdip_200p1",
    "fowd_cdip_201p1",
    "fowd_cdip_203p1",
    "fowd_cdip_204p1",
    "fowd_cdip_209p1",
    "fowd_cdip_213p1",
    "fowd_cdip_214p1",
    "fowd_cdip_215p1",
    "fowd_cdip_217p1",
    "fowd_cdip_220p1",
    "fowd_cdip_222p1",
    "fowd_cdip_225p1",
    "fowd_cdip_226p1",
    "fowd_cdip_233p1",
    "fowd_cdip_433p1",
]

FEATURE_NAMES = {
    "sea_state_dynamic_significant_wave_height_spectral": "Significant wave height",
    "sea_state_dynamic_mean_period_spectral": "Mean period",
    "sea_state_dynamic_peak_relative_depth_log10": "Relative depth (log$_{10}$)",
    "sea_state_dynamic_peak_relative_depth": "Relative depth",
    "sea_state_dynamic_steepness": "Peak steepness",
    "sea_state_dynamic_steepness_mean": "Mean steepness",
    "sea_state_dynamic_crest_trough_correlation": "Crest-trough correlation",
    "direction_dominant_spread": "Directional spread",
    "sea_state_dynamic_benjamin_feir_index_peakedness": "Benjamin-Feir index",
    "sea_state_dynamic_peak_ursell_number_log10": "Ursell number (log$_{10}$)",
    "sea_state_dynamic_rel_energy_in_frequency_interval_4": "Rel. wind energy",
    "sea_state_dynamic_bandwidth_peakedness": "Spectral peakedness",
    "sea_state_dynamic_bandwidth_narrowness": "Spectral narrowness",
    "sea_state_dynamic_kurtosis": "Kurtosis",
    "sea_state_dynamic_skewness": "Skewness",
    "sea_state_dynamic_peak_wavelength": "Peak wavelenggth",
    "direction_directionality_index": "Directionality index",
    "direction_directionality_index_log10": "Directionality index (log$_{10}$)",
}


DATA_SUBSETS = {
    "socal": dict(
        constraints={
            "meta_deploy_longitude": (-123.5, -117),
            "meta_deploy_latitude": (32, 38),
        }
    ),
    "deep-stations": dict(
        constraints={
            "meta_water_depth": (1000, None),
        }
    ),
    "shallow-stations": dict(
        constraints={
            "meta_water_depth": (None, 100),
        }
    ),
    "summer": dict(
        constraints={
            "day_of_year": (160, 220),
        }
    ),
    "winter": dict(
        constraints={
            "day_of_year": (0, 60),
        }
    ),
    "Hs > 3m": dict(
        constraints={
            "sea_state_dynamic_significant_wave_height_spectral": (3, None),
        },
    ),
    "high-frequency": dict(
        constraints={
            "sea_state_dynamic_rel_energy_in_frequency_interval_2": (None, 0.15),
        }
    ),
    "low-frequency": dict(
        constraints={
            "sea_state_dynamic_rel_energy_in_frequency_interval_2": (0.7, None),
        }
    ),
    "low-period": dict(
        constraints={
            "sea_state_dynamic_mean_period_direct": (None, 6.0),
        }
    ),
    "high-period": dict(
        constraints={
            "sea_state_dynamic_mean_period_direct": (9.0, None),
        }
    ),
    "cnoidal": dict(
        constraints={
            "sea_state_dynamic_peak_ursell_number": (8, None),
        }
    ),
    "weakly-nonlinear": dict(
        constraints={
            "sea_state_dynamic_steepness": (0.04, None),
        }
    ),
    "narrow": dict(
        constraints={
            "direction_dominant_spread": (None, 20),
        }
    ),
    "wide": dict(
        constraints={
            "direction_dominant_spread": (40, None),
        }
    ),
}

# drop all data that falls outside these ranges
BASE_CONSTRAINTS = (
    # excessive skewness
    ("sea_state_dynamic_skewness", -0.5, 0.5),
    # excessive low-frequency drift
    ("sea_state_dynamic_rel_energy_in_frequency_interval_1", 0, 0.1),
    # just no
    ("sea_state_dynamic_kurtosis", -1, 1),
    ("sea_state_dynamic_valid_data_ratio", 0.95, 1.0),
)


LABEL_FEATURE = "aggregate_100_max_rel_wave_height"
ROGUE_WAVE_THRESHOLD = 2.0

# set random state to ensure reproducibility
RANDOM_SEED = 17

L1_REG = 0
L2_REG = 1e-5

LEARNING_RATE = 1e-4
MLP_LAYERS = (32, 16, 8)

TRAIN_SIZE = 0.6

EPOCHS = 50
SWAG_EPOCHS = 50

REPORT_FILE = "results.json"
