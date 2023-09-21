import sys
import math
import json
import numpy as np


CMAP = [
    [244, 109, 67],
    [253, 174, 97],
    [254, 224, 139],
    [217, 239, 139],
    [166, 217, 106],
    [102, 189, 99],
]

# out of range colors
CMAP_OOR = [
    [215, 48, 39],
    [26, 152, 80],
]

SKELETON = """
%s

\\begingroup
\\renewcommand{\\arraystretch}{1.2}
\\begin{tabular}{rlllrrr}
    & \\multicolumn{3}{c}{Feature groups} & \\multicolumn{3}{c}{Scores} \\\\ \\cmidrule(lr){2-4} \\cmidrule(lr){5-7}
    ID & 1 & 2 & 3 & $\\mathcal{L} \\times 10^4$ & $\\mathcal{E} \\times 10^2$ & $\\mathcal{C} \\times 10^2$ \\\\ \\midrule
    %s
    \\bottomrule
\\end{tabular}
\\endgroup

\\vspace{2em}

\\begingroup
\\renewcommand{\\arraystretch}{1.0}
\\begin{tabular}{ll@{\\hskip 3em}ll}
    \\multicolumn{4}{l}{Symbols} \\\\
    \\toprule
    $r$ & Crest-trough correlation &
    $\\nu$ & Spectral bandwidth (narrowness) \\\\
    $\\sigma_f$ & Spectral bandwidth (peakedness) &
    $\\sigma_\\theta$ & Directional spread \\\\
    $\\varepsilon$ & Peak steepness $H_s k_p$ &
    $R$ & Directionality index $\\sigma_\\theta^2 / (2 \\nu^2)$ \\\\
    $\\mathrm{BFI}$ & Benjamin-Feir index &
    $\\widetilde{D}$ & Relative peak water depth $D k_p / (2\\pi)$ \\\\
    $E_h$ & Relative high-frequency energy &
    $\\mathrm{Ur}$ & Ursell number \\\\
    $\\overline{T}$ & Mean period &
    $\\kappa$ & Kurtosis \\\\
    $\\mu$ & Skewness &
    $H_s$ & Significant wave height \\\\
    \\bottomrule
\\end{tabular}
\\endgroup
"""

FEATURE_TO_SYMBOL = {
    "sea_state_dynamic_peak_relative_depth_log10": "\\widetilde{D}",
    "sea_state_dynamic_peak_relative_depth": "\\widetilde{D}",
    "sea_state_dynamic_steepness": "\\varepsilon",
    "sea_state_dynamic_steepness_mean": "\\varepsilon_m",
    "sea_state_dynamic_crest_trough_correlation": "r",
    "direction_dominant_spread": "\\sigma_\\theta",
    "sea_state_dynamic_benjamin_feir_index_peakedness": "\\mathrm{BFI}",
    "sea_state_dynamic_peak_ursell_number_log10": "\\mathrm{Ur}",
    "sea_state_dynamic_rel_energy_in_frequency_interval_4": "E_h",
    "sea_state_dynamic_bandwidth_peakedness": "\\sigma_f",
    "sea_state_dynamic_bandwidth_narrowness": "\\nu",
    "direction_directionality_index": "R",
    "direction_directionality_index_log10": "R",
    "sea_state_dynamic_mean_period_spectral": "\\overline{T}",
    "sea_state_dynamic_rel_energy_in_frequency_interval_2": "E_l",
    "sea_state_dynamic_kurtosis": "\\kappa",
    "sea_state_dynamic_skewness": "\\mu",
    "sea_state_dynamic_significant_wave_height_spectral": "H_s",
    "sea_state_dynamic_peak_wavelength": "\\lambda_p",
}


def generate_table(datafile):
    preamble = []
    body = []

    with open(datafile, "r") as f:
        data = json.load(f)

    # sort by total number of parameter interactions
    num_interactions = lambda n: 0.5 * (n - 1)**2 + n
    data = sorted(data, key=lambda d: sum(num_interactions(len(k)) for k in d["feature_groups"]), reverse=False)

    scores = {}
    for row in data:
        for col in ("consistency_score", "consistency_err", "calibration_err_val"):
            if col not in scores:
                scores[col] = []
            scores[col].append(row["scores"][col])

    score_limits = {}
    for key, val in scores.items():
        qlow, qmed, qhigh = np.quantile(val, [0.25, 0.5, 0.75])
        iqr = (qhigh - qlow)
        score_limits[key] = (qmed - iqr, qmed + iqr)

    def get_color(val, limits, reverse=False):
        val_norm = (val - limits[0]) / (limits[1] - limits[0])
        val_idx = math.floor(val_norm * len(CMAP))

        if reverse:
            val_idx = len(CMAP) - val_idx - 1

        if val_idx < 0:
            return "Clow"

        if val_idx >= len(CMAP):
            return "Chigh"

        return f"C{val_idx}"

    for i, color in enumerate(CMAP):
        preamble.append(f"\\definecolor{{C{i}}}{{RGB}}{{{', '.join(map(str, color))}}}")

    preamble.append(f"\\definecolor{{Clow}}{{RGB}}{{{', '.join(map(str, CMAP_OOR[0]))}}}")
    preamble.append(f"\\definecolor{{Chigh}}{{RGB}}{{{', '.join(map(str, CMAP_OOR[1]))}}}")

    def get_feature_group_str(feature_groups, i):
        if len(feature_groups) <= i:
            return " "

        return "$\\{" + "$, $".join(FEATURE_TO_SYMBOL[f] for f in d["feature_groups"][i]) + "\\}$"

    conscore = "consistency_score"
    conerr = "consistency_err"
    calerr = "calibration_err_val"

    highlighted_model = 18

    for n, d in enumerate(data, 1):
            
        line = [
            str(n),
            get_feature_group_str(d["feature_groups"], 0),
            get_feature_group_str(d["feature_groups"], 1),
            get_feature_group_str(d["feature_groups"], 2),
            f"\\cellcolor{{{get_color(d['scores'][conscore], score_limits[conscore])}}} \\num{{{d['scores'][conscore]*1e4:.2f}}}",
            f"\\cellcolor{{{get_color(d['scores'][conerr], score_limits[conerr], reverse=True)}}} \\num{{{d['scores'][conerr]*1e2:.2f}}}",
            f"\\cellcolor{{{get_color(d['scores'][calerr], score_limits[calerr], reverse=True)}}} \\num{{{d['scores'][calerr]*1e2:.2f}}}",
        ]
        if n == highlighted_model:
            line = [f"\\boldmath \\bfseries {l}" for l in line]

        body.append(" & ".join(line) + "\\\\")

        if n % 4 == 0 and n != len(data):
            body.append("&" * (len(line) - 1) + "\\\\[-1.2ex]")

    return SKELETON % ("\n".join(preamble), "\n".join(body))


if __name__ == "__main__":
    table_code = generate_table(sys.argv[1])

    with open("generated/experiment-table.tex", "w") as f:
        f.write(table_code)