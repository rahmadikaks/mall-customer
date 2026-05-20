from __future__ import annotations
import io
import warnings
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, chatterjeexi
import dcor
import arviz as az
import pymc as pm
import panel as pn
import param
import threading
pn.extension("tabulator", sizing_mode="stretch_width")

# =============================================================================
# DataValidator
# =============================================================================

_DOMAIN_CHECKS: dict[str, dict[str, Any]] = {
    "gaussian":  {"min": None, "max": None,  "integer": False, "open": False},
    "beta":      {"min": 0.0,  "max": 1.0,   "integer": False, "open": True },
    "lognormal": {"min": 0.0,  "max": None,  "integer": False, "open": True },
    "binomial":  {"min": 0.0,  "max": None,  "integer": True,  "open": False},
}
_MIN_SAMPLE_SIZE = 10

@dataclass
class ValidationReport:
    n_rows          : int
    n_cols          : int
    n_duplicates    : int
    missing_summary : pd.DataFrame
    dtype_summary   : pd.DataFrame
    domain_issues   : list[str] = field(default_factory=list)
    warnings        : list[str] = field(default_factory=list)
    errors          : list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


class DataValidationError(ValueError):
    pass


class DataValidator:
    VALID_HANDLE_MISSING = {"deleted", "keep"}

    def __init__(self, handle_missing: str = "deleted") -> None:
        if handle_missing not in self.VALID_HANDLE_MISSING:
            raise ValueError(f"handle_missing harus salah satu dari {self.VALID_HANDLE_MISSING}.")
        self._handle_missing = handle_missing

    def validate(
        self,
        data: pd.DataFrame,
        response: str,
        predictors: list[str],
        *,
        family: str,
        group: str | None = None,
    ) -> pd.DataFrame:
        report = self.check(data, response, predictors, family=family, group=group)
        if not report.is_valid:
            msg = "Data validation failed:\n" + "\n".join(f"  [ERROR] {e}" for e in report.errors)
            raise DataValidationError(msg)
        df   = data.copy()
        cols = [response] + list(predictors) + ([group] if group else [])
        if self._handle_missing == "deleted":
            df = df.dropna(subset=cols)
        return df

    def check(
        self,
        data: pd.DataFrame,
        response: str | None = None,
        predictors: list[str] | None = None,
        *,
        family: str | None = None,
        group: str | None = None,
    ) -> ValidationReport:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"data harus pd.DataFrame, bukan {type(data).__name__}.")

        errors, warns, domain_issues = [], [], []
        n_rows, n_cols = data.shape

        if n_rows < _MIN_SAMPLE_SIZE:
            warns.append(
                f"Jumlah observasi sangat kecil ({n_rows} baris). "
                f"Disarankan minimal {_MIN_SAMPLE_SIZE} observasi."
            )

        n_duplicates = int(data.duplicated().sum())
        if n_duplicates > 0:
            warns.append(f"Ditemukan {n_duplicates} baris duplikat.")

        missing_counts  = data.isna().sum()
        missing_pct     = (missing_counts / n_rows * 100).round(2)
        missing_summary = pd.DataFrame({
            "Column":      missing_counts.index,
            "Missing (n)": missing_counts.values,
            "Missing (%)": missing_pct.values,
            "Has Missing": (missing_counts > 0).values,
        }).reset_index(drop=True)

        total_missing = int(missing_counts.sum())
        if total_missing > 0:
            cols_miss = missing_counts[missing_counts > 0].index.tolist()
            warns.append(
                f"Ditemukan {total_missing} missing value di kolom: {', '.join(cols_miss)}."
            )

        dtype_summary = pd.DataFrame({
            "Column":   data.columns,
            "Dtype":    [str(dt) for dt in data.dtypes],
            "Type":     [
                "Numeric" if pd.api.types.is_numeric_dtype(data[c]) else "Categorical"
                for c in data.columns
            ],
            "N Unique": [data[c].nunique() for c in data.columns],
        }).reset_index(drop=True)

        all_required = []
        if response:   all_required.append(response)
        if predictors: all_required.extend(predictors)
        if group:      all_required.append(group)
        for col in all_required:
            if col not in data.columns:
                errors.append(f"Kolom '{col}' tidak ditemukan di data.")

        if response and response in data.columns:
            if not pd.api.types.is_numeric_dtype(data[response]):
                errors.append(
                    f"Kolom response '{response}' harus numerik, bukan {data[response].dtype}."
                )
        if predictors:
            for p in predictors:
                if p in data.columns and not pd.api.types.is_numeric_dtype(data[p]):
                    warns.append(
                        f"Kolom prediktor '{p}' bukan numerik ({data[p].dtype}). "
                        "Pastikan sudah di-encode sebelum modelling."
                    )

        if family and response and response in data.columns:
            domain_issues = self._check_domain(data[response], family)
            errors.extend(domain_issues)

        return ValidationReport(
            n_rows=n_rows, n_cols=n_cols, n_duplicates=n_duplicates,
            missing_summary=missing_summary, dtype_summary=dtype_summary,
            domain_issues=domain_issues, warnings=warns, errors=errors,
        )

    def _check_domain(self, y: pd.Series, family: str) -> list[str]:
        issues = []
        spec   = _DOMAIN_CHECKS.get(family)
        if spec is None:
            issues.append(
                f"Family '{family}' tidak dikenal. Pilih: {list(_DOMAIN_CHECKS.keys())}."
            )
            return issues
        y_valid = y.dropna()
        if spec["min"] is not None:
            violated = (y_valid <= spec["min"]).sum() if spec["open"] else (y_valid < spec["min"]).sum()
            op       = ">"  if spec["open"] else ">="
            if violated:
                issues.append(f"Family '{family}' memerlukan y {op} {spec['min']}, "
                               f"tapi ditemukan {violated} nilai yang melanggar.")
        if spec["max"] is not None:
            violated = (y_valid >= spec["max"]).sum() if spec["open"] else (y_valid > spec["max"]).sum()
            op       = "<"  if spec["open"] else "<="
            if violated:
                issues.append(f"Family '{family}' memerlukan y {op} {spec['max']}, "
                               f"tapi ditemukan {violated} nilai yang melanggar.")
        if spec["integer"]:
            non_int = (~y_valid.apply(float.is_integer)).sum() if y_valid.dtype == float else 0
            if non_int:
                issues.append(f"Family '{family}' memerlukan y bilangan bulat, "
                               f"tapi ditemukan {non_int} nilai non-integer.")
        return issues


# =============================================================================
# Shared State
# =============================================================================
class AppState(param.Parameterized):
    """State global yang di-share ke semua tab."""
    data  : pd.DataFrame | None = param.Parameter(default=None)
    model : object | None       = param.Parameter(default=None)

# =============================================================================
# DataTab 
# =============================================================================
class DataTab(param.Parameterized):
    state: AppState = param.Parameter()

    def __init__(self, state: AppState, **params):
        super().__init__(state=state, **params)

        self._validator = DataValidator(handle_missing="deleted")

        self._file_input = pn.widgets.FileInput(
            accept=".csv,.xls,.xlsx", name="Upload CSV or Excel File"
        )
        self._file_input.param.watch(self._on_upload, "value")

        self._response_sel   = pn.widgets.Select(name="Response Column (y)", options=[])
        self._predictors_sel = pn.widgets.MultiSelect(
            name="Predictor Columns (x)", options=[], size=5
        )
        self._family_sel = pn.widgets.Select(
            name="HB Family",
            options={"Gaussian": "gaussian", "Beta": "beta",
                     "Lognormal": "lognormal", "Binomial": "binomial"},
            value="gaussian",
        )
        self._group_sel = pn.widgets.Select(
            name="Group/Area Column (optional)", options=[], value=None
        )
        self._check_btn   = pn.widgets.Button(name="Validate Data", button_type="primary")
        self._confirm_btn = pn.widgets.Button(name="Confirm & Use Data",
                                              button_type="success", disabled=True)
        self._check_btn.on_click(self._on_check)
        self._confirm_btn.on_click(self._on_confirm)

        self._summary      = pn.pane.Markdown("*No data uploaded yet*")
        self._var_badges   = pn.pane.HTML("")
        self._preview      = pn.widgets.Tabulator(
            pd.DataFrame(), pagination="remote", page_size=10, show_index=False
        )

        self._val_status   = pn.pane.HTML("")           
        self._val_errors   = pn.pane.Markdown("")       
        self._val_warnings = pn.pane.Markdown("")       
        self._val_missing  = pn.widgets.Tabulator(
            pd.DataFrame(), show_index=False, height=200
        )
        self._val_dtypes   = pn.widgets.Tabulator(
            pd.DataFrame(), show_index=False, height=200
        )

        self._df_raw: pd.DataFrame | None = None


    def _on_upload(self, event):
        if not self._file_input.value:
            return
        fname = self._file_input.filename
        raw   = io.BytesIO(self._file_input.value)
        df    = pd.read_csv(raw) if fname.endswith(".csv") else pd.read_excel(raw)
        self._df_raw = df
        self._refresh_preview(df)
        self._reset_validation_ui()
        self._confirm_btn.disabled = True

    def _refresh_preview(self, df: pd.DataFrame):
        n_rows, n_cols = df.shape
        n_miss         = int(df.isna().sum().sum())
        self._summary.object = (
            f"**Total Rows:** {n_rows} &nbsp;|&nbsp; "
            f"**Total Columns:** {n_cols} &nbsp;|&nbsp; "
            f"**Missing Values:** {n_miss}"
        )
        badges = " ".join(
            f'<span style="background:#0072B2;color:white;padding:4px 10px;'
            f'border-radius:20px;margin:3px;display:inline-block">{c}</span>'
            for c in df.columns
        )
        self._var_badges.object = badges
        self._preview.value     = df

        cols     = df.columns.tolist()
        num_cols = df.select_dtypes(include="number").columns.tolist()
        self._response_sel.options   = num_cols
        self._response_sel.value     = num_cols[0] if num_cols else None
        self._predictors_sel.options = num_cols
        self._predictors_sel.value   = num_cols[1:] if len(num_cols) > 1 else []
        self._group_sel.options      = [None] + cols
        self._group_sel.value        = None

    def _reset_validation_ui(self):
        self._val_status.object   = ""
        self._val_errors.object   = ""
        self._val_warnings.object = ""
        self._val_missing.value   = pd.DataFrame()
        self._val_dtypes.value    = pd.DataFrame()

    def _on_check(self, event):
        if self._df_raw is None:
            self._val_status.object = (
                '<span style="color:orange;font-weight:bold">⚠ Upload data terlebih dahulu.</span>'
            )
            return

        response   = self._response_sel.value
        predictors = list(self._predictors_sel.value)
        family     = self._family_sel.value
        group      = self._group_sel.value or None

        report = self._validator.check(
            self._df_raw,
            response=response,
            predictors=predictors,
            family=family,
            group=group,
        )
        self._render_report(report)
        self._confirm_btn.disabled = not report.is_valid

    def _render_report(self, report: ValidationReport):
        if report.is_valid:
            self._val_status.object = (
                '<span style="background:#2ca02c;color:white;padding:6px 18px;'
                'border-radius:20px;font-weight:bold;font-size:1.05em">✔ VALID</span>'
            )
        else:
            self._val_status.object = (
                f'<span style="background:#d62728;color:white;padding:6px 18px;'
                f'border-radius:20px;font-weight:bold;font-size:1.05em">'
                f'INVALID — {len(report.errors)} error</span>'
            )

        if report.errors:
            errs = "\n".join(f"- {e}" for e in report.errors)
            self._val_errors.object = f"**Errors (fatal):**\n{errs}"
        else:
            self._val_errors.object = "**Errors:** *(none)*"

        if report.warnings:
            warns = "\n".join(f"- {w}" for w in report.warnings)
            self._val_warnings.object = f"**Warnings:**\n{warns}"
        else:
            self._val_warnings.object = "**Warnings:** *(none)*"

        self._val_missing.value = report.missing_summary
        self._val_dtypes.value  = report.dtype_summary

    def _on_confirm(self, event):
        if self._df_raw is None:
            return
        response   = self._response_sel.value
        predictors = list(self._predictors_sel.value)
        family     = self._family_sel.value
        group      = self._group_sel.value or None
        try:
            df_clean = self._validator.validate(
                self._df_raw,
                response=response,
                predictors=predictors,
                family=family,
                group=group,
            )
            self.state.data = df_clean
            self._val_status.object = (
                f'<span style="background:#2ca02c;color:white;padding:6px 18px;'
                f'border-radius:20px;font-weight:bold;font-size:1.05em">'
                f'✔ Data confirmed — {len(df_clean)} rows ready</span>'
            )
        except DataValidationError as exc:
            self._val_status.object = (
                f'<span style="background:#d62728;color:white;padding:6px 18px;'
                f'border-radius:20px;font-weight:bold">✘ {exc}</span>'
            )

    def panel(self) -> pn.Column:
        guidelines = pn.pane.Markdown("""
                                      **Accepted formats:** `.csv`, `.xls`, `.xlsx`
                                      **Dataset Structure:**
                                      - Tabular format; each row = one observation/area.
                                      - First row must contain column names (headers).
                                      - Include a column for direct estimates.
                                      - Include one or more auxiliary variables (predictors).
                                      - Optionally include a unique identifier column per area.
                                      **File format notes:**
                                      - `.csv`: comma (`,`) separator, period (`.`) decimal.
                                      - `.xlsx/.xls`: may use comma as decimal (European settings) — check numeric columns.
                                      **Missing Values:** auto-detected; handling method chosen at modelling stage.""")

        validation_panel = pn.Column(
            pn.Row(self._response_sel, self._family_sel, self._group_sel),
            self._predictors_sel,
            pn.Row(self._check_btn, self._confirm_btn),
            pn.layout.Divider(),
            self._val_status,
            self._val_errors,
            self._val_warnings,
            pn.Tabs(
                ("Missing Values",  self._val_missing),
                ("Column Dtypes",   self._val_dtypes),
            ),
        )

        return pn.Column(
            pn.Card(guidelines,           title="Data Requirements & Format Guidelines", margin=10),
            pn.Card(self._file_input,     title="Upload File",                           margin=10),
            pn.Card(self._summary,        title="Summary Dataset",                       margin=10),
            pn.Card(self._var_badges,     title="Variable List",                         margin=10),
            pn.Card(self._preview,        title="Data Preview",                          margin=10),
            pn.Card(validation_panel,     title="Data Validation",                       margin=10),
        )

# =============================================================================
# ExploreTab 
# =============================================================================
class ExploreTab(param.Parameterized):
    state: AppState = param.Parameter()

    def __init__(self, state: AppState, **params):
        super().__init__(state=state, **params)

        self._hist_var = pn.widgets.Select(name="Variable (Histogram)", options=[])
        self._n_bins   = pn.widgets.IntSlider(name="Number of bins", start=1, end=100, value=20)
        self._box_var  = pn.widgets.Select(name="Variable (Boxplot)", options=[])
        self._x_var    = pn.widgets.Select(name="X-axis Variable", options=[])
        self._y_var    = pn.widgets.Select(name="Y-axis Variable", options=[])
        self._x_trans  = pn.widgets.Select(
            name="X Transformation",
            options={"None": "none", "Log": "log", "Z-score": "zscore"},
            value="none",
        )
        self._y_trans  = pn.widgets.Select(
            name="Y Transformation",
            options={"None": "none", "Log": "log", "Logit": "logit", "Z-score": "zscore"},
            value="none",
        )

        self._summary_table = pn.widgets.Tabulator(pd.DataFrame(), show_index=False)
        self._hist_pane     = pn.pane.Matplotlib(sizing_mode="stretch_width", tight=True)
        self._box_pane      = pn.pane.Matplotlib(sizing_mode="stretch_width", tight=True)
        self._corr_title    = pn.pane.Markdown("")
        self._corr_table    = pn.widgets.Tabulator(pd.DataFrame(), show_index=False)
        self._scatter_pane  = pn.pane.Matplotlib(sizing_mode="stretch_width", tight=True)

        self.state.param.watch(self._on_data_change, "data")

        self._hist_var.param.watch(self._update_histogram, "value")
        self._n_bins.param.watch(self._update_histogram, "value")
        self._box_var.param.watch(self._update_boxplot, "value")
        for w in [self._x_var, self._y_var, self._x_trans, self._y_trans]:
            w.param.watch(self._update_scatter_corr, "value")

    def _on_data_change(self, event):
        df = self.state.data
        if df is None:
            return
        cols  = df.select_dtypes(include="number").columns.tolist()
        first = cols[0] if cols else None
        for w in [self._hist_var, self._box_var, self._x_var, self._y_var]:
            w.options = cols
            w.value   = first
        self._update_summary()
        self._update_histogram()
        self._update_boxplot()
        self._update_scatter_corr()

    def _numeric_df(self) -> pd.DataFrame | None:
        df = self.state.data
        return df.select_dtypes(include="number") if df is not None else None

    def _update_summary(self):
        df = self._numeric_df()
        if df is None:
            return
        self._summary_table.value = (
            df.describe().T.reset_index()
            .rename(columns={
                "index": "Variable", "min": "Min", "25%": "1st Qu.",
                "50%": "Median", "mean": "Mean", "75%": "3rd Qu.", "max": "Max",
            })
            .round(3)
        )

    def _update_histogram(self, *_):
        df  = self._numeric_df()
        var = self._hist_var.value
        if df is None or not var:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[var].dropna(), bins=self._n_bins.value, kde=True, ax=ax)
        ax.set_title(f"Histogram of {var}")
        ax.set_xlabel(var); ax.set_ylabel("Density")
        plt.tight_layout()
        self._hist_pane.object = fig
        plt.close(fig)

    def _update_boxplot(self, *_):
        df  = self._numeric_df()
        var = self._box_var.value
        if df is None or not var:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(df[var].dropna(), ax=ax)
        ax.set_title(f"Boxplot of {var}")
        ax.set_xlabel(var); ax.set_ylabel("Value")
        plt.tight_layout()
        self._box_pane.object = fig
        plt.close(fig)

    def _transformed_xy(self):
        df    = self._numeric_df()
        x_var = self._x_var.value
        y_var = self._y_var.value
        if df is None or not x_var or not y_var:
            return None
        x, y           = df[x_var].copy(), df[y_var].copy()
        x_name, y_name = x_var, y_var

        if self._x_trans.value == "log":
            x = np.log(np.maximum(x, 1e-6)); x_name = f"{x_var}_log"
        elif self._x_trans.value == "zscore":
            x = (x - x.mean()) / x.std();    x_name = f"{x_var}_zscore"

        if self._y_trans.value == "log":
            y = np.log(np.maximum(y, 1e-6)); y_name = f"{y_var}_log"
        elif self._y_trans.value == "logit":
            y_clip = np.clip(y, 1e-6, 1 - 1e-6)
            y = np.log(y_clip / (1 - y_clip)); y_name = f"{y_var}_logit"
        elif self._y_trans.value == "zscore":
            y = (y - y.mean()) / y.std();    y_name = f"{y_var}_zscore"

        return x, y, x_name, y_name

    def _update_scatter_corr(self, *_):
        res = self._transformed_xy()
        if res is None:
            return
        x, y, x_name, y_name = res
        xn     = x.dropna().to_numpy()
        yn     = y.dropna().to_numpy()
        xi_res = chatterjeexi(xn, yn)

        rows = [
            ("Pearson's r",          *pearsonr(xn, yn)),
            ("Spearman's rho",       *spearmanr(xn, yn)),
            ("Chatterjee's Xi",      xi_res.statistic, xi_res.pvalue),
            ("Distance Correlation", dcor.distance_correlation(xn, yn), None),
        ]
        self._corr_title.object = f"### Correlation: **{x_name}** vs **{y_name}**"
        self._corr_table.value  = pd.DataFrame([
            {"Metric": name, "Value": round(val, 4),
             "p-value": round(pval, 4) if pval is not None else "N/A"}
            for name, val, pval in rows
        ])

        mask   = x.notna() & y.notna()
        xc, yc = x[mask].to_numpy(), y[mask].to_numpy()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(xc, yc, alpha=0.7)
        m, b   = np.polyfit(xc, yc, 1)
        x_line = np.linspace(xc.min(), xc.max(), 200)
        ax.plot(x_line, m * x_line + b, linestyle="--", color="red",
                linewidth=1.5, label="Trendline")
        ax.set_xlabel(x_name); ax.set_ylabel(y_name)
        ax.set_title(f"{x_name} vs {y_name}"); ax.legend()
        plt.tight_layout()
        self._scatter_pane.object = fig
        plt.close(fig)

    def panel(self) -> pn.Column:
        return pn.Column(
            pn.pane.Markdown("Explore dataset characteristics before modelling."),
            pn.Tabs(
                (
                    "Summary Statistics",
                    pn.Card(self._summary_table, title="Summary Statistics", margin=10),
                ),
                (
                    "Visualize Distribution",
                    pn.Row(
                        pn.Card(
                            pn.Column(self._hist_var, self._n_bins, self._hist_pane),
                            title="Histogram", margin=10,
                        ),
                        pn.Card(
                            pn.Column(self._box_var, self._box_pane),
                            title="Boxplot", margin=10,
                        ),
                    ),
                ),
                (
                    "Scatter & Correlation",
                    pn.Column(
                        pn.Row(self._x_var, self._y_var),
                        pn.Row(self._x_trans, self._y_trans),
                        self._corr_title,
                        self._corr_table,
                        self._scatter_pane,
                    ),
                ),
            ),
        )

# =============================================================================
# Konstanta
# =============================================================================
_FAMILY_OPTIONS: dict[str, str] = {
    "Gaussian":  "gaussian",
    "Beta":      "beta",
    "Lognormal": "lognormal",
    "Binomial":  "binomial",
}
 
_FAMILY_LINK_DEFAULT: dict[str, str] = {
    "gaussian":  "identity",
    "beta":      "logit",
    "lognormal": "log",
    "binomial":  "logit",
}
 
_FAMILY_LINK_OPTIONS: dict[str, list[str]] = {
    "gaussian":  ["identity", "log"],
    "beta":      ["logit", "probit", "cloglog"],
    "lognormal": ["log"],
    "binomial":  ["logit", "probit", "cloglog"],
}
 
_FAMILY_DESC: dict[str, str] = {
    "gaussian": (
        "**Gaussian** — response berskala kontinu tak terbatas. "
        "Link default: *identity*. Cocok untuk y ∈ ℝ."
    ),
    "beta": (
        "**Beta** — response berupa proporsi / rate ∈ (0, 1). "
        "Link default: *logit*. Parameter presisi φᵢ dihitung dari kolom `n` dan `deff`."
    ),
    "lognormal": (
        "**Lognormal** — response kontinu positif (y > 0). "
        "Link default: *log*."
    ),
    "binomial": (
        "**Binomial** — response berupa count sukses dari sejumlah trial. "
        "Link default: *logit*. Kolom `trials` **wajib** diisi."
    ),
}

# =============================================================================
# ModelTab
# =============================================================================
class ModelTab(param.Parameterized):
    state: "AppState" = param.Parameter() 
 
    def __init__(self, state: "AppState", **params):  
        super().__init__(state=state, **params)
 
        self._draws = pn.widgets.IntInput(
            name="Draws  (post-warmup per chain)",
            value=1000, start=1, step=100, width=230,
        )
        self._tune = pn.widgets.IntInput(
            name="Tune / Warmup  (steps per chain)",
            value=1000, start=0, step=100, width=230,
        )
        self._chains = pn.widgets.IntInput(
            name="Chains",
            value=4, start=1, end=16, width=230,
        )
        self._cores = pn.widgets.IntInput(
            name="CPU Cores",
            value=1, start=1, width=230,
        )
        self._target_accept = pn.widgets.FloatInput(
            name="Target Accept  (NUTS)",
            value=0.8, start=0.01, end=0.999, step=0.05, width=230,
        )
        self._random_seed = pn.widgets.IntInput(
            name="Random Seed  (blank = None)",
            value=None, start=0, width=230,
        )
        self._progressbar = pn.widgets.Checkbox(
            name="Show Progressbar", value=True,
        )

        self._total_draws_badge = pn.pane.HTML(self._render_draws_badge())
        self._draws.param.watch(self._update_draws_badge, "value")
        self._chains.param.watch(self._update_draws_badge, "value")
 
        self._sampler_apply_btn = pn.widgets.Button(
            name="✔  Apply Sampler Configuration",
            button_type="primary", width=230,
        )
        self._sampler_status = pn.pane.HTML("")
        self._sampler_applied = False         
        self._sampler_apply_btn.on_click(self._on_apply_sampler)
 
        self._response_sel = pn.widgets.Select(
            name="Response Variable  (y)",
            options=[], width=240,
        )
        self._predictors_sel = pn.widgets.MultiSelect(
            name="Auxiliary / Predictor Variables  (x)",
            options=[], size=5, width=240,
        )
        self._group_sel = pn.widgets.Select(
            name="Group / Area Variable  (optional)",
            options=[], value=None, width=240,
        )
 
        self._family_sel = pn.widgets.Select(
            name="HB Family",
            options=_FAMILY_OPTIONS, value="gaussian", width=210,
        )
        self._link_sel = pn.widgets.Select(
            name="Link Function",
            options=_FAMILY_LINK_OPTIONS["gaussian"],
            value=_FAMILY_LINK_DEFAULT["gaussian"],
            width=210,
        )
        self._family_desc = pn.pane.Markdown(
            _FAMILY_DESC["gaussian"], margin=(4, 0, 8, 0),
        )
        self._family_sel.param.watch(self._on_family_change, "value")
 
        self._beta_n_sel = pn.widgets.Select(
            name="n column  (survey sample size)",
            options=[], value=None, width=240,
        )
        self._beta_deff_sel = pn.widgets.Select(
            name="deff column  (design effect)",
            options=[], value=None, width=240,
        )
        self._binomial_trials_sel = pn.widgets.Select(
            name="trials column  (number of trials)",
            options=[], value=None, width=240,
        )
        self._extra_params_pane = pn.Column()   
 
        self._handle_missing_sel = pn.widgets.Select(
            name="Handle Missing Values",
            options={
                "Drop rows  (listwise deletion)": "deleted",
                "Keep  (pass-through)":           "keep",
            },
            value="deleted", width=290,
        )
 
        self._formula_preview = pn.pane.HTML(self._render_formula_html())
        for w in [self._response_sel, self._predictors_sel, self._group_sel]:
            w.param.watch(self._update_formula_preview, "value")

        self._build_btn = pn.widgets.Button(
            name="Build Model",
            button_type="success", width=200,
        )
        self._build_status  = pn.pane.HTML("")
        self._model_summary = pn.pane.HTML("")
        self._build_btn.on_click(self._on_build)
 
        self.state.param.watch(self._on_data_change, "data")
        if self.state.data is not None:
            self._populate_selectors(self.state.data)
 
    def _render_draws_badge(self) -> str:
        draws  = self._draws.value  or 1000
        chains = self._chains.value or 4
        return (
            f'<span style="background:#0072B2;color:white;padding:5px 14px;'
            f'border-radius:20px;font-weight:bold;font-size:0.95em">'
            f'Total Posterior Draws: {draws * chains:,}'
            f'&nbsp;({draws:,} draws × {chains} chains)</span>'
        )
 
    def _update_draws_badge(self, *_) -> None:
        self._total_draws_badge.object = self._render_draws_badge()
 
    def _validate_sampler(self) -> list[str]:
        errors: list[str] = []
        if not self._draws.value or self._draws.value < 1:
            errors.append("`Draws` harus ≥ 1.")
        if self._tune.value is None or self._tune.value < 0:
            errors.append("`Tune` harus ≥ 0.")
        if not self._chains.value or self._chains.value < 1:
            errors.append("`Chains` harus ≥ 1.")
        if not self._cores.value or self._cores.value < 1:
            errors.append("`CPU Cores` harus ≥ 1.")
        ta = self._target_accept.value
        if ta is None or not (0 < ta < 1):
            errors.append("`Target Accept` harus berada di antara 0 dan 1 (eksklusif).")
        return errors
 
    def _on_apply_sampler(self, event) -> None:
        errors = self._validate_sampler()
        if errors:
            self._sampler_applied = False
            self._sampler_status.object = _error_box(
                "⚠ Validation Error", "<br>".join(f"• {e}" for e in errors)
            )
            return
 
        self._sampler_applied = True
        self._sampler_status.object = _success_box(
            "✔ Sampler configuration applied. "
            "You may now proceed to <b>Model Building</b>."
        )
 
    def get_sampler_kwargs(self) -> dict:
        """Return dict siap di-unpack ke ``bambi.Model.fit(**kwargs)``."""
        return {
            "draws":         self._draws.value  or 1000,
            "tune":          self._tune.value   or 1000,
            "chains":        self._chains.value or 4,
            "cores":         self._cores.value  or 1,
            "target_accept": self._target_accept.value or 0.8,
            "random_seed":   self._random_seed.value,
            "progressbar":   self._progressbar.value,
        }
 
    def _on_data_change(self, event) -> None:
        if event.new is not None:
            self._populate_selectors(event.new)
 
    def _populate_selectors(self, df: pd.DataFrame) -> None:
        all_cols = df.columns.tolist()
        num_cols = df.select_dtypes(include="number").columns.tolist()
 
        self._response_sel.options   = num_cols
        self._response_sel.value     = num_cols[0] if num_cols else None
 
        self._predictors_sel.options = num_cols
        self._predictors_sel.value   = num_cols[1:] if len(num_cols) > 1 else []
 
        self._group_sel.options = [None] + all_cols
        self._group_sel.value   = None

        for w in [self._beta_n_sel, self._beta_deff_sel, self._binomial_trials_sel]:
            w.options = [None] + num_cols
            w.value   = None
 
        self._update_formula_preview()
 
    def _on_family_change(self, event) -> None:
        family = event.new
        self._link_sel.options = _FAMILY_LINK_OPTIONS[family]
        self._link_sel.value   = _FAMILY_LINK_DEFAULT[family]
        self._family_desc.object = _FAMILY_DESC[family]
        self._refresh_extra_params(family)
 
    def _refresh_extra_params(self, family: str) -> None:
        if family == "beta":
            self._extra_params_pane.objects = [
                pn.pane.Markdown(
                    "**Beta-specific parameters** — untuk menghitung φᵢ = nᵢ / deffᵢ − 1. "
                    "Kosongkan jika tidak tersedia (Bambi estimasi φ otomatis).",
                    margin=(4, 0, 6, 0),
                ),
                pn.Row(self._beta_n_sel, self._beta_deff_sel),
            ]
        elif family == "binomial":
            self._extra_params_pane.objects = [
                pn.pane.Markdown(
                    "**Binomial-specific parameter** — kolom jumlah trial (nᵢ) per observasi. "
                    "**Wajib diisi.**",
                    margin=(4, 0, 6, 0),
                ),
                self._binomial_trials_sel,
            ]
        else:
            self._extra_params_pane.objects = []
 
    def _build_formula_str(self) -> str:
        response   = self._response_sel.value   or "y"
        predictors = list(self._predictors_sel.value or [])
        group      = self._group_sel.value
 
        rhs = predictors if predictors else ["1"]
        if group:
            rhs.append(f"(1|{group})")
 
        return f"{response} ~ {' + '.join(rhs)}"
 
    def _render_formula_html(self) -> str:
        formula = self._build_formula_str()
        return (
            f'<div style="background:#f4f4f4;border-left:4px solid #0072B2;'
            f'padding:10px 16px;border-radius:6px;font-family:monospace;font-size:1.05em">'
            f'<b>Formula:</b>&nbsp; {formula}</div>'
        )
 
    def _update_formula_preview(self, *_) -> None:
        self._formula_preview.object = self._render_formula_html()
 
    def _validate_build(self) -> list[str]:
        errors: list[str] = []
 
        if not self._sampler_applied:
            errors.append(
                "Sampler configuration belum di-apply. "
                "Klik <b>✔ Apply Sampler Configuration</b> terlebih dahulu."
            )
        if self.state.data is None:
            errors.append(
                "Data belum dikonfirmasi. Buka tab <b>Data Upload</b> dan klik "
                "<b>▶ Confirm & Use Data</b>."
            )
            return errors   
 
        if not self._response_sel.value:
            errors.append("Response variable belum dipilih.")
        if not self._predictors_sel.value:
            errors.append("Minimal satu predictor variable harus dipilih.")
 
        family = self._family_sel.value
        if family == "binomial" and not self._binomial_trials_sel.value:
            errors.append(
                "Family <b>binomial</b> memerlukan kolom <code>trials</code>. "
                "Pilih kolom yang sesuai."
            )
        if family == "beta":
            has_n    = bool(self._beta_n_sel.value)
            has_deff = bool(self._beta_deff_sel.value)
            if has_n != has_deff:
                errors.append(
                    "Family <b>beta</b>: isi <b>keduanya</b> "
                    "(<code>n</code> dan <code>deff</code>) atau kosongkan keduanya."
                )
        return errors
 
    def _render_model_summary_html(self) -> str:
        response   = self._response_sel.value or "—"
        predictors = list(self._predictors_sel.value or [])
        group      = self._group_sel.value or "*(none)*"
        family     = self._family_sel.value
        link       = self._link_sel.value
        formula    = self._build_formula_str()
        missing    = self._handle_missing_sel.value
        total      = (self._draws.value or 1000) * (self._chains.value or 4)
 
        pred_html = " ".join(
            f'<span style="background:#0072B2;color:white;padding:3px 10px;'
            f'border-radius:14px;font-size:0.88em;margin:2px">{p}</span>'
            for p in predictors
        ) or "<i>(none)</i>"
 
        extra = ""
        if family == "binomial":
            trials = self._binomial_trials_sel.value or "—"
            extra  = f"<tr><td><b>Trials column</b></td><td><code>{trials}</code></td></tr>"
        elif family == "beta":
            n    = self._beta_n_sel.value    or "*(auto)*"
            deff = self._beta_deff_sel.value or "*(auto)*"
            extra = (
                f"<tr><td><b>n column</b></td><td><code>{n}</code></td></tr>"
                f"<tr><td><b>deff column</b></td><td><code>{deff}</code></td></tr>"
            )
 
        def row(label, value, shade=False):
            bg = ' style="background:#e8f0fa"' if shade else ""
            return f"<tr{bg}><td style='padding:6px 12px;width:38%'><b>{label}</b></td><td style='padding:6px 12px'>{value}</td></tr>"
 
        return f"""
<div style="border:1px solid #d0d0d0;border-radius:10px;padding:16px;
            margin-top:8px;background:#fafafa;font-size:0.95em">
  <div style="font-size:1.1em;font-weight:bold;margin-bottom:10px;color:#0072B2">
    ✔ Model Configuration Summary
  </div>
  <table style="border-collapse:collapse;width:100%">
    {row("Formula",   f"<code>{formula}</code>",                       shade=True)}
    {row("Response (y)",  f"<code>{response}</code>")}
    {row("Predictors (x)", pred_html,                                  shade=True)}
    {row("Group / Area",  f"<code>{group}</code>")}
    {row("Family",        f"<code>{family}</code>",                    shade=True)}
    {row("Link Function", f"<code>{link}</code>")}
    {extra}
    {row("Handle Missing", f"<code>{missing}</code>",                  shade=True)}
    {row("Draws × Chains",
         f'<span style="background:#0072B2;color:white;padding:2px 10px;'
         f'border-radius:12px;font-weight:bold">{total:,}</span>')}
  </table>
</div>"""
 
    def _on_build(self, event) -> None:
        errors = self._validate_build()
        if errors:
            self._build_status.object  = _error_box(
                "⚠ Cannot build model", "<br>".join(f"• {e}" for e in errors)
            )
            self._model_summary.object = ""
            return

        self._build_status.object  = _success_box(
            "✔ Model built successfully. "
            "Proceed to <b>Prior Checking</b> or <b>Fit Model</b>."
        )
        self._model_summary.object = self._render_model_summary_html()
 
    def panel(self) -> pn.Column:
        overview = pn.pane.Markdown("""
**This section allows you to specify the variables and model settings used for \
hierarchical Bayesian modeling.**
- **Response Variable:** The outcome variable being modeled.
- **Auxiliary Variables:** Explanatory (independent) variables — fixed effects.
- **Group Variables:** Grouping variable (e.g., area, cluster) for random effects.
- **HB Family and Link Function:** Bayesian hierarchical family and corresponding \
link function (e.g., log, logit).
""")

        configuresample_card = pn.Card(
            pn.Column(
                pn.Row(self._draws,         self._tune),
                pn.Row(self._chains,        self._cores),
                pn.Row(self._target_accept, self._random_seed),
                self._progressbar,
                pn.layout.Divider(),
                self._total_draws_badge,
                pn.layout.Divider(),
                pn.Row(self._sampler_apply_btn),
                self._sampler_status,
            ),
            title="Configure Sampler",
            margin=10,
        )

        createmodel_card = pn.Card(
            pn.Column(
                pn.pane.Markdown("Select Variables", margin=(4, 0, 4, 0)),
                pn.Row(self._response_sel, self._group_sel),
                self._predictors_sel,
                pn.layout.Divider(),

                pn.pane.Markdown(
                    "Distribution Family & Link Function",
                    margin=(4, 0, 4, 0),
                ),
                pn.Row(self._family_sel, self._link_sel),
                self._family_desc,
                self._extra_params_pane,  
                pn.layout.Divider(),
 
                pn.pane.Markdown("Additional Options", margin=(4, 0, 4, 0)),
                self._handle_missing_sel,
                pn.layout.Divider(),
 
                pn.pane.Markdown("Formula Preview", margin=(4, 0, 4, 0)),
                self._formula_preview,
                pn.layout.Divider(),
 
                # Build
                pn.Row(self._build_btn),
                self._build_status,
                self._model_summary,
            ),
            title="Model Building",
            margin=10,
        )

        prior_card = pn.Card(
            pn.Column(),
            title="Prior Checking",
            margin=10,
        )
        fitmodel_card = pn.Card(
            pn.Column(),
            title="Fit Model",
            margin=10,
        )
 
        return pn.Column(
            pn.Card(overview, title="Overview", margin=10),
            configuresample_card,
            createmodel_card,
            prior_card,
            fitmodel_card,
        )
 
 
# =============================================================================
# HTML helper 
# =============================================================================
 
def _error_box(title: str, body: str) -> str:
    return (
        f'<div style="background:#d62728;color:white;padding:10px 16px;'
        f'border-radius:8px;margin-top:8px"><b>{title}:</b><br>{body}</div>'
    )
 
def _success_box(body: str) -> str:
    return (
        f'<div style="background:#2ca02c;color:white;padding:10px 16px;'
        f'border-radius:8px;margin-top:8px">{body}</div>'
    )
 
# =============================================================================
# ResultsTab 
# =============================================================================

class ResultsTab(param.Parameterized):
    state: AppState = param.Parameter()

    def panel(self) -> pn.Column:
        modelsummary_card = pn.Card(
            pn.Column(
            ),
            title="Model Summary",
            margin=10,
        )

        convergenceevaluation_card = pn.Card(
            pn.Column(
            ),
            title="Convergence Evaluation",
            margin=10,
        )

        saeestimation_card = pn.Card(
            pn.Column(
            ),
            title="SAE Estimation",
            margin=10,
        )

        saeestimationresults_card = pn.Card(
            pn.Column(
            ),
            title="SAE Estimation Results",
            margin=10,
        )

        return pn.Column(
            modelsummary_card,
            convergenceevaluation_card,
            saeestimation_card,
            saeestimationresults_card,
        )


# =============================================================================
# App
# =============================================================================

class App:
    """HBSAEMP Panel dashboard."""

    def __init__(self):
        self._state       = AppState()
        self._data_tab    = DataTab(state=self._state)
        self._explore_tab = ExploreTab(state=self._state)
        self._model_tab   = ModelTab(state=self._state)
        self._results_tab = ResultsTab(state=self._state)

    def build(self) -> pn.template.FastListTemplate:
        tabs = pn.Tabs(
            ("Data Upload",      self._data_tab.panel()),
            ("Data Exploration", self._explore_tab.panel()),
            ("Modeling",         self._model_tab.panel()),
            ("Results",          self._results_tab.panel()),
            sizing_mode="stretch_width",
        )
        return pn.template.FastListTemplate(
            title="HBSAEMP APP",
            main=[tabs],
            accent="#A01346",
        )

    def serve(self, port: int = 5006, show: bool = True) -> None:
        pn.serve(self.build(), port=port, show=show)


# =============================================================================
# Entry point
# =============================================================================
_app = App()
_app.build().servable()

if __name__ == "__main__":
    _app.serve()