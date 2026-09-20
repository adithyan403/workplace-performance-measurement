"""WPM Module 5: REPORTING -- dashboards, analytics charts and PDF reports
(paper's REPORTING module; KPIs delivered to stakeholders).

Builds:
  * metrics/dashboard JSON consumed by the Flask front-end (Chart.js),
  * an automatically-assembled PDF report (matplotlib pages).
"""
import io, json, os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    import pandas as pd
    _PD = True
except Exception:
    _PD = False


class ReportingModule:
    def __init__(self, cfg, proc_module, root="."):
        self.cfg = cfg
        self.proc = proc_module
        self.root = Path(root)

    # -------------------------------------------------------------------- #
    # JSON payloads for the web dashboard                                  #
    # -------------------------------------------------------------------- #
    def dashboard_payload(self, summary, mongo_events=None, title="KSRTC Duty Logs"):
        payload = {
            "title": title,
            "process": self.cfg["process_name"],
            "activities": [a["label"] for a in self.cfg["activities"]],
            "activity_colors": [a["color"] for a in self.cfg["activities"]],
            "activity_ids": [a["id"] for a in self.cfg["activities"]],
            "summary": summary,
        }
        if mongo_events is not None:
            payload["mongo_events"] = mongo_events
        return payload

    def zone_heatmap(self, cell_mean, rows=None, cols=None):
        rows, cols = rows or self.cfg["grid"]["rows"], cols or self.cfg["grid"]["cols"]
        mat = np.asarray(cell_mean, dtype=float).reshape(rows, cols)
        return [[round(float(v), 2) for v in row] for row in mat]

    # -------------------------------------------------------------------- #
    # matplotlib figure builders (kept simple; front-end uses Chart.js)    #
    # -------------------------------------------------------------------- #
    @staticmethod
    def activity_donut_fig(dist, colors, title="Activity distribution"):
        labels, vals = list(dist.keys()), [d["sec"] for d in dist.values()]
        fig, ax = plt.subplots(figsize=(5, 4.4), facecolor="white")
        wedges, _, autotexts = ax.pie(vals, labels=None, autopct="%1.1f%%",
                                      colors=colors, startangle=90,
                                      wedgeprops=dict(width=0.42, edgecolor="white"))
        for t in autotexts:
            t.set_fontsize(8); t.set_color("white")
        ax.legend(wedges, [f"{l} ({v}s)" for l, v in zip(labels, vals)],
                  loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
        ax.text(0, 0, title, ha="center", va="center", fontsize=9, fontweight="bold")
        ax.set(aspect="equal")
        return fig

    @staticmethod
    def va_nva_fig(va_nva, title="Value-added vs Non-value-added"):
        fig, ax = plt.subplots(figsize=(5, 4.4), facecolor="white")
        colors = ["#16a34a", "#ef4444"]
        labels = [f"{k} ({v['pct']}%)" for k, v in va_nva.items()]
        vals = [va_nva[k]["sec"] for k in va_nva]
        ax.bar(labels, vals, color=colors, width=0.55)
        ax.set_ylabel("seconds"); ax.set_title(title, fontsize=11, fontweight="bold")
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals)*0.01, f"{v:,}s", ha="center", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        return fig

    # -------------------------------------------------------------------- #
    # PDF report generation                                                #
    # -------------------------------------------------------------------- #
    def generate_pdf(self, summary, event_samples, out_path, va_nva=None,
                     extra_figs=None):
        out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
        dist = summary["activity_distribution"]
        colors = [self.proc.color(a) for a in dist.keys()]
        acolors = [a["color"] for a in self.cfg["activities"] if a["id"] in dist]
        with PdfPages(out_path) as pdf:
            # ---- page 1 : title + overview ---- #
            fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
            fig.text(0.06, 0.93, "WPM - Workplace Performance Measurement", fontsize=20, fontweight="bold", color="#0f766e")
            fig.text(0.06, 0.885, "KSRTC Station-Master Waybill Entry - Duty Logs Report", fontsize=13, color="#334155")
            fig.text(0.06, 0.845, f"Event count: {summary['event_count']}   |   "
                                  f"Observed: {summary.get('total_sec',0)} s   |   "
                                  f"Observations: {self.cfg['process_name']}", fontsize=10, color="#475569")
            dist_ax = fig.add_axes([0.12, 0.42, 0.36, 0.34])
            self._donut(dist_ax, dist, colors)
            vanva = va_nva or summary.get("va_nva")
            vax = fig.add_axes([0.58, 0.42, 0.36, 0.30])
            labels = list(vanva.keys()); vals = [vanva[k]["sec"] for k in labels]
            bars = vax.bar(labels, vals, color=["#16a34a", "#ef4444"])
            vax.set_title("Value-added vs Non-value-added (seconds)", fontsize=9)
            for b, v in zip(bars, vals):
                vax.text(b.get_x() + b.get_width()/2, v + max(vals)*0.02, f"{v:,}s", ha="center", fontsize=8)
            facts = [f"VA {vanva['VA']['pct']}% of duty time",
                     f"NVA {vanva['NVA']['pct']}%",
                     f"{summary['event_count']} activity intervals",
                     f"Primary writing zone: cell {self.cfg['zones']['primary']}"]
            fig.text(0.1, 0.28, "Key facts", fontsize=11, fontweight="bold", color="#0f766e")
            for i, f in enumerate(facts):
                fig.text(0.1, 0.24 - i*0.04, f"- {f}", fontsize=10, color="#334155")
            pdf.savefig(fig); plt.close(fig)

            # ---- page 2 : event log extract ---- #
            fig, ax = plt.subplots(figsize=(8.27, 11.69), facecolor="white")
            ax.axis("off")
            ax.set_title("Activity event log (extract)", fontsize=14, fontweight="bold", loc="left")
            cols = ["start", "end", "duration", "activity", "VA_NVA"]
            tab = ax.table(cellText=event_samples, colLabels=cols, loc="upper center", cellLoc="center")
            tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(1.0, 1.4)
            for (r, c), cell in tab.get_celld().items():
                if r == 0:
                    cell.set_facecolor("#0f766e"); cell.set_text_props(color="white", fontweight="bold")
            pdf.savefig(fig); plt.close(fig)

            # ---- page 3+ : extra figures ---- #
            for f in (extra_figs or []):
                pdf.savefig(f); plt.close(f)
        return str(out_path)

    @staticmethod
    def _donut(ax, dist, colors):
        labels = list(dist.keys()); vals = [dist[l]["sec"] for l in labels]
        wedges, texts, autotexts = ax.pie(vals, labels=labels, colors=colors,
                                          autopct="%1.0f%%", startangle=90,
                                          wedgeprops=dict(width=0.35, edgecolor="white"))
        for t in autotexts:
            t.set_fontsize(7)
        ax.set_title("Activity distribution (seconds)", fontsize=9)

    # -------------------------------------------------------------------- #
    # CSV / data export                                                    #
    # -------------------------------------------------------------------- #
    def to_csv(self, rows, path, fieldnames=None):
        import csv
        fieldnames = fieldnames or (list(rows[0].keys()) if rows else [])
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
        return path