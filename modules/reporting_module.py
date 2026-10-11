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

    # -------------------------------------------------------------------- #
    # CNN deep-learning analysis report (matplotlib pages)                 #
    # -------------------------------------------------------------------- #
    def generate_cnn_report(self, analysis, out_path, cnn_report=None,
                            class_colors=None, title="WPM - CNN Deep-Learning Analysis"):
        """Build a PDF report for a CNN prediction run: KPIs, VA/NVA donut,
        activity mix, prediction timeline, confusion matrix + event log."""
        out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
        class_colors = class_colors or {}
        def ccolor(a):
            return class_colors.get(str(a).lower()) or class_colors.get(str(a).upper()) or "#94a3b8"

        events = analysis.get("events", [])
        total_sec = int(analysis.get("total", 0)) or sum(e.get("duration", 0) for e in events)
        vanva = analysis.get("va_nva") or {}
        if not vanva:
            va = sum(e["duration"] for e in events
                     if str(e.get("activity", "")).lower() in {"write"})
            vanva = {"total": total_sec,
                     "VA": {"sec": va, "pct": round(100*va/total_sec, 2) if total_sec else 0},
                     "NVA": {"sec": total_sec - va,
                             "pct": round(100*(total_sec-va)/total_sec, 2) if total_sec else 0}}
        dist = {}
        for e in events:
            a = str(e.get("activity", "")).lower()
            if not a:
                continue
            d = dist.setdefault(a, {"count": 0, "sec": 0})
            d["count"] += 1
            d["sec"] += e.get("duration", 0)
        acc = (cnn_report or {}).get("test_accuracy")

        with PdfPages(out_path) as pdf:
            # ---- page 1 : title + key stats + donut + VA/NVA ---- #
            fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
            fig.text(0.06, 0.93, title, fontsize=19, fontweight="bold", color="#0f766e")
            fig.text(0.06, 0.895, "ResNet-18 frame classifier · write / crew / check / wait",
                     fontsize=12, color="#334155")
            fig.text(0.06, 0.855, f"{analysis.get('session','')}   |   "
                                  f"generated {analysis.get('generated_at','')}   |   "
                                  f"source: {analysis.get('source','—')}",
                     fontsize=9, color="#64748b")
            facts = [f"Frames / seconds predicted: {total_sec:,}",
                     f"Duty events detected: {len(events):,}",
                     f"VA {vanva['VA']['pct']}%  ·  NVA {vanva['NVA']['pct']}%",
                     f"Model test accuracy: {round(acc*100,1) if acc else 'n/a'}%"]
            for i, f in enumerate(facts):
                fig.text(0.06, 0.81 - i*0.040, f"- {f}", fontsize=12, color="#0f766e",
                         fontweight="bold")
            dist_ax = fig.add_axes([0.10, 0.34, 0.42, 0.34])
            if dist:
                self._donut(dist_ax, dist, [ccolor(a) for a in dist])
            vax = fig.add_axes([0.60, 0.36, 0.34, 0.26])
            vlabels = [k for k in ("VA", "NVA") if k in vanva]
            vvals = [vanva[k]["sec"] for k in vlabels]
            bars = vax.bar(vlabels, vvals, color=["#16a34a", "#ef4444"])
            vax.set_title("Value-added vs Non-value-added (s)", fontsize=9)
            for b, v in zip(bars, vvals):
                vax.text(b.get_x() + b.get_width()/2, v + max(vvals)*0.02,
                         f"{v:,}", ha="center", fontsize=8)
            vax.spines[["top", "right"]].set_visible(False)
            pdf.savefig(fig); plt.close(fig)

            # ---- page 2 : prediction timeline ---- #
            bands = []
            for e in events[:120]:
                bands.append({"activity": str(e["activity"]).lower(),
                              "start": e["start"], "end": e["end"]})
            fig, axes = plt.subplots(figsize=(8.27, 5.0), facecolor="white")
            if bands:
                y = 1.0
                for b in bands[:80]:
                    axes.barh(y, b["end"] - b["start"] + 1, left=b["start"],
                              height=0.8, color=ccolor(b["activity"]));
                    y -= 1
                axes.set_xlim(0, max(total_sec, 1))
                axes.set_ylim(0, min(80, len(bands)) + 1)
                axes.set_yticks([])
                axes.set_title(f"Prediction timeline — first {min(80,len(bands))} events (seconds)",
                               fontsize=11, fontweight="bold", loc="left")
                axes.spines[["top", "right", "left"]].set_visible(False)
                handles = [plt.Rectangle((0, 0), 1, 1, color=ccolor(a)) for a in dist]
                axes.legend(handles, list(dist.keys()), loc="lower right", fontsize=8)
            else:
                axes.text(0.5, 0.5, "no predicted events", ha="center", fontsize=12)
            pdf.savefig(fig); plt.close(fig)

            # ---- page 3 : confusion matrix + per-class report ---- #
            if cnn_report and cnn_report.get("confusion_matrix"):
                fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
                cm = np.asarray(cnn_report["confusion_matrix"])
                classes = cnn_report.get("classes", [])
                ax = fig.add_axes([0.12, 0.52, 0.42, 0.34])
                im = ax.imshow(cm, cmap="Purples")
                ax.set_xticks(range(cm.shape[1])); ax.set_yticks(range(cm.shape[0]))
                ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
                ax.set_yticklabels(classes, fontsize=8)
                ax.set_xlabel("predicted"); ax.set_ylabel("true")
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, f"{cm[i,j]}", ha="center", va="center", fontsize=9)
                fig.colorbar(im, ax=ax, fraction=0.046)
                ax.set_title("CNN confusion matrix (hold-out test)",
                             fontsize=10, fontweight="bold")
                pax = fig.add_axes([0.12, 0.08, 0.76, 0.34])
                pax.axis("off")
                pcl = cnn_report.get("per_class", {})
                rows = [[c, pcl.get(c, {}).get("test_support", 0),
                         pcl.get(c, {}).get("precision", 0),
                         pcl.get(c, {}).get("recall", 0)] for c in classes]
                tab = pax.table(cellText=rows, colLabels=["class", "test #", "precision", "recall"],
                                loc="upper center", cellLoc="center")
                tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(1, 1.6)
                for (r, c), cell in tab.get_celld().items():
                    if r == 0:
                        cell.set_facecolor("#0f766e"); cell.set_text_props(color="white", fontweight="bold")
                pdf.savefig(fig); plt.close(fig)

            # ---- page 4 : event log extract ---- #
            fig, ax = plt.subplots(figsize=(8.27, 11.69), facecolor="white")
            ax.axis("off")
            ax.set_title("Detected duty events (extract)", fontsize=14, fontweight="bold", loc="left")
            rows = [[e["start"], e["end"], e["duration"], e["activity"],
                     e.get("va_nva", "VA" if str(e["activity"]).lower() == "write" else "NVA")]
                    for e in events[:45]]
            cols = ["start", "end", "duration", "activity", "VA_NVA"]
            tab = ax.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="center")
            tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(1.0, 1.4)
            for (r, c), cell in tab.get_celld().items():
                if r == 0:
                    cell.set_facecolor("#0f766e"); cell.set_text_props(color="white", fontweight="bold")
            pdf.savefig(fig); plt.close(fig)
        return str(out_path)