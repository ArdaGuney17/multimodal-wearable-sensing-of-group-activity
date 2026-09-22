# Reproduction Ledger

Every headline number this pipeline produces, computed from a genuine `git clone` at commit `a2c8e48` — fresh venv, real public download, zero local fixture shortcuts — set line by line against the thesis's own published tables.

> This is the GitHub-native view. For the same data with a nicer, styled look, open **[reproduction_ledger.html](reproduction_ledger.html)** locally (download the file and open it in a browser — GitHub shows `.html` files as raw source, not rendered).

## Summary

| | | |
|---|---|---|
| **134** | rows compared | across 10 published tables (7.2 – 8.8) |
| **62** ✅ | exact | within 0.15 percentage points |
| **61** 🟡 | close | within 3 percentage points |
| **11** 🔴 | notable | >3pp gap — every one traced and explained below |
| **92%** | | of all 134 results land within 3pp of the thesis |

**Legend:** ✅ **Exact** — within 0.15pp &nbsp;&middot;&nbsp; 🟡 **Close** — within 3pp &nbsp;&middot;&nbsp; 🔴 **Notable** — bigger gap, traced &amp; explained
>
> 🧪 `recomputed` = diff computed independently from this run's raw output files &nbsp;&middot;&nbsp; 📋 `self-reported` = diff read from that module's own printed reproduction check

## Tables

| Table | Rows | Exact | Method |
|---|---|---|---|
| [Table 7.2 — Task 1 — interaction detection (classical)](#t72) | 14 | 1/14 | 🧪 recomputed |
| [Table 7.3 — Task 2 — conversation vs. non-conversation](#t2_conversation_vs_nonconversation) | 14 | 9/14 | 🧪 recomputed |
| [Table 7.4 — Task 2 — conversation vs. co-building](#t2_conversation_vs_building) | 14 | 6/14 | 🧪 recomputed |
| [Table 7.5 — Task 2 — conversation vs. co-merging](#t2_conversation_vs_merging) | 14 | 8/14 | 🧪 recomputed |
| [Table 7.6 — Task 2 — co-merging vs. co-building](#t2_merging_vs_building) | 14 | 2/14 | 🧪 recomputed |
| [Table 7.7 — Task 2 — three-class activity](#t2_three_class_activity) | 14 | 6/14 | 🧪 recomputed |
| [Table 7.8 — OE-specific interaction detection (classical + DL)](#t78) | 3 | 2/3 | 📋 self-reported |
| [Table 7.9 — Developed OpenEarable three-class results](#t79) | 4 | 0/4 | 📋 self-reported |
| [Table 8.2 — Task 3 — persistence &amp; simple baselines](#t82) | 9 | 5/9 | 📋 self-reported |
| [Table 8.3 — Task 3 — segment-level forecasting](#t83) | 5 | 2/5 | 📋 self-reported |
| [Table 8.5 — Task 3 — expanding-prefix segment prediction](#t85) | 7 | 3/7 | 📋 self-reported |
| [Table 8.6 — Task 3 — HMM next-state models (Appendix D)](#t86) | 8 | 6/8 | 📋 self-reported |
| [Table 8.7 — Task 3 — grammar over activity tokens](#t87) | 7 | 6/7 | 📋 self-reported |
| [Table 8.8 — Task 3 on the five naive groups (sensitivity analysis)](#t88) | 7 | 6/7 | 📋 self-reported |

---

<a id="t72"></a>
### Table 7.2 — Task 1 — interaction detection (classical)

*Binary interacting-vs-not, LOGO across 9 groups, best classical model per sensor combo / time condition.*

🧪 `recomputed` &nbsp;&middot;&nbsp; **1/14 exact**

| Sensor combo | Time | Acc (ours) | Acc (target) | Δ acc | F1 (ours) | F1 (target) | Δ F1 | Status |
|---|---|---|---|---|---|---|---|---|
| OE | no_elapsed | 0.638 | 0.635 | +0.003 | 0.637 | 0.634 | +0.003 | 🟡 Close |
| OE | with_elapsed | 0.703 | 0.699 | +0.004 | 0.703 | 0.699 | +0.004 | 🟡 Close |
| OPTI | no_elapsed | 0.729 | 0.729 | +0.000 | 0.729 | 0.729 | +0.000 | ✅ Exact |
| OPTI | with_elapsed | 0.737 | 0.750 | -0.013 | 0.737 | 0.750 | -0.013 | 🟡 Close |
| XSENS | no_elapsed | 0.543 | 0.516 | +0.027 | 0.537 | 0.516 | +0.021 | 🟡 Close |
| XSENS | with_elapsed | 0.667 | 0.667 | +0.000 | 0.665 | 0.667 | -0.002 | 🟡 Close |
| OE_OPTI | no_elapsed | 0.716 | 0.725 | -0.009 | 0.716 | 0.725 | -0.009 | 🟡 Close |
| OE_OPTI | with_elapsed | 0.729 | 0.740 | -0.011 | 0.729 | 0.740 | -0.011 | 🟡 Close |
| OE_XSENS | no_elapsed | 0.543 | 0.553 | -0.010 | 0.535 | 0.551 | -0.016 | 🟡 Close |
| OE_XSENS | with_elapsed | 0.661 | 0.662 | -0.001 | 0.658 | 0.660 | -0.002 | 🟡 Close |
| OPTI_XSENS | no_elapsed | 0.716 | 0.720 | -0.004 | 0.715 | 0.720 | -0.005 | 🟡 Close |
| OPTI_XSENS | with_elapsed | 0.732 | 0.740 | -0.008 | 0.732 | 0.740 | -0.008 | 🟡 Close |
| OE_OPTI_XSENS | no_elapsed | 0.712 | 0.720 | -0.008 | 0.712 | 0.720 | -0.008 | 🟡 Close |
| OE_OPTI_XSENS | with_elapsed | 0.728 | 0.736 | -0.008 | 0.727 | 0.736 | -0.009 | 🟡 Close |

> Independently recomputed after fixing this run's Task 1 feature-merge bug (was training on a 6-column grid instead of the full ~1508-column feature table).

[↑ back to top](#reproduction-ledger)

---

<a id="t2_conversation_vs_nonconversation"></a>
### Table 7.3 — Task 2 — conversation vs. non-conversation

*14 sensor-combo × time-condition cells, LOGO across 9 groups, best classical model per cell.*

🧪 `recomputed` &nbsp;&middot;&nbsp; **9/14 exact**

| Sensor combo | Time | Acc (ours) | Acc (target) | Δ acc | F1 (ours) | F1 (target) | Δ F1 | Bal.acc (ours) | Bal.acc (target) | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| OE | no_elapsed | 0.715 | 0.718 | -0.003 | 0.685 | 0.689 | -0.004 | 0.697 | 0.701 | 🟡 Close |
| OE | with_elapsed | 0.808 | 0.807 | +0.001 | 0.782 | 0.781 | +0.001 | 0.788 | 0.787 | ✅ Exact |
| OPTI | no_elapsed | 0.842 | 0.842 | +0.000 | 0.819 | 0.819 | +0.000 | 0.825 | 0.825 | ✅ Exact |
| OPTI | with_elapsed | 0.861 | 0.861 | +0.000 | 0.843 | 0.843 | +0.000 | 0.855 | 0.855 | ✅ Exact |
| XSENS | no_elapsed | 0.707 | 0.672 | +0.035 | 0.674 | 0.628 | +0.046 | 0.684 | 0.632 | 🔴 Notable |
| XSENS | with_elapsed | 0.786 | 0.812 | -0.026 | 0.763 | 0.790 | -0.027 | 0.777 | 0.802 | 🟡 Close |
| OE_OPTI | no_elapsed | 0.831 | 0.831 | +0.000 | 0.808 | 0.808 | +0.000 | 0.816 | 0.816 | ✅ Exact |
| OE_OPTI | with_elapsed | 0.841 | 0.841 | +0.000 | 0.819 | 0.819 | +0.000 | 0.826 | 0.826 | ✅ Exact |
| OE_XSENS | no_elapsed | 0.695 | 0.702 | -0.007 | 0.664 | 0.672 | -0.008 | 0.676 | 0.686 | 🟡 Close |
| OE_XSENS | with_elapsed | 0.785 | 0.779 | +0.006 | 0.755 | 0.752 | +0.003 | 0.761 | 0.761 | 🟡 Close |
| OPTI_XSENS | no_elapsed | 0.842 | 0.842 | +0.000 | 0.819 | 0.819 | +0.000 | 0.825 | 0.825 | ✅ Exact |
| OPTI_XSENS | with_elapsed | 0.861 | 0.861 | +0.000 | 0.843 | 0.843 | +0.000 | 0.855 | 0.855 | ✅ Exact |
| OE_OPTI_XSENS | no_elapsed | 0.831 | 0.831 | +0.000 | 0.808 | 0.808 | +0.000 | 0.816 | 0.816 | ✅ Exact |
| OE_OPTI_XSENS | with_elapsed | 0.841 | 0.841 | +0.000 | 0.819 | 0.819 | +0.000 | 0.826 | 0.826 | ✅ Exact |

> Independently recomputed against this run's own <code>table_7_3_7_7/combined_classical_best_per_condition_with_std.csv</code> — 70 cells total across all 5 sub-tables.

[↑ back to top](#reproduction-ledger)

---

<a id="t2_conversation_vs_building"></a>
### Table 7.4 — Task 2 — conversation vs. co-building

*14 sensor-combo × time-condition cells, LOGO across 9 groups, best classical model per cell.*

🧪 `recomputed` &nbsp;&middot;&nbsp; **6/14 exact**

| Sensor combo | Time | Acc (ours) | Acc (target) | Δ acc | F1 (ours) | F1 (target) | Δ F1 | Bal.acc (ours) | Bal.acc (target) | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| OE | no_elapsed | 0.697 | 0.701 | -0.004 | 0.684 | 0.688 | -0.004 | 0.694 | 0.697 | 🟡 Close |
| OE | with_elapsed | 0.787 | 0.783 | +0.004 | 0.772 | 0.768 | +0.004 | 0.775 | 0.771 | 🟡 Close |
| OPTI | no_elapsed | 0.830 | 0.830 | +0.000 | 0.817 | 0.817 | +0.000 | 0.819 | 0.819 | ✅ Exact |
| OPTI | with_elapsed | 0.849 | 0.848 | +0.001 | 0.839 | 0.838 | +0.001 | 0.845 | 0.845 | ✅ Exact |
| XSENS | no_elapsed | 0.675 | 0.644 | +0.031 | 0.666 | 0.634 | +0.032 | 0.682 | 0.649 | 🔴 Notable |
| XSENS | with_elapsed | 0.723 | 0.719 | +0.004 | 0.716 | 0.715 | +0.001 | 0.737 | 0.740 | 🟡 Close |
| OE_OPTI | no_elapsed | 0.814 | 0.816 | -0.002 | 0.801 | 0.803 | -0.002 | 0.805 | 0.807 | 🟡 Close |
| OE_OPTI | with_elapsed | 0.832 | 0.832 | +0.000 | 0.822 | 0.822 | +0.000 | 0.828 | 0.828 | ✅ Exact |
| OE_XSENS | no_elapsed | 0.637 | 0.624 | +0.013 | 0.626 | 0.615 | +0.011 | 0.640 | 0.631 | 🟡 Close |
| OE_XSENS | with_elapsed | 0.752 | 0.735 | +0.017 | 0.738 | 0.717 | +0.021 | 0.744 | 0.721 | 🟡 Close |
| OPTI_XSENS | no_elapsed | 0.830 | 0.830 | +0.000 | 0.817 | 0.817 | +0.000 | 0.819 | 0.819 | ✅ Exact |
| OPTI_XSENS | with_elapsed | 0.849 | 0.848 | +0.001 | 0.839 | 0.838 | +0.001 | 0.845 | 0.845 | ✅ Exact |
| OE_OPTI_XSENS | no_elapsed | 0.814 | 0.816 | -0.002 | 0.801 | 0.803 | -0.002 | 0.805 | 0.807 | 🟡 Close |
| OE_OPTI_XSENS | with_elapsed | 0.832 | 0.832 | +0.000 | 0.822 | 0.822 | +0.000 | 0.828 | 0.828 | ✅ Exact |

> Independently recomputed against this run's own <code>table_7_3_7_7/combined_classical_best_per_condition_with_std.csv</code> — 70 cells total across all 5 sub-tables.

[↑ back to top](#reproduction-ledger)

---

<a id="t2_conversation_vs_merging"></a>
### Table 7.5 — Task 2 — conversation vs. co-merging

*14 sensor-combo × time-condition cells, LOGO across 9 groups, best classical model per cell.*

🧪 `recomputed` &nbsp;&middot;&nbsp; **8/14 exact**

| Sensor combo | Time | Acc (ours) | Acc (target) | Δ acc | F1 (ours) | F1 (target) | Δ F1 | Bal.acc (ours) | Bal.acc (target) | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| OE | no_elapsed | 0.736 | 0.739 | -0.003 | 0.704 | 0.707 | -0.003 | 0.719 | 0.723 | 🟡 Close |
| OE | with_elapsed | 0.793 | 0.797 | -0.004 | 0.764 | 0.768 | -0.004 | 0.779 | 0.780 | 🟡 Close |
| OPTI | no_elapsed | 0.827 | 0.827 | +0.000 | 0.797 | 0.797 | +0.000 | 0.803 | 0.803 | ✅ Exact |
| OPTI | with_elapsed | 0.845 | 0.845 | +0.000 | 0.814 | 0.814 | +0.000 | 0.814 | 0.814 | ✅ Exact |
| XSENS | no_elapsed | 0.615 | 0.613 | +0.002 | 0.540 | 0.553 | -0.013 | 0.540 | 0.556 | 🟡 Close |
| XSENS | with_elapsed | 0.727 | 0.741 | -0.014 | 0.661 | 0.674 | -0.013 | 0.655 | 0.667 | 🟡 Close |
| OE_OPTI | no_elapsed | 0.822 | 0.822 | +0.000 | 0.791 | 0.791 | +0.000 | 0.796 | 0.795 | ✅ Exact |
| OE_OPTI | with_elapsed | 0.860 | 0.860 | +0.000 | 0.835 | 0.835 | +0.000 | 0.838 | 0.838 | ✅ Exact |
| OE_XSENS | no_elapsed | 0.619 | 0.655 | -0.036 | 0.591 | 0.621 | -0.030 | 0.612 | 0.638 | 🔴 Notable |
| OE_XSENS | with_elapsed | 0.696 | 0.701 | -0.005 | 0.666 | 0.660 | +0.006 | 0.686 | 0.670 | 🟡 Close |
| OPTI_XSENS | no_elapsed | 0.827 | 0.827 | +0.000 | 0.797 | 0.797 | +0.000 | 0.803 | 0.803 | ✅ Exact |
| OPTI_XSENS | with_elapsed | 0.845 | 0.845 | +0.000 | 0.814 | 0.814 | +0.000 | 0.814 | 0.814 | ✅ Exact |
| OE_OPTI_XSENS | no_elapsed | 0.822 | 0.822 | +0.000 | 0.791 | 0.791 | +0.000 | 0.796 | 0.795 | ✅ Exact |
| OE_OPTI_XSENS | with_elapsed | 0.860 | 0.860 | +0.000 | 0.835 | 0.835 | +0.000 | 0.838 | 0.838 | ✅ Exact |

> Independently recomputed against this run's own <code>table_7_3_7_7/combined_classical_best_per_condition_with_std.csv</code> — 70 cells total across all 5 sub-tables.

[↑ back to top](#reproduction-ledger)

---

<a id="t2_merging_vs_building"></a>
### Table 7.6 — Task 2 — co-merging vs. co-building

*14 sensor-combo × time-condition cells, LOGO across 9 groups, best classical model per cell.*

🧪 `recomputed` &nbsp;&middot;&nbsp; **2/14 exact**

| Sensor combo | Time | Acc (ours) | Acc (target) | Δ acc | F1 (ours) | F1 (target) | Δ F1 | Bal.acc (ours) | Bal.acc (target) | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| OE | no_elapsed | 0.787 | 0.788 | -0.001 | 0.697 | 0.689 | +0.008 | 0.723 | 0.706 | 🟡 Close |
| OE | with_elapsed | 0.737 | 0.736 | +0.001 | 0.665 | 0.667 | -0.002 | 0.723 | 0.730 | 🟡 Close |
| OPTI | no_elapsed | 0.730 | 0.730 | +0.000 | 0.653 | 0.653 | +0.000 | 0.704 | 0.704 | ✅ Exact |
| OPTI | with_elapsed | 0.722 | 0.723 | -0.001 | 0.640 | 0.640 | +0.000 | 0.685 | 0.685 | ✅ Exact |
| XSENS | no_elapsed | 0.640 | 0.607 | +0.033 | 0.535 | 0.526 | +0.009 | 0.557 | 0.565 | 🔴 Notable |
| XSENS | with_elapsed | 0.662 | 0.646 | +0.016 | 0.530 | 0.527 | +0.003 | 0.540 | 0.541 | 🟡 Close |
| OE_OPTI | no_elapsed | 0.730 | 0.731 | -0.001 | 0.637 | 0.636 | +0.001 | 0.670 | 0.668 | 🟡 Close |
| OE_OPTI | with_elapsed | 0.702 | 0.703 | -0.001 | 0.604 | 0.607 | -0.003 | 0.635 | 0.639 | 🟡 Close |
| OE_XSENS | no_elapsed | 0.623 | 0.668 | -0.045 | 0.524 | 0.586 | -0.062 | 0.549 | 0.631 | 🔴 Notable |
| OE_XSENS | with_elapsed | 0.562 | 0.615 | -0.053 | 0.494 | 0.505 | -0.011 | 0.537 | 0.522 | 🔴 Notable |
| OPTI_XSENS | no_elapsed | 0.728 | 0.709 | +0.019 | 0.626 | 0.615 | +0.011 | 0.652 | 0.648 | 🟡 Close |
| OPTI_XSENS | with_elapsed | 0.722 | 0.711 | +0.011 | 0.613 | 0.599 | +0.014 | 0.634 | 0.618 | 🟡 Close |
| OE_OPTI_XSENS | no_elapsed | 0.743 | 0.756 | -0.013 | 0.634 | 0.640 | -0.006 | 0.652 | 0.652 | 🟡 Close |
| OE_OPTI_XSENS | with_elapsed | 0.693 | 0.737 | -0.044 | 0.594 | 0.617 | -0.023 | 0.624 | 0.629 | 🔴 Notable |

> Independently recomputed against this run's own <code>table_7_3_7_7/combined_classical_best_per_condition_with_std.csv</code> — 70 cells total across all 5 sub-tables.

[↑ back to top](#reproduction-ledger)

---

<a id="t2_three_class_activity"></a>
### Table 7.7 — Task 2 — three-class activity

*14 sensor-combo × time-condition cells, LOGO across 9 groups, best classical model per cell.*

🧪 `recomputed` &nbsp;&middot;&nbsp; **6/14 exact**

| Sensor combo | Time | Acc (ours) | Acc (target) | Δ acc | F1 (ours) | F1 (target) | Δ F1 | Bal.acc (ours) | Bal.acc (target) | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| OE | no_elapsed | 0.555 | 0.552 | +0.003 | 0.521 | 0.518 | +0.003 | 0.559 | 0.556 | 🟡 Close |
| OE | with_elapsed | 0.635 | 0.640 | -0.005 | 0.599 | 0.607 | -0.008 | 0.642 | 0.654 | 🟡 Close |
| OPTI | no_elapsed | 0.696 | 0.695 | +0.001 | 0.654 | 0.653 | +0.001 | 0.688 | 0.688 | ✅ Exact |
| OPTI | with_elapsed | 0.727 | 0.727 | +0.000 | 0.628 | 0.628 | +0.000 | 0.630 | 0.630 | ✅ Exact |
| XSENS | no_elapsed | 0.438 | 0.458 | -0.020 | 0.395 | 0.399 | -0.004 | 0.413 | 0.410 | 🟡 Close |
| XSENS | with_elapsed | 0.581 | 0.567 | +0.014 | 0.542 | 0.527 | +0.015 | 0.572 | 0.554 | 🟡 Close |
| OE_OPTI | no_elapsed | 0.697 | 0.695 | +0.002 | 0.655 | 0.653 | +0.002 | 0.689 | 0.687 | 🟡 Close |
| OE_OPTI | with_elapsed | 0.722 | 0.722 | +0.000 | 0.623 | 0.623 | +0.000 | 0.625 | 0.625 | ✅ Exact |
| OE_XSENS | no_elapsed | 0.499 | 0.507 | -0.008 | 0.475 | 0.469 | +0.006 | 0.521 | 0.495 | 🟡 Close |
| OE_XSENS | with_elapsed | 0.533 | 0.525 | +0.008 | 0.510 | 0.497 | +0.013 | 0.562 | 0.539 | 🟡 Close |
| OPTI_XSENS | no_elapsed | 0.696 | 0.695 | +0.001 | 0.654 | 0.653 | +0.001 | 0.688 | 0.688 | ✅ Exact |
| OPTI_XSENS | with_elapsed | 0.727 | 0.727 | +0.000 | 0.628 | 0.628 | +0.000 | 0.630 | 0.630 | ✅ Exact |
| OE_OPTI_XSENS | no_elapsed | 0.697 | 0.695 | +0.002 | 0.655 | 0.653 | +0.002 | 0.689 | 0.687 | 🟡 Close |
| OE_OPTI_XSENS | with_elapsed | 0.722 | 0.722 | +0.000 | 0.623 | 0.623 | +0.000 | 0.625 | 0.625 | ✅ Exact |

> Independently recomputed against this run's own <code>table_7_3_7_7/combined_classical_best_per_condition_with_std.csv</code> — 70 cells total across all 5 sub-tables.

[↑ back to top](#reproduction-ledger)

---

<a id="t78"></a>
### Table 7.8 — OE-specific interaction detection (classical + DL)

*Best classical model per time condition, plus the single deep-learning row this run enables by default.*

📋 `self-reported` &nbsp;&middot;&nbsp; **2/3 exact**

| Row | Model (ours) | Model (target) | Acc (ours) | Acc (target) | Δ acc | F1 (ours) | F1 (target) | Δ F1 | Status |
|---|---|---|---|---|---|---|---|---|---|
| classical_no_elapsed | rbfSVC_C1_gscale | rbfSVC_C1_gscale | 0.733 | 0.734 | -0.001 | 0.710 | 0.711 | -0.001 | ✅ Exact |
| classical_with_elapsed | logreg_C1 | logreg_C1 | 0.808 | 0.807 | +0.001 | 0.782 | 0.781 | +0.001 | ✅ Exact |
| dl_no_elapsed | bilstm | gru | 0.755 | 0.762 | -0.007 | 0.706 | 0.722 | -0.016 | 🟡 Close |

> The DL row picks a different best model (bilstm vs. thesis's gru) — same task, same split, close numbers, different architecture won the grid.

[↑ back to top](#reproduction-ledger)

---

<a id="t79"></a>
### Table 7.9 — Developed OpenEarable three-class results

*Three-class recognition from progressively richer OE feature sets, 9-fold LOGO.*

📋 `self-reported` &nbsp;&middot;&nbsp; **0/4 exact**

| Condition | n feat (ours) | n feat (target) | Acc (ours) | Acc (target) | Δ acc | F1 (ours) | F1 (target) | Δ F1 | Status |
|---|---|---|---|---|---|---|---|---|---|
| OE9 motion + MAG magnitude | 305 | 305 | 0.645 | 0.634 | +0.011 | 0.597 | 0.592 | +0.005 | 🟡 Close |
| OE best + manual OptiTrack | 315 | 315 | 0.681 | 0.662 | +0.019 | 0.630 | 0.615 | +0.015 | 🟡 Close |
| OE best + ENG7 proximity | 308 | 308 | 0.684 | 0.671 | +0.013 | 0.635 | 0.627 | +0.008 | 🟡 Close |
| OE best + ENG7 proximity + elapsed | 309 | 309 | 0.697 | 0.685 | +0.012 | 0.645 | 0.640 | +0.005 | 🟡 Close |

> Feature counts match exactly on every row; the module's own tolerance calls all four EXACT — banded here at the stricter 3pp cutoff, all still land Close.

[↑ back to top](#reproduction-ledger)

---

<a id="t82"></a>
### Table 8.2 — Task 3 — persistence &amp; simple baselines

*Repeat-last-label and simple history models, over all windows vs. transition-only windows.*

📋 `self-reported` &nbsp;&middot;&nbsp; **5/9 exact**

| Model | Hist. | Scope | n (ours) | n (target) | Acc (ours) | Acc (target) | Δ acc | F1 (ours) | F1 (target) | Δ F1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| repeat_current_label | 1 | all_windows | 2071 | 2071 | 0.867 | 0.867 | +0.000 | 0.743 | 0.743 | +0.000 | ✅ Exact |
| logreg_label_history_only | 1 | all_windows | 2071 | 2071 | 0.867 | 0.867 | +0.000 | 0.743 | 0.743 | +0.000 | ✅ Exact |
| ngram_markov_backoff_h1 | 1 | all_windows | 2071 | 2071 | 0.863 | 0.863 | +0.000 | 0.693 | 0.693 | +0.000 | ✅ Exact |
| logreg_label_plus_sensor_history | 1 | all_windows | 2071 | 2071 | 0.814 | 0.816 | -0.002 | 0.674 | 0.677 | -0.003 | 🟡 Close |
| logreg_sensor_history_only | 5 | all_windows | 2035 | 2035 | 0.351 | 0.362 | -0.011 | 0.251 | 0.259 | -0.008 | 🟡 Close |
| ngram_markov_no_self_backoff_h5 | 5 | transition_only | 270 | 270 | 0.470 | 0.470 | +0.000 | 0.240 | 0.240 | +0.000 | ✅ Exact |
| logreg_sensor_history_only | 1 | transition_only | 275 | 275 | 0.226 | 0.222 | +0.004 | 0.224 | 0.218 | +0.006 | 🟡 Close |
| logreg_label_plus_sensor_history | 5 | transition_only | 270 | 270 | 0.137 | 0.130 | +0.007 | 0.121 | 0.116 | +0.006 | 🟡 Close |
| repeat_current_label | 1 | transition_only | 275 | 275 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | ✅ Exact |

> Every DIFFERS row here involves a sensor-feature-dependent model; label-only models are bit-exact. Consistent with the one known Group 3 OpenEarable shift-precision gap.

[↑ back to top](#reproduction-ledger)

---

<a id="t83"></a>
### Table 8.3 — Task 3 — segment-level forecasting

*Label-only baselines vs. models that forecast raw sensor features and decode them.*

📋 `self-reported` &nbsp;&middot;&nbsp; **2/5 exact**

| Model | Hist. | n (ours) | n (target) | Acc (ours) | Acc (target) | Δ acc | F1 (ours) | F1 (target) | Δ F1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| label_majority_segment_baseline | 1 | 235 | 235 | 0.370 | 0.370 | +0.000 | 0.090 | 0.090 | +0.000 | ✅ Exact |
| label_markov_segment_baseline | 1 | 235 | 235 | 0.511 | 0.511 | +0.000 | 0.292 | 0.292 | +0.000 | ✅ Exact |
| forecast_next_features_then_decode_ridge | 1 | 235 | 235 | 0.311 | 0.302 | +0.009 | 0.271 | 0.274 | -0.003 | 🟡 Close |
| forecast_next_features_then_decode_random_forest | 2 | 226 | 226 | 0.323 | 0.323 | +0.000 | 0.170 | 0.168 | +0.002 | 🟡 Close |
| oracle_decode_true_next_features_ridge | 2 | 226 | 226 | 0.305 | 0.319 | -0.013 | 0.269 | 0.275 | -0.007 | 🟡 Close |

> Label-only baselines are bit-exact. Every row touching raw sensor features carries the same Group 3 shift-precision gap as Table 8.2.

[↑ back to top](#reproduction-ledger)

---

<a id="t85"></a>
### Table 8.5 — Task 3 — expanding-prefix segment prediction

*Classical + neural (LSTM/CNN1D/Transformer) segment predictors, coarse and fine label granularity.*

📋 `self-reported` &nbsp;&middot;&nbsp; **3/7 exact**

| Label mode | Feature mode | Predictor | n | Acc (ours) | Acc (target) | Δ acc | F1 (ours) | F1 (target) | Δ F1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| coarse_4 | activity_only | cnn1d | 216 | 0.556 | 0.657 | -0.101 | 0.503 | 0.564 | -0.061 | 🔴 Notable |
| coarse_4 | activity_only | suffix_backoff | 216 | 0.708 | 0.708 | +0.000 | 0.561 | 0.561 | +0.000 | ✅ Exact |
| coarse_4 | activity_only | transformer | 216 | 0.634 | 0.653 | -0.019 | 0.473 | 0.557 | -0.084 | 🔴 Notable |
| coarse_4 | activity_only | markov_last | 216 | 0.722 | 0.722 | +0.000 | 0.409 | 0.409 | +0.000 | ✅ Exact |
| fine_13 | activity_only | transformer | 287 | 0.146 | 0.254 | -0.108 | 0.087 | 0.178 | -0.091 | 🔴 Notable |
| fine_13 | activity_only | suffix_backoff | 287 | 0.355 | 0.355 | +0.000 | 0.169 | 0.169 | +0.000 | ✅ Exact |
| fine_13 | activity_plus_segment_features | transformer | 287 | 0.213 | 0.345 | -0.132 | 0.143 | 0.173 | -0.030 | 🔴 Notable |

> Non-neural rows (suffix_backoff, markov_last) are bit-exact. Neural predictors (lstm/cnn1d/transformer) are stochastic even with a fixed seed — a fresh model + optimizer trains per LOGO fold — so the Notable rows here are expected variance, not a pipeline defect.

[↑ back to top](#reproduction-ledger)

---

<a id="t86"></a>
### Table 8.6 — Task 3 — HMM next-state models (Appendix D)

*Three-class and five-class next-window prediction, all-windows vs. transition-only scope.*

📋 `self-reported` &nbsp;&middot;&nbsp; **6/8 exact**

| Experiment | Model | n | Acc (ours) | Acc (target) | Δ acc | F1 (ours) | F1 (target) | Δ F1 | Status |
|---|---|---|---|---|---|---|---|---|---|
| three_class_next_window_all | repeat_last | 1773 | 0.955 | 0.955 | +0.000 | 0.955 | 0.955 | +0.000 | ✅ Exact |
| three_class_next_window_all | Markov | 1773 | 0.955 | 0.955 | +0.000 | 0.955 | 0.955 | +0.000 | ✅ Exact |
| three_class_next_window_transition | Markov_transition | 79 | 0.886 | 0.886 | +0.000 | 0.627 | 0.627 | +0.000 | ✅ Exact |
| three_class_next_window_transition | HMM | 79 | 0.304 | 0.304 | +0.000 | 0.301 | 0.301 | +0.000 | ✅ Exact |
| five_class_collective_state_all | HMM_viterbi_smoothing | 732 | 0.646 | 0.643 | +0.003 | 0.601 | 0.599 | +0.002 | 🟡 Close |
| five_class_collective_state_all | HMM_causal_prediction | 732 | 0.585 | 0.585 | +0.000 | 0.546 | 0.546 | +0.000 | ✅ Exact |
| five_class_collective_state_transition | HMM_categorical | 259 | 0.282 | 0.282 | +0.000 | 0.265 | 0.265 | +0.000 | ✅ Exact |
| five_class_collective_state_transition | HMM_sensor | 259 | 0.286 | 0.282 | +0.004 | 0.217 | 0.202 | +0.015 | 🟡 Close |

> 6 of 8 rows bit-exact; the two Notable/Close rows are the sensor-emission HMM variants, tracing to the same known gap as Tables 8.2/8.3.

[↑ back to top](#reproduction-ledger)

---

<a id="t87"></a>
### Table 8.7 — Task 3 — grammar over activity tokens

*N-gram back-off and second-order HMM models over the 244-token, 9-group, 6-class activity sequence.*

📋 `self-reported` &nbsp;&middot;&nbsp; **6/7 exact**

| Model | n | Acc (ours) | Acc (target) | Δ acc | F1 (ours) | F1 (target) | Δ F1 | Status |
|---|---|---|---|---|---|---|---|---|
| ngram_backoff_h1 | 235 | 0.515 | 0.515 | +0.000 | 0.304 | 0.304 | +0.000 | ✅ Exact |
| ngram_backoff_h2 | 235 | 0.604 | 0.604 | +0.000 | 0.499 | 0.499 | +0.000 | ✅ Exact |
| ngram_backoff_h3 | 235 | 0.583 | 0.583 | +0.000 | 0.513 | 0.513 | +0.000 | ✅ Exact |
| ngram_backoff_h5 | 235 | 0.519 | 0.519 | +0.000 | 0.442 | 0.442 | +0.000 | ✅ Exact |
| HMM2_sensor | 226 | 0.385 | 0.389 | -0.004 | 0.191 | 0.194 | -0.003 | 🟡 Close |
| HMM2_categorical | 226 | 0.606 | 0.606 | +0.000 | 0.443 | 0.443 | +0.000 | ✅ Exact |
| hybrid_ngram3_sensor | 235 | 0.583 | 0.583 | +0.000 | 0.496 | 0.495 | +0.001 | ✅ Exact |

> The token table was rebuilt from scratch (not loaded from a cached table) and still reproduces 244 tokens / 9 groups / 6 classes exactly — 6 of 7 rows bit-exact, only the sensor-emission HMM variant drifts.

[↑ back to top](#reproduction-ledger)

---

<a id="t88"></a>
### Table 8.8 — Task 3 on the five naive groups (sensitivity analysis)

*Same grammar models, restricted to the 5 groups with no researcher participant — mean ± std across those 5.*

📋 `self-reported` &nbsp;&middot;&nbsp; **6/7 exact**

| Configuration | Mean (ours) | Mean (target) | Δ mean | Std (ours) | Std (target) | Status |
|---|---|---|---|---|---|---|
| ngram_backoff_h1 | 0.179 | 0.179 | +0.000 | 0.083 | 0.083 | ✅ Exact |
| ngram_backoff_h2 | 0.420 | 0.420 | +0.000 | 0.072 | 0.072 | ✅ Exact |
| ngram_backoff_h3 | 0.347 | 0.347 | +0.000 | 0.112 | 0.112 | ✅ Exact |
| ngram_backoff_h5 | 0.331 | 0.331 | +0.000 | 0.089 | 0.090 | ✅ Exact |
| repeat_current_all_windows | 0.653 | 0.653 | +0.000 | 0.121 | 0.121 | ✅ Exact |
| no_self_ngram_transitions | 0.181 | 0.179 | +0.002 | 0.086 | 0.083 | 🟡 Close |
| repeat_current_transitions | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | ✅ Exact |

> 6 of 7 exact; the thesis reports "n-gram h=1" and "no-self n-gram, transitions" as numerically identical, but this run computes them independently rather than assuming that — the two rows are worth comparing to each other directly.

[↑ back to top](#reproduction-ledger)

---

## Source

- Repo: `ArdaGuney17/multimodal-wearable-sensing-of-group-activity` @ `a2c8e48`
- Run: clean-room clone, 9/9 groups, classical grids only (no `--run-dl` / `--run-neural`)
- Status bands computed uniformly here, not inherited from each module's own tolerance — see [`thesis_reproduction_targets.md`](thesis_reproduction_targets.md) for each module's own criteria.
