# Scorecard: tag `rho10`

Each claim is evaluated on the worst of its phantoms (seed mean of per-seed medians / shares). Arms that pass are listed in full; for failed claims the three best arms are shown.


## interface fixed  —  threshold <= 10.0 deg, phantoms P3-40-t0.25, P3-40-t0.5, P3-15-t0.25

| arm | value | threshold | result |
|---|---|---|---|
| anatomix48 excise_all rho10 | 16.157 | 10.000 | FAIL |
| anatomix48 excise_cl rho10 | 23.758 | 10.000 | FAIL |
| anatomix48 excise_tca_dom rho10 | 56.192 | 10.000 | FAIL |

0 of 12 arms pass.


## nothing lost in clear tissue (> 2 mm from the step)  —  threshold <= ST + 1 1.0 deg, phantoms P3-40, P3-40-t0.5, P3-40-t0.25, P1, P1-t0.5

| arm | value | threshold | result |
|---|---|---|---|
| intensity eigen rho10 | 10.147 | 11.147 | PASS |
| anatomix48 excise_cl rho10 | 10.148 | 11.147 | PASS |
| anatomix48 excise_all rho10 | 10.149 | 11.147 | PASS |
| anatomix48 excise_tca_dom rho10 | 10.150 | 11.147 | PASS |
| anatomix48 switch_tca_dom rho10 | 10.164 | 11.147 | PASS |
| anatomix48 switch_cl rho10 | 10.206 | 11.147 | PASS |
| anatomix48 switch_gm rho10 | 10.249 | 11.147 | PASS |
| anatomix48 blend_tca_dom rho10 | 10.354 | 11.147 | PASS |
| anatomix48 blend_cl rho10 | 11.036 | 11.147 | PASS |

9 of 12 arms pass.


## U-fibres kept  —  threshold <= 10.0 deg, phantoms P2, P2-t0.5

| arm | value | threshold | result |
|---|---|---|---|
| anatomix48 blend_cl rho10 | 1.808 | 10.000 | PASS |
| intensity eigen rho10 | 1.813 | 10.000 | PASS |
| anatomix48 blend_tca_dom rho10 | 1.816 | 10.000 | PASS |
| anatomix48 switch_tca_dom rho10 | 1.827 | 10.000 | PASS |
| anatomix48 excise_tca_dom rho10 | 1.846 | 10.000 | PASS |
| anatomix48 switch_cl rho10 | 1.880 | 10.000 | PASS |
| anatomix48 excise_cl rho10 | 2.073 | 10.000 | PASS |
| anatomix48 excise_all rho10 | 2.221 | 10.000 | PASS |
| anatomix48 blend_gm rho10 | 2.241 | 10.000 | PASS |
| anatomix48 tca rho10 | 2.440 | 10.000 | PASS |
| anatomix48 switch_gm rho10 | 2.560 | 10.000 | PASS |
| anatomix48 eigen rho10 | 2.795 | 10.000 | PASS |

12 of 12 arms pass.


## vessels  —  threshold <= 15.0 deg, phantoms P6, P6-t0.5

| arm | value | threshold | result |
|---|---|---|---|
| anatomix48 excise_all rho10 | 6.274 | 15.000 | PASS |
| anatomix48 switch_gm rho10 | 7.610 | 15.000 | PASS |
| anatomix48 blend_gm rho10 | 7.698 | 15.000 | PASS |
| anatomix48 tca rho10 | 7.719 | 15.000 | PASS |
| anatomix48 blend_cl rho10 | 14.190 | 15.000 | PASS |

5 of 12 arms pass.


## crossings resolved (both within 15 deg at >= 90% of sites, 60 deg)  —  threshold >= 0.9 share, phantoms P4-lines-c60-f1

| arm | value | threshold | result |
|---|---|---|---|
| anatomix48 csd rho10 | 0.642 | 0.900 | FAIL |
| anatomix48 csd_refined rho10 | 0.172 | 0.900 | FAIL |
| intensity csd rho10 | 0.167 | 0.900 | FAIL |

0 of 8 arms pass.


## crossings resolved (45 deg)  —  threshold >= 0.9 share, phantoms P4-lines-c45-f1

| arm | value | threshold | result |
|---|---|---|---|
| anatomix48 csd rho10 | 0.288 | 0.900 | FAIL |
| anatomix48 overcomplete rho10 | 0.040 | 0.900 | FAIL |
| anatomix48 csd_refined rho10 | 0.028 | 0.900 | FAIL |

0 of 8 arms pass.


## no false populations (single)  —  threshold <= 0.02 share, phantoms P4-lines-single, P4-lines-single-t0.5

| arm | value | threshold | result |
|---|---|---|---|
| anatomix48 bic rho10 | 0.000 | 0.020 | PASS |
| anatomix48 frt rho10 | 0.001 | 0.020 | PASS |
| intensity frt rho10 | 0.002 | 0.020 | PASS |

3 of 8 arms pass.


## fractions (4:1)  —  threshold <= 0.1 abs, phantoms P4-lines-c60-f4

| arm | value | threshold | result |
|---|---|---|---|
| anatomix48 csd rho10 | 0.061 | 0.100 | PASS |
| intensity csd rho10 | 0.064 | 0.100 | PASS |

2 of 6 arms pass.


## boundary detector (PR-AUC)  —  threshold >= 0.9 auc, phantoms P3-40-t0.25, P3-40-t0.5, P3-40

| arm | value | threshold | result |
|---|---|---|---|
| intensity gm rho10 | 0.616 | 0.900 | FAIL |
| intensity trace rho10 | 0.201 | 0.900 | FAIL |
| anatomix48 tca_dom rho10 | 0.103 | 0.900 | FAIL |

0 of 11 arms pass.
