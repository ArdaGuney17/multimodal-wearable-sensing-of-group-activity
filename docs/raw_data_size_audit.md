# Raw Data Size Audit — thesis/data/

Metadata-only recursive enumeration (no file downloads or content reads) of the RAW per-group sensor data under `thesis/data/` in Google Drive, performed as a sizing exercise ahead of a data-hosting decision.

- Root folder: `data` (id `1a9vaiEdgkG-FBquFY3ySzfyWUoL1eMWH`)
- Groups covered: 1, 2, 3, 5, 6, 7, 8, 9, 10 (no group_4)
- Scope: RAW sensor files only inside `optitrack/`, `xsens/`, `openearable/` (incl. Participant1/2/3 subfolders), `elan/`. Derived/processed subfolders (`model_ready/`, `*_labeled/`, `*_final/`, `optitrack_cleaned_manual_stitched/`, `optitrack_inspection/`) and `.ipynb_checkpoints/` clutter were identified but NOT enumerated (their contents are excluded from all totals below).

## Summary Totals

### Grand Total

| Metric | Value |
|---|---|
| Total raw files | 315 |
| Total bytes | 2,653,205,886 B |
| Total size | 2.4710 GiB (binary) |
| Total size | 2.6532 GB (decimal) |
| Total size | 2530.3 MiB |

### By Sensor Type (across all 9 groups)

| Sensor | Files | Bytes | GiB |
|---|---:|---:|---:|
| optitrack | 38 | 1,197,096,771 | 1.1149 |
| xsens | 27 | 405,061,903 | 0.3772 |
| openearable | 216 | 1,050,099,305 | 0.9780 |
| elan | 34 | 947,907 | 0.0009 |

### By Group

| Group | Bytes | GiB |
|---|---:|---:|
| group_1 | 261,711,212 | 0.2437 |
| group_2 | 274,892,645 | 0.2560 |
| group_3 | 294,605,349 | 0.2744 |
| group_5 | 419,380,732 | 0.3906 |
| group_6 | 254,821,705 | 0.2373 |
| group_7 | 307,629,688 | 0.2865 |
| group_8 | 313,324,422 | 0.2918 |
| group_9 | 381,907,766 | 0.3557 |
| group_10 | 144,932,367 | 0.1350 |

## Flags / Anomalies

- **group_10** has an `optitrack_labeled` folder duplicated at the group root level (id `1g_JKEhl7DkSE1mPA0Kw0s2vkZbTAdwrO`, sibling of `optitrack/`, `xsens/`, etc.) in addition to the normal `optitrack/optitrack_labeled` subfolder present in every other group. This is structurally inconsistent with groups 1-9 and worth checking — it was not enumerated (derived data) but its existence at the wrong nesting level suggests a misplaced upload.
- **group_2** optitrack has five derived/processed subfolders (`model_ready`, `optitrack_labeled`, `optitrack_final`, `optitrack_cleaned_manual_stitched`, `optitrack_inspection`) versus the standard three (`model_ready`, `optitrack_labeled`, `optitrack_final`) seen in all other groups — extra manual-cleanup artifacts, none enumerated/counted.
- **Take-count varies a lot across groups**: group_6 and group_10 optitrack have only 3 takes, group_1 has 2, most groups have 4-5, and group_9 has 6 takes. This is expected variability in recording sessions but is flagged since raw optitrack size per group swings roughly 3x (144.9 MB group_10 vs 419.4 MB group_5) partly as a result.
- **group_9 elan** contains two BACKUP_before_sync_patch duplicate files (`Group_9_individual_build_renamed_BACKUP_before_sync_patch.csv` and `Group_9_with_individual_build_BACKUP_before_sync_patch.csv`) that are byte-identical in size to their non-backup counterparts — likely redundant copies inflating the elan total for this group only.
- **group_9 Participant3 (openearable)** files are dramatically smaller (226 KB – 807 KB per stream) than every other participant/group (typically 1.5–12 MB per stream) — roughly 10-20x smaller. This looks like a short/partial recording session and should be verified before being treated as equivalent data.
- **elan is negligible in size** (under 1 MB across all 9 groups, ~948 KB total) — it consists of small labeling/build CSVs, not raw sensor streams, and does not materially affect hosting-size decisions.
- All 9 groups have all four sensor types present (optitrack, xsens, openearable, elan) — no group is missing a sensor type entirely.
- All 27 openearable Participant subfolders (9 groups x 3 participants) contain exactly 8 raw CSV streams each (acc, baro, bone_acc, env_temp, gyro, mgnt, ppg, skin_temp) — consistent coverage, no missing streams.

## Per-File Detail

### group_1

**Total: 261,711,212 bytes (0.2437 GiB)**

#### optitrack (2 files, 129,053,758 bytes)

| Title | File ID | Size (bytes) |
|---|---|---:|
| Arda_Thesis_Group-1_Take_2.csv | 1cSbZQ84bnrnVagvpbw8kJLC-_43ZbEGd | 32,417,324 |
| Arda_Thesis_Group-1_Take_1.csv | 1Xb8_rJBeeFhVOw2X5ppST3KbItpZ_Um6 | 96,636,434 |

#### xsens (3 files, 35,966,832 bytes)

| Title | File ID | Size (bytes) |
|---|---|---:|
| Participant3.csv | 18vkk-Mtff4WWbC0FO81Q0XrvvulV42m1 | 11,875,853 |
| Participant2.csv | 1V1uKE6aPr2z1Q84d8c-vA_xazj9MYB-k | 12,077,045 |
| Participant1.csv | 1lhmoJDR2A2kN55u6fqRizA0d0slDzHwb | 12,013,934 |

#### openearable (24 files, 96,661,915 bytes)

**Participant1** (8 files, 32,695,702 bytes)

| Title | Size (bytes) |
|---|---:|
| Participant1_skin_temp.csv | 1,764,084 |
| Participant1_ppg.csv | 6,186,078 |
| Participant1_mgnt.csv | 4,990,197 |
| Participant1_gyro.csv | 4,245,966 |
| Participant1_env_temp.csv | 2,797,536 |
| Participant1_bone_acc.csv | 5,227,352 |
| Participant1_baro.csv | 3,197,184 |
| Participant1_acc.csv | 4,287,305 |

**Participant2** (8 files, 31,658,061 bytes)

| Title | Size (bytes) |
|---|---:|
| Participant2_skin_temp.csv | 1,693,300 |
| Participant2_ppg.csv | 6,038,191 |
| Participant2_mgnt.csv | 4,739,819 |
| Participant2_gyro.csv | 4,105,032 |
| Participant2_env_temp.csv | 2,712,500 |
| Participant2_bone_acc.csv | 5,109,483 |
| Participant2_baro.csv | 3,100,000 |
| Participant2_acc.csv | 4,159,736 |

**Participant3** (8 files, 32,308,152 bytes)

| Title | Size (bytes) |
|---|---:|
| Participant3_skin_temp.csv | 1,740,928 |
| Participant3_ppg.csv | 6,176,942 |
| Participant3_mgnt.csv | 4,768,813 |
| Participant3_gyro.csv | 4,193,556 |
| Participant3_env_temp.csv | 2,767,940 |
| Participant3_bone_acc.csv | 5,347,986 |
| Participant3_baro.csv | 3,163,360 |
| Participant3_acc.csv | 4,148,627 |

#### elan (1 files, 28,707 bytes)

| Title | File ID | Size (bytes) |
|---|---|---:|
| Group_1_individual_build_renamed.csv | 13MAG91BpmudRVojdpWvEYaKYBH_Ewo4t | 28,707 |

### group_2

**Total: 274,892,645 bytes (0.2560 GiB)**

#### optitrack (5 files, 95,611,110 bytes)

| Title | File ID | Size (bytes) |
|---|---|---:|
| Arda_Group_2_Take_3.csv | 1WhLIuu5Edbj4xafA_OqkokYMb0l8MfIB | 20,843,343 |
| Arda_Group_2_Take_2.csv | 1Rb7U_Ob1Wol7zfle9ETw7pw_eVExdu0K | 21,163,462 |
| Arda_Group_2_Take_1.csv | 1PpwHoh3yVn5roPzETtoGpI3RrmuZoIBz | 23,184,408 |
| Arda_Group_2_Take_5.csv | 1BzlEVY9AQfpXZbPk6TWiU6TTWrSzJehe | 10,882,835 |
| Arda_Group_2_Take_4.csv | 1CpGs_flc7deq7Z8vExyeFygff-lT2op0 | 19,537,062 |

#### xsens (3 files, 49,915,566 bytes)

| Title | File ID | Size (bytes) |
|---|---|---:|
| Participant1.csv | 1fAKHjAR6dm9YcGx_okDODIVmCnbZu7Tg | 16,635,988 |
| Participant3.csv | 1wJ7OZHumbEuydzP-XkXmV9RvIucyxzhM | 16,672,217 |
| Participant2.csv | 1L8VgeLCv43h4rVCG6zl0I_8Whu9tx_T9 | 16,607,361 |

#### openearable (24 files, 129,236,738 bytes)

**Participant1** (8 files, 46,493,133 bytes)

| Title | Size (bytes) |
|---|---:|
| Participant1_skin_temp.csv | 2,508,912 |
| Participant1_ppg.csv | 8,773,141 |
| Participant1_mgnt.csv | 7,100,333 |
| Participant1_gyro.csv | 6,040,390 |
| Participant1_env_temp.csv | 3,975,692 |
| Participant1_bone_acc.csv | 7,449,352 |
| Participant1_baro.csv | 4,543,648 |
| Participant1_acc.csv | 6,101,665 |

**Participant2** (8 files, 43,020,080 bytes)

| Title | Size (bytes) |
|---|---:|
| Participant2_skin_temp.csv | 2,300,872 |
| Participant2_ppg.csv | 8,257,650 |
| Participant2_mgnt.csv | 6,494,776 |
| Participant2_gyro.csv | 5,629,052 |
| Participant2_env_temp.csv | 3,630,116 |
| Participant2_bone_acc.csv | 6,859,115 |
| Participant2_baro.csv | 4,148,704 |
| Participant2_acc.csv | 5,699,795 |

**Participant3** (8 files, 39,723,525 bytes)

| Title | Size (bytes) |
|---|---:|
| Participant3_skin_temp.csv | 2,153,200 |
| Participant3_ppg.csv | 7,458,516 |
| Participant3_mgnt.csv | 6,072,864 |
| Participant3_gyro.csv | 5,256,764 |
| Participant3_env_temp.csv | 3,381,644 |
| Participant3_bone_acc.csv | 6,206,434 |
| Participant3_baro.csv | 3,864,736 |
| Participant3_acc.csv | 5,329,367 |

#### elan (4 files, 129,231 bytes)

| Title | File ID | Size (bytes) |
|---|---|---:|
| Group_2_individual_build_renamed.csv | 1z-8pyH4NDTvSLteyp8_4NqCNw-ZJ8b3J | 40,679 |
| Group_2_with_individual_build.csv | 1sOttvIwkcD9d6DUjeFIy7NC9c126C9Aq | 37,797 |
| Group_2_clean.csv | 1aHo3873JQhguAqftxYOJ25gIODdZXWKM | 27,968 |
| Group_2.csv | 16hI8gffsMMOkYjpTWOsd1uf78ZY6aJCX | 22,787 |

### group_3

**Total: 294,605,349 bytes (0.2744 GiB)**

#### optitrack (5 files, 118,049,865 bytes)

| Title | File ID | Size (bytes) |
|---|---|---:|
| Arda_Group_3_Take_3.csv | 1grW6dEfSs_bRF_3IRcnbN1tiP04_Qe_T | 22,402,450 |
| Arda_Group_3_Take_2.csv | 1CDTsxiRQeJppI1Y7zHwZN0c6gdRV8rNX | 25,011,556 |
| Arda_Group_3_Take_1.csv | 1v_i1k8-ApY0vTPdZGZCiQFSscJnsqmVQ | 26,908,299 |
| Arda_Group_3_Take_5.csv | 1tPhP5OZoodvqMdJAa_TjrvwKhdh-74DQ | 3,712,854 |
| Arda_Group_3_Take_4.csv | 1zCcJf3mXBl8mQJdFW4ONkJ2Jh03yJ0CE | 40,014,706 |

#### xsens (3 files, 49,538,072 bytes)

| Title | File ID | Size (bytes) |
|---|---|---:|
| Participant1.csv | 1Cj9OCt5o3nf6IEK4McaHLQbt5HZ5KtgM | 16,605,441 |
| Participant2.csv | 1if6l06gHK_f6oRw-jPWVSTOO_ftpnh9w | 16,451,542 |
| Participant3.csv | 1tlAYQ3fa2PGmlZmGCVBNGwWoNL7CUJsK | 16,481,089 |

#### openearable (24 files, 126,939,274 bytes)

**Participant1** (8 files, 40,698,785 bytes)

| Title | Size (bytes) |
|---|---:|
| Participant1_skin_temp.csv | 2,202,312 |
| Participant1_ppg.csv | 7,776,739 |
| Participant1_mgnt.csv | 6,069,238 |
| Participant1_gyro.csv | 5,250,643 |
| Participant1_env_temp.csv | 3,482,976 |
| Participant1_bone_acc.csv | 6,612,397 |
| Participant1_baro.csv | 3,980,544 |
| Participant1_acc.csv | 5,323,936 |

**Participant2** (8 files, 41,267,124 bytes)

| Title | Size (bytes) |
|---|---:|
| Participant2_skin_temp.csv | 2,234,120 |
| Participant2_ppg.csv | 7,878,123 |
| Participant2_mgnt.csv | 6,142,078 |
| Participant2_gyro.csv | 5,431,972 |
| Participant2_env_temp.csv | 3,520,524 |
| Participant2_bone_acc.csv | 6,664,200 |
| Participant2_baro.csv | 4,023,456 |
| Participant2_acc.csv | 5,372,651 |

**Participant3** (8 files, 44,973,365 bytes)

| Title | Size (bytes) |
|---|---:|
| Participant3_skin_temp.csv | 2,403,016 |
| Participant3_ppg.csv | 8,612,662 |
| Participant3_mgnt.csv | 6,750,889 |
| Participant3_gyro.csv | 5,847,071 |
| Participant3_env_temp.csv | 3,856,832 |
| Participant3_bone_acc.csv | 7,199,742 |
| Participant3_baro.csv | 4,407,808 |
| Participant3_acc.csv | 5,895,345 |

#### elan (6 files, 78,138 bytes)

| Title | File ID | Size (bytes) |
|---|---|---:|
| Group_3_individual_build_renamed.csv | 1xZstai6m8BnvCvjV56hI3EroWDA8_E4Y | 19,633 |
| Group_3_with_individual_build.csv | 1yUnGLkao9jfKGGPRBsJJpvAZZnZJqFm2 | 18,361 |
| Group_3_concatenated.csv | 1FRXxlqPwCf0wBVIulxN8njX34T66arK9 | 12,320 |
| Group_3_clean.csv | 1bdF0_psOqGwJot_J9qQOjnXMlXT-gR8r | 15,635 |
| Group_3_Part2.csv | 1kZa9UlzIOh5csFmuBCi5Ibx5wLtUJ7zQ | 8,134 |
| Group_3_Part1.csv | 1CtkLwwispIzBTgB-uhzSkMJwJmjqgcDW | 4,055 |

### group_5

**Total: 419,380,732 bytes (0.3906 GiB)**

#### optitrack (5 files, 163,575,325 bytes)

| Title | File ID | Size (bytes) |
|---|---|---:|
| Arda_Group-5_Take_3.csv | 19SvSsXwXgaKsg3vbjhkcLIFRCljkEgTy | 30,979,726 |
| Arda_Group-5_Take_2.csv | 1ne5K91Wx8cirphqe47_PpWtoQ2IHLZwy | 25,626,688 |
| Arda_Group-5_Take_4.csv | 11GQDpzLA1Hy2Bg5AhHCRMTVmOaC_mKhg | 41,516,953 |
| Arda_Group-5_Take_1.csv | 1dZHu1wmdWLHKjrLV8CB7sY97I06CcqLI | 33,995,817 |
| Arda_Group-5_Take_5.csv | 1ECMlngPsmZLOZHKRtML9OgGsgQLfvLIf | 31,456,141 |

#### xsens (3 files, 72,363,440 bytes)

| Title | File ID | Size (bytes) |
|---|---|---:|
| Participant3.csv | 1SEDPwGTk0ARzzOjtiu9iB-swnP4LVLWc | 24,248,850 |
| Participant2.csv | 1ddEjpu7-gdZrLJ6qkKnWI0XPwtuCV_z_ | 24,164,794 |
| Participant1.csv | 17hzc9MIJyFZjJ6n64QxUoAL5T0vNyokd | 23,949,796 |

#### openearable (24 files, 183,299,706 bytes)

**Participant1** (8 files, 55,159,522 bytes)

| Title | Size (bytes) |
|---|---:|
| Participant1_skin_temp.csv | 2,950,108 |
| Participant1_ppg.csv | 10,551,373 |
| Participant1_mgnt.csv | 8,421,287 |
| Participant1_gyro.csv | 7,298,365 |
| Participant1_env_temp.csv | 4,565,232 |
| Participant1_bone_acc.csv | 8,712,345 |
| Participant1_baro.csv | 5,217,408 |
| Participant1_acc.csv | 7,443,404 |

**Participant2** (8 files, 62,171,262 bytes)

| Title | Size (bytes) |
|---|---:|
| Participant2_skin_temp.csv | 3,365,712 |
| Participant2_ppg.csv | 11,802,883 |
| Participant2_mgnt.csv | 9,332,344 |
| Participant2_gyro.csv | 8,070,020 |
| Participant2_env_temp.csv | 5,354,804 |
| Participant2_bone_acc.csv | 9,937,577 |
| Participant2_baro.csv | 6,119,776 |
| Participant2_acc.csv | 8,188,146 |

**Participant3** (8 files, 65,968,922 bytes)

| Title | Size (bytes) |
|---|---:|
| Participant3_skin_temp.csv | 3,539,172 |
| Participant3_ppg.csv | 12,511,458 |
| Participant3_mgnt.csv | 9,968,876 |
| Participant3_gyro.csv | 8,538,222 |
| Participant3_env_temp.csv | 5,618,284 |
| Participant3_bone_acc.csv | 10,946,283 |
| Participant3_baro.csv | 6,420,837 |
| Participant3_acc.csv | 8,425,790 |

#### elan (4 files, 142,261 bytes)

| Title | File ID | Size (bytes) |
|---|---|---:|
| Group_5_individual_build_renamed.csv | 1cMS61Rl1AaUC0KUQubohjRwi0ztgyuAh | 40,731 |
| Group_5_with_individual_build.csv | 1QVcJEM2-InAFkLMsx5s3NFzevybA6ILg | 37,716 |
| Group_5_clean.csv | 1wWAgBWJVKHe08cQfeuzPR3XDSqdlYgHw | 35,154 |
| Group_5.csv | 1IrcbSD_nDCeproMaiMuOl3eVU3PxtGjg | 28,660 |

### group_6

**Total: 254,821,705 bytes (0.2373 GiB)**

#### optitrack (3 files, 121,034,445 bytes)

| Title | File ID | Size (bytes) |
|---|---|---:|
| Arda_Group-6_Take_3.csv | 1_WXcjhP87nHT4Aj5W5UNY9am2aMCvaAa | 5,865,966 |
| Arda_Group-6_Take_2.csv | 1WYI6F39ImwtO6tnsk6TGuJmIYFHbH_Sr | 58,955,838 |
| Arda_Group-6_Take_1.csv | 1oNlBoLPbeBrDtOjNqZ5HWuXlHMhpWBdC | 56,212,641 |

#### xsens (3 files, 35,676,402 bytes)

| Title | File ID | Size (bytes) |
|---|---|---:|
| Participant2.csv | 1sgiib_g-FK6DY0XjIPdx_AXvS10ljBcJ | 11,907,154 |
| Participant3.csv | 1_9DeTRiz46pjD3yMnVlBUdhy5T8FihvK | 11,846,363 |
| Participant1.csv | 1na93-SfEVQdX7wQbzqLgUiVsYFuRc8cd | 11,922,885 |

#### openearable (24 files, 98,044,204 bytes)

**Participant1** (8 files, 33,001,056 bytes)

| Title | Size (bytes) |
|---|---:|
| Participant1_skin_temp.csv | 1,773,072 |
| Participant1_ppg.csv | 6,362,855 |
| Participant1_mgnt.csv | 4,922,446 |
| Participant1_gyro.csv | 4,286,244 |
| Participant1_env_temp.csv | 2,819,432 |
| Participant1_bone_acc.csv | 5,367,954 |
| Participant1_baro.csv | 3,222,208 |
| Participant1_acc.csv | 4,246,845 |

**Participant2** (8 files, 31,826,919 bytes)

| Title | Size (bytes) |
|---|---:|
| Participant2_skin_temp.csv | 1,691,536 |
| Participant2_ppg.csv | 6,192,119 |
| Participant2_mgnt.csv | 4,767,945 |
| Participant2_gyro.csv | 4,143,007 |
| Participant2_env_temp.csv | 2,690,772 |
| Participant2_bone_acc.csv | 5,164,111 |
| Participant2_baro.csv | 3,075,168 |
| Participant2_acc.csv | 4,102,261 |

**Participant3** (8 files, 33,216,229 bytes)

| Title | Size (bytes) |
|---|---:|
| Participant3_skin_temp.csv | 1,788,780 |
| Participant3_ppg.csv | 6,296,196 |
| Participant3_mgnt.csv | 5,029,285 |
| Participant3_gyro.csv | 4,313,850 |
| Participant3_env_temp.csv | 2,836,316 |
| Participant3_bone_acc.csv | 5,434,854 |
| Participant3_baro.csv | 3,241,504 |
| Participant3_acc.csv | 4,275,444 |

#### elan (4 files, 66,654 bytes)

| Title | File ID | Size (bytes) |
|---|---|---:|
| Group_6_individual_build_renamed.csv | 1biywluA08ULJH1Zxlp2GzG-585sJwuIo | 20,511 |
| Group_6_with_individual_build.csv | 1xcBf1LdSFzCVJdserm7ekJ-uNtODaqwv | 18,803 |
| Group_6_clean.csv | 1Qq2GiWOCKcNcO7mcmiaEOTWd82VUUk7o | 15,013 |
| Group_6.csv | 10v5ZROeqVCycxQuGTqAdAjspRlkhYo7J | 12,327 |

### group_7

**Total: 307,629,688 bytes (0.2865 GiB)**

#### optitrack (4 files, 146,514,224 bytes)

| Title | File ID | Size (bytes) |
|---|---|---:|
| Arda_Group_7_Take_2.csv | 1wOJ1bZ8PIfTB5TAjl25t3RAnTO2a9tW8 | 26,635,585 |
| Arda_Group_7_Take_1.csv | 1q_7HjfnpRZ6gIKGr4tFL1E7Xucx8V4D8 | 20,919,357 |
| Arda_Group_7_Take_4.csv | 1AWPYZkNIHYydUbnk1k2F1sFTNzvP2xGm | 71,393,637 |
| Arda_Group_7_Take_3.csv | 1nK7kyg-tjQ4BvYzIdFMQ2WX0lcrrQ6b4 | 27,565,645 |

#### xsens (3 files, 52,151,888 bytes)

| Title | File ID | Size (bytes) |
|---|---|---:|
| Participant3.csv | 1zrcT10mUeIq4Kr2FGnxy-ZgO-IHgx2Nk | 16,650,252 |
| Participant2.csv | 15IsDzf16yRs5kovXESTAwznwzwCiYEPV | 17,709,908 |
| Participant1.csv | 1iMwifUHkacpoypOsVMoFnAwh-yGN4axi | 17,791,728 |

#### openearable (24 files, 108,810,545 bytes)

**Participant1** (8 files, 48,516,043 bytes)

| Title | Size (bytes) |
|---|---:|
| Participant1_skin_temp.csv | 2,586,836 |
| Participant1_ppg.csv | 9,201,878 |
| Participant1_mgnt.csv | 7,295,333 |
| Participant1_gyro.csv | 6,319,911 |
| Participant1_env_temp.csv | 4,168,528 |
| Participant1_bone_acc.csv | 7,782,518 |
| Participant1_baro.csv | 4,764,032 |
| Participant1_acc.csv | 6,397,007 |

**Participant2** (8 files, 48,987,607 bytes)

| Title | Size (bytes) |
|---|---:|
| Participant2_skin_temp.csv | 2,637,152 |
| Participant2_ppg.csv | 9,328,842 |
| Participant2_mgnt.csv | 7,344,802 |
| Participant2_gyro.csv | 6,359,332 |
| Participant2_env_temp.csv | 4,196,444 |
| Participant2_bone_acc.csv | 7,883,698 |
| Participant2_baro.csv | 4,795,936 |
| Participant2_acc.csv | 6,441,401 |

**Participant3** (8 files, 11,306,895 bytes)

| Title | Size (bytes) |
|---|---:|
| Participant3_acc.csv | 1,503,108 |
| Participant3_ppg.csv | 2,147,254 |
| Participant3_mgnt.csv | 1,713,432 |
| Participant3_gyro.csv | 1,482,652 |
| Participant3_bone_acc.csv | 1,788,673 |
| Participant3_skin_temp.csv | 607,236 |
| Participant3_baro.csv | 1,101,088 |
| Participant3_env_temp.csv | 963,452 |

#### elan (4 files, 153,031 bytes)

| Title | File ID | Size (bytes) |
|---|---|---:|
| Group_7_individual_build_renamed.csv | 1gGBEz7oNVxrdlBW0XkHhzOxDAhsZgLr6 | 46,038 |
| Group_7_with_individual_build.csv | 1M26TGMVcQYDxDSqFnmSIOOR0d1gcqPZt | 42,397 |
| Group_7_clean.csv | 1JtID83wZRQ-gwTjDio9OQq1H_BoSKbhx | 34,394 |
| Group_7.csv | 1bM8A9rq6IyZdor49CGKDOxlP85w-z3QP | 30,202 |

### group_8

**Total: 313,324,422 bytes (0.2918 GiB)**

#### optitrack (5 files, 154,083,031 bytes)

| Title | File ID | Size (bytes) |
|---|---|---:|
| Arda_Group_8_Take_3.csv | 1vrkOf42uDyD-ldntQR2L30sq6vvIuF8k | 66,421,912 |
| Arda_Group_8_Take_5.csv | 1nyMvszoowSbrPGscXeaXD8da4gmXKhMi | 10,208,007 |
| Arda_Group_8_Take_2.csv | 1Ysol_Ak56l6GESzP2ugAgkDKnCKIcirj | 31,683,384 |
| Arda_Group_8_Take_1.csv | 1y0PMgjItRAAKohVzGVnMYX5S1pz4adk- | 31,341,503 |
| Arda_Group_8_Take_4.csv | 1yl6_ckBJGNCHOO4aT0RB5brQBRiU1X9D | 14,428,225 |

#### xsens (3 files, 42,912,067 bytes)

| Title | File ID | Size (bytes) |
|---|---|---:|
| Participant2.csv | 1ZWf9zbGQ1AnxmN3kCNNNyHcr8jvN56Co | 14,304,132 |
| Participant1.csv | 1C1OH5Ce5RAzFyfOKFlp85bJvMNfA-mBr | 14,313,793 |
| Participant3.csv | 1Tz9mLSl-WA80XSLkjC8cVfEltdzYVtBS | 14,294,142 |

#### openearable (24 files, 116,243,497 bytes)

**Participant1** (8 files, 38,755,893 bytes)

| Title | Size (bytes) |
|---|---:|
| Participant1_skin_temp.csv | 2,070,376 |
| Participant1_ppg.csv | 7,380,485 |
| Participant1_mgnt.csv | 5,814,879 |
| Participant1_gyro.csv | 5,046,650 |
| Participant1_env_temp.csv | 3,322,284 |
| Participant1_bone_acc.csv | 6,328,975 |
| Participant1_baro.csv | 3,796,896 |
| Participant1_acc.csv | 4,995,348 |

**Participant2** (8 files, 38,735,737 bytes)

| Title | Size (bytes) |
|---|---:|
| Participant2_skin_temp.csv | 2,082,360 |
| Participant2_ppg.csv | 7,401,960 |
| Participant2_mgnt.csv | 5,804,277 |
| Participant2_gyro.csv | 5,031,385 |
| Participant2_env_temp.csv | 3,315,816 |
| Participant2_bone_acc.csv | 6,216,900 |
| Participant2_baro.csv | 3,789,504 |
| Participant2_acc.csv | 5,093,535 |

**Participant3** (8 files, 38,751,867 bytes)

| Title | Size (bytes) |
|---|---:|
| Participant3_skin_temp.csv | 2,087,176 |
| Participant3_ppg.csv | 7,345,999 |
| Participant3_mgnt.csv | 5,912,500 |
| Participant3_gyro.csv | 5,013,636 |
| Participant3_env_temp.csv | 3,310,496 |
| Participant3_bone_acc.csv | 6,213,880 |
| Participant3_baro.csv | 3,783,424 |
| Participant3_acc.csv | 5,084,756 |

#### elan (3 files, 85,827 bytes)

| Title | File ID | Size (bytes) |
|---|---|---:|
| Group_8_individual_build_renamed.csv | 1uyjFGSwMZ-ZvlpBhjH2zt2sBI1YGuq7J | 33,025 |
| Group_8_with_individual_build.csv | 1mmcuRzWct1Gv8F86Fr164DEiJQgS3klv | 30,880 |
| Group_8.csv | 1AfEKOQNgRq5k0Ci4Zoe3y63m6UE90ZTi | 21,922 |

### group_9

**Total: 381,907,766 bytes (0.3557 GiB)**

#### optitrack (6 files, 214,507,170 bytes)

| Title | File ID | Size (bytes) |
|---|---|---:|
| Arda_Group_9_Take_4.csv | 1ZieKmABOdB0lEXuM7wcq-lTtJOGl5yI2 | 50,659,565 |
| Arda_Group_9_Take_3.csv | 1zGGyv1CvvaYYGNd4BdF159qk8mpg1scI | 30,007,135 |
| Arda_Group_9_Take_2.csv | 1671S8wdVs8_BVvnO9Kr1yDSYxPagTmG1 | 38,415,184 |
| Arda_Group_9_Take_1.csv | 1IFBAkZjvGQl8ewZ2KfR3E-ZF6AG07JJC | 40,164,443 |
| Arda_Group_9_Take_6.csv | 1wIDgKtjAMiohi-EFTxm30dTTU2x-HjBB | 8,250,478 |
| Arda_Group_9_Take_5.csv | 1BAYlkxVaJO8isuK8T4n4k-zL5YYqko08 | 47,010,365 |

#### xsens (3 files, 43,843,141 bytes)

| Title | File ID | Size (bytes) |
|---|---|---:|
| Participant3.csv | 1BKn6cZIFwAI9A-aArxw9VlLH8ogwIvVX | 14,679,917 |
| Participant1.csv | 1uBnjk7NOrpH59SIWsJwSsFboNrpeYQN6 | 14,417,763 |
| Participant2.csv | 1o3eVUfTJ13kutvlk2FSmVWZG62mT0HYm | 14,745,461 |

#### openearable (24 files, 123,338,263 bytes)

**Participant1** (8 files, 59,428,108 bytes)

| Title | Size (bytes) |
|---|---:|
| Participant1_skin_temp.csv | 3,197,404 |
| Participant1_ppg.csv | 11,330,119 |
| Participant1_mgnt.csv | 8,903,255 |
| Participant1_gyro.csv | 7,706,826 |
| Participant1_env_temp.csv | 5,086,788 |
| Participant1_bone_acc.csv | 9,574,733 |
| Participant1_baro.csv | 5,813,472 |
| Participant1_acc.csv | 7,815,511 |

**Participant2** (8 files, 59,654,843 bytes)

| Title | Size (bytes) |
|---|---:|
| Participant2_skin_temp.csv | 3,202,668 |
| Participant2_ppg.csv | 11,400,942 |
| Participant2_mgnt.csv | 9,081,898 |
| Participant2_gyro.csv | 7,714,948 |
| Participant2_env_temp.csv | 5,085,304 |
| Participant2_bone_acc.csv | 9,715,041 |
| Participant2_baro.csv | 5,811,767 |
| Participant2_acc.csv | 7,642,275 |

**Participant3** (8 files, 4,255,312 bytes)

| Title | Size (bytes) |
|---|---:|
| Participant3_ppg.csv | 807,352 |
| Participant3_mgnt.csv | 639,842 |
| Participant3_bone_acc.csv | 683,013 |
| Participant3_gyro.csv | 555,353 |
| Participant3_acc.csv | 561,425 |
| Participant3_baro.csv | 417,023 |
| Participant3_env_temp.csv | 364,896 |
| Participant3_skin_temp.csv | 226,408 |

#### elan (5 files, 219,192 bytes)

| Title | File ID | Size (bytes) |
|---|---|---:|
| Group_9_individual_build_renamed.csv | 1neo8pgTUmC-s_R816d_czl-HF4Z3McnY | 48,483 |
| Group_9_with_individual_build.csv | 1l18OuKYggaKeQnBCzhkXutnj9V4YkBun | 44,988 |
| Group_9_individual_build_renamed_BACKUP_before_sync_patch.csv | 1HXhFCdlKz9hRFSTF05gxiIZLCwQUNWZX | 48,483 |
| Group_9_with_individual_build_BACKUP_before_sync_patch.csv | 1gGUyz_Zk6FlBL7CDI9Y98Qy6hLQtWiBy | 44,988 |
| Group_9.csv | 162PeQeDTHRvo3_nnRukqmUtC7AGOFG03 | 32,250 |

### group_10

**Total: 144,932,367 bytes (0.1350 GiB)**

#### optitrack (3 files, 54,667,843 bytes)

| Title | File ID | Size (bytes) |
|---|---|---:|
| Arda_Group_10_Take_1.csv | 1PLVwLjZJHwVUbZtKaW2ck-CmUcNcv5oW | 23,169,212 |
| Arda_Group_10_Take_3.csv | 1Y1kbDBeOQ578WRRo0YN-ttdHnThVJpZl | 10,238,532 |
| Arda_Group_10_Take_2.csv | 1RR6IKYZfgKqn4ozJwyGXH249hkDdQXWI | 21,260,099 |

#### xsens (3 files, 22,694,495 bytes)

| Title | File ID | Size (bytes) |
|---|---|---:|
| Participant2.csv | 12Sa5lY-8LYvLGV-p6kyFSiK75XxMhMMM | 5,431,186 |
| Participant1.csv | 1QcYwgbKruRn9svTPNKAEqx_Y3inJEOst | 8,231,348 |
| Participant3.csv | 1Pt7hd2zgHsFCVKCd2sR0wG1szgOWE3FX | 9,031,961 |

#### openearable (24 files, 67,525,163 bytes)

**Participant1** (8 files, 24,292,488 bytes)

| Title | Size (bytes) |
|---|---:|
| Participant1_skin_temp.csv | 1,287,804 |
| Participant1_ppg.csv | 4,730,452 |
| Participant1_mgnt.csv | 3,627,862 |
| Participant1_gyro.csv | 3,142,045 |
| Participant1_env_temp.csv | 2,072,588 |
| Participant1_bone_acc.csv | 3,879,856 |
| Participant1_baro.csv | 2,368,672 |
| Participant1_acc.csv | 3,183,209 |

**Participant2** (8 files, 18,357,028 bytes)

| Title | Size (bytes) |
|---|---:|
| Participant2_ppg.csv | 3,496,041 |
| Participant2_mgnt.csv | 2,758,063 |
| Participant2_gyro.csv | 2,383,498 |
| Participant2_env_temp.csv | 1,572,648 |
| Participant2_bone_acc.csv | 2,940,019 |
| Participant2_baro.csv | 1,797,312 |
| Participant2_acc.csv | 2,421,103 |
| Participant2_skin_temp.csv | 988,344 |

**Participant3** (8 files, 24,875,647 bytes)

| Title | Size (bytes) |
|---|---:|
| Participant3_skin_temp.csv | 1,328,432 |
| Participant3_ppg.csv | 4,779,891 |
| Participant3_mgnt.csv | 3,779,000 |
| Participant3_gyro.csv | 3,212,393 |
| Participant3_env_temp.csv | 2,115,736 |
| Participant3_bone_acc.csv | 4,067,893 |
| Participant3_baro.csv | 2,417,984 |
| Participant3_acc.csv | 3,174,318 |

#### elan (3 files, 44,866 bytes)

| Title | File ID | Size (bytes) |
|---|---|---:|
| Group_10_individual_build_renamed.csv | 1Qz0YgTQPCPtMjcNQlvGuexCUTx8V2clC | 17,233 |
| Group_10_with_individual_build.csv | 1-lCWfVjXcLouaCLyH1jzRdzH6ecpWIXj | 15,993 |
| Group_10.csv | 1LO1WHdEjy_YQf9aQgVuXUHXx5C4_4xZj | 11,640 |

## Methodology Notes

- Enumeration performed via Google Drive `search_files` with `parentId` queries, `excludeContentSnippets: true` — no file content was downloaded or read.
- Derived/processed subfolders found but deliberately not expanded (contents excluded from totals): `model_ready/`, `optitrack_labeled/`, `optitrack_final/`, `xsens_labeled/`, `openearable_labeled/`, `optitrack_cleaned_manual_stitched/`, `optitrack_inspection/`, and all `.ipynb_checkpoints/` folders.
- `fileSize` as reported by Drive metadata (exact byte count, not an estimate).
- Report generated 2026-09-03.
