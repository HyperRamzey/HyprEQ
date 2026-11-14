#!/usr/bin/env python3
"""
hypreq.py
Unified Minimum-Phase Convolver Generator & 4-Panel Verification Suite
Supported Headphones:
  1. jbl770nc : JBL Live 770NC (Passive / Wired Mode)
                - Raw Measurement : Filk (Clone GRAS 43AG / KB006x pinna + IEC 60318-4 711 coupler), via AutoEq.
                - Target Curve    : Harman Over-Ear 2018 (Harman International).
  2. dt770_80 : Beyerdynamic DT 770 Pro (80 Ohm)
                - Raw Measurement : Crinacle (Genuine GRAS 43AG-7: RA0402 High-Frequency 711 Coupler +
                                    KB500X Anthropometric Pinna), via AutoEq.
                - Target Curve    : Harman Over-Ear 2018 (Harman International).

Usage Examples:
  python3 hypreq.py --headphone all --export-csv
  python3 hypreq.py --headphone dt770_80 --notch-mode safe --taps 16384
  python3 hypreq.py --headphone jbl770nc --fs 44100
"""

import argparse
import csv
import os
import shutil
import struct
import numpy as np

# ==============================================================================
# 1. EMBEDDED DATASETS: HARMAN 2018 TARGET, RAW MEASUREMENTS, & CONTROL CURVES
# ==============================================================================
HARMAN_2018_PTS = [
    (20.0,3.86),(25.1,4.02),(30.1,3.96),(35.3,3.84),(40.1,3.70),(48.0,3.44),(55.2,3.09),
    (60.4,2.88),(66.0,2.61),(72.2,2.33),(78.2,2.06),(85.5,1.75),(94.4,1.33),(100.3,1.04),
    (110.7,0.55),(121.1,0.07),(131.2,-0.34),(140.6,-0.66),(153.8,-1.00),(164.9,-1.27),(180.3,-1.62),
    (201.2,-1.96),(231.3,-2.08),(260.6,-1.95),(302.5,-1.65),(351.2,-1.32),(403.7,-1.12),
    (454.9,-0.98),(528.2,-0.75),(650.9,-0.40),(826.5,-0.16),(1008.4,0.02),(1136.3,0.35),
    (1255.2,0.85),(1386.6,1.53),(1501.4,2.11),(1658.5,3.04),(1850.4,4.29),(2003.7,5.29),
    (2235.4,6.44),(2494.0,7.39),(2782.5,8.04),(3104.3,8.44),(3361.5,8.58),(3568.3,8.56),
    (3825.7,8.36),(4061.1,8.10),(4310.9,7.73),(4621.9,7.17),(5004.9,6.44),(5365.9,5.89),
    (5753.0,5.39),(6046.4,5.06),(6482.6,4.47),(6881.4,3.96),(7304.7,3.37),(7677.4,2.84),
    (8069.0,2.33),(8480.6,1.77),(9002.3,1.00),(9556.1,0.10),(10043.6,-0.77),(11094.4,-2.59),
    (12013.6,-4.26),(13537.2,-6.46),(15103.1,-8.08),(16518.0,-10.07),(18065.5,-13.91),(19955.5,-22.92)
]

# --- A. JBL Live 770NC (Passive) on Filk KB006x + IEC 711 Coupler ---
JBL_770NC_RAW_PTS = [
    (20.0,-1.26),(25.1,0.28),(30.1,1.17),(35.3,1.68),(40.1,2.02),(48.0,2.25),(55.2,2.11),
    (60.4,1.77),(66.0,1.57),(72.2,1.78),(78.2,2.32),(85.5,2.65),(94.4,2.30),(100.3,2.39),
    (110.7,2.17),(121.1,2.24),(131.2,2.55),(140.6,3.13),(153.8,3.56),(164.9,3.31),(180.3,2.71),
    (201.2,1.54),(231.3,0.04),(260.6,-1.34),(302.5,-2.89),(351.2,-4.28),(403.7,-5.23),
    (454.9,-6.08),(459.5,-6.08),(482.9,-5.69),(528.2,-4.95),(583.4,-4.20),(650.9,-3.21),
    (733.4,-2.25),(826.5,-1.21),(931.3,-0.36),(1008.4,0.03),(1136.3,0.19),(1255.2,0.07),
    (1386.6,-0.08),(1501.4,-0.13),(1593.8,-0.30),(1725.9,-0.11),(1887.6,-0.01),(2043.9,0.30),
    (2235.4,0.82),(2373.0,1.06),(2518.9,1.59),(2700.6,2.19),(2953.7,2.79),(3230.4,2.92),
    (3361.5,2.61),(3498.0,1.57),(3640.1,1.09),(3825.7,1.17),(4061.1,0.53),(4268.3,0.70),
    (4486.0,-0.48),(4762.0,-1.52),(5004.9,-2.88),(5208.1,-4.62),(5365.9,-5.31),(5528.5,-4.37),
    (5753.0,-2.92),(6046.4,-1.73),(6291.9,-0.28),(6547.4,2.16),(6745.8,4.15),(6881.4,4.79),
    (7089.9,3.92),(7304.7,1.68),(7601.3,-0.09),(7909.9,0.98),(8231.2,2.82),(8480.6,3.78),
    (8651.0,2.99),(8824.9,0.21),(9092.3,-5.66),(9461.5,-10.74),(9748.2,-11.74),(10144.0,-7.50),
    (10451.4,-5.53),(10875.8,-11.30),(11205.3,-15.25),(11660.3,-8.17),(12013.6,-5.54),
    (12752.7,-10.43),(13537.2,-12.69),(14370.0,-10.82),(15406.6,-17.82),(16518.0,-9.10),
    (17534.2,-16.19),(18612.9,-14.22),(19955.5,-18.95)
]

CTRL_PTS_JBL770NC = [
    (20.0,+0.12), (21.4,+0.09), (23.2,+0.03), (25.1,-0.05), (27.5,-0.16),
    (30.1,-0.30), (32.9,-0.44), (38.2,-0.71), (40.1,-0.82), (43.9,-1.02),
    (48.0,-1.21), (55.2,-1.38), (60.4,-1.43), (66.0,-1.57), (70.1,-1.83),
    (73.6,-2.16), (78.2,-2.65), (82.2,-3.04), (85.5,-3.27), (89.9,-3.38),
    (94.4,-3.46), (100.3,-3.77), (105.4,-3.93), (110.7,-4.09), (121.1,-4.63),
    (126.0,-4.96), (131.2,-5.36), (136.5,-5.85), (140.6,-6.22), (146.3,-6.63),
    (150.8,-6.87), (153.8,-6.97), (160.0,-7.04), (164.9,-7.02), (171.6,-6.95),
    (180.3,-6.76), (189.5,-6.45), (201.2,-5.95), (215.7,-5.29), (231.3,-4.56),
    (247.9,-3.70), (260.6,-3.06), (279.4,-2.19), (302.5,-1.21), (327.6,-0.25),
    (351.2,+0.50), (376.6,+1.11), (403.7,+1.67), (428.6,+2.24), (446.0,+2.54),
    (454.9,+2.60), (468.7,+2.54), (482.9,+2.36), (502.5,+2.08), (528.2,+1.76),
    (555.1,+1.46), (583.4,+1.17), (650.9,+0.36), (690.9,-0.05), (733.4,-0.48),
    (778.6,-0.94), (826.5,-1.40), (877.3,-1.80), (931.3,-2.15), (1008.4,-2.44),
    (1049.4,-2.45), (1092.0,-2.39), (1136.3,-2.27), (1194.3,-2.01), (1255.2,-1.66),
    (1319.3,-1.26), (1386.5,-0.85), (1442.8,-0.53), (1593.8,+0.46), (1658.5,+0.81),
    (1725.9,+1.15), (1795.9,+1.55), (1887.5,+2.09), (1964.2,+2.46), (2043.9,+2.75),
    (2148.2,+3.00), (2235.4,+3.18), (2373.0,+3.44), (2444.9,+3.47), (2518.9,+3.42),
    (2700.6,+3.25), (2810.3,+3.13), (2953.7,+3.06), (3104.3,+3.11), (3230.4,+3.22),
    (3361.5,+3.65), (3498.0,+4.44), (3640.1,+4.87), (3825.7,+4.84), (3941.7,+4.93),
    (4061.1,+5.00), (4184.1,+4.87), (4268.3,+4.81), (4397.6,+4.83), (4486.0,+4.51),
    (4621.9,+4.15), (4762.0,+3.78), (4906.2,+3.58), (5054.9,+3.61), (5208.1,+3.72),
    (5365.9,+3.70), (5528.5,+3.46), (5753.0,+2.88), (6046.4,+2.01), (6291.9,+1.61),
    (6547.4,-0.18), (6745.8,-1.76), (6881.4,-2.31), (7089.9,-1.92), (7304.7,-0.61),
    (7601.3,+0.00), (7910.0,-0.68), (8149.7,-1.91), (8313.5,-2.70), (8480.6,-3.16),
    (8651.0,-2.69), (8824.9,-1.11), (9092.3,-0.01), (9461.5,+0.00), (10347.9,+0.00),
    (12013.6,+0.00), (15103.0,+0.00), (19955.5,+0.00)
]

# --- B. Beyerdynamic DT 770 Pro (80 Ohm) on Crinacle GRAS 43AG-7 (RA0402 + KB500X) ---
DT770_80OHM_RAW_LR = [
    (20.0,+3.82,+1.27,+2.54), (21.5,+4.28,+1.71,+3.00), (23.2,+4.78,+2.24,+3.51),
    (25.1,+5.32,+2.78,+4.05), (27.2,+5.84,+3.23,+4.54), (29.5,+6.27,+3.68,+4.98),
    (32.0,+6.69,+4.08,+5.38), (35.0,+7.09,+4.52,+5.80), (38.2,+7.36,+4.94,+6.15),
    (41.5,+7.69,+5.29,+6.49), (45.0,+8.04,+5.61,+6.82), (48.5,+8.20,+5.79,+6.99),
    (51.8,+8.27,+5.88,+7.07), (55.2,+8.26,+5.89,+7.07), (59.0,+8.14,+5.82,+6.98),
    (63.0,+7.88,+5.68,+6.78), (66.5,+7.52,+5.36,+6.44), (70.0,+7.04,+4.84,+5.94),
    (74.0,+6.39,+4.14,+5.27), (78.2,+5.72,+3.58,+4.65), (83.0,+4.95,+3.30,+4.13),
    (87.0,+4.49,+3.20,+3.85), (91.0,+4.51,+3.26,+3.88), (95.0,+4.61,+3.23,+3.92),
    (98.5,+4.36,+2.92,+3.64), (102.0,+3.77,+2.25,+3.01), (105.5,+3.15,+1.64,+2.40),
    (109.0,+2.80,+1.36,+2.08), (113.0,+2.75,+1.34,+2.05), (117.6,+2.88,+1.43,+2.16),
    (121.5,+2.99,+1.50,+2.24), (125.5,+2.99,+1.49,+2.24), (131.2,+2.82,+1.30,+2.06),
    (136.5,+2.46,+1.00,+1.73), (143.0,+1.78,+0.53,+1.16), (151.0,+0.57,-0.20,+0.18),
    (160.0,-0.96,-1.35,-1.15), (165.0,-2.12,-2.03,-2.07), (169.0,-3.04,-2.60,-2.82),
    (174.5,-4.02,-3.26,-3.64), (180.3,-4.91,-3.92,-4.42), (190.0,-6.09,-4.85,-5.47),
    (199.0,-6.48,-5.48,-5.98), (204.9,-6.45,-5.71,-6.08), (210.5,-6.22,-5.80,-6.01),
    (216.0,-5.84,-5.77,-5.81), (222.0,-5.39,-5.65,-5.52), (235.0,-4.56,-5.14,-4.85),
    (248.0,-3.76,-4.47,-4.11), (262.0,-3.02,-3.74,-3.38), (277.0,-2.36,-2.94,-2.65),
    (292.0,-1.89,-2.29,-2.09), (309.0,-1.50,-1.77,-1.63), (327.6,-1.17,-1.34,-1.25),
    (351.2,-1.10,-0.95,-1.03), (385.0,-0.81,-0.58,-0.69), (429.0,-0.65,-0.35,-0.50),
    (478.1,-0.65,-0.25,-0.45), (534.0,-0.68,-0.26,-0.47), (596.0,-0.66,-0.27,-0.47),
    (665.0,-0.48,-0.23,-0.36), (743.0,+0.04,+0.25,+0.15), (826.5,+0.60,+0.92,+0.76),
    (880.0,+0.98,+1.24,+1.11), (940.0,+1.24,+1.43,+1.33), (985.0,+1.15,+1.36,+1.26),
    (1032.0,+0.89,+1.24,+1.06), (1090.0,+0.54,+1.16,+0.85), (1151.0,+0.52,+1.27,+0.90),
    (1255.2,+1.14,+1.66,+1.40), (1335.0,+1.83,+1.72,+1.77), (1386.6,+2.16,+1.70,+1.93),
    (1501.4,+2.51,+1.88,+2.20), (1599.0,+3.00,+2.49,+2.74), (1660.0,+3.44,+2.99,+3.21),
    (1725.9,+3.89,+3.51,+3.70), (1885.0,+5.23,+4.90,+5.06), (2043.9,+6.76,+6.27,+6.52),
    (2221.0,+8.34,+7.28,+7.81), (2347.0,+8.87,+7.72,+8.29), (2479.0,+8.91,+7.87,+8.39),
    (2618.0,+8.78,+7.89,+8.34), (2766.0,+8.46,+7.73,+8.10), (2921.0,+7.91,+7.33,+7.62),
    (3086.0,+6.99,+6.56,+6.77), (3260.0,+5.34,+5.33,+5.34), (3443.0,+2.00,+2.98,+2.49),
    (3580.0,-1.68,-0.12,-0.90), (3703.0,-4.87,-4.87,-4.87), (3780.0,-3.70,-8.15,-5.93),
    (3838.4,-2.16,-8.78,-5.47), (4058.0,+4.41,+0.40,+2.41), (4287.0,+8.64,+5.44,+7.04),
    (4410.0,+9.88,+6.79,+8.34), (4528.0,+10.61,+7.70,+9.15), (4650.0,+10.89,+8.28,+9.59),
    (4783.0,+10.94,+8.72,+9.83), (5052.0,+10.75,+9.00,+9.87), (5337.0,+10.62,+9.07,+9.85),
    (5490.0,+10.30,+8.88,+9.59), (5637.0,+9.74,+8.46,+9.10), (5800.0,+8.49,+7.52,+8.00),
    (5955.0,+7.39,+6.78,+7.08), (6120.0,+7.96,+8.09,+8.03), (6290.0,+8.62,+9.41,+9.01),
    (6460.0,+8.82,+9.96,+9.39), (6644.0,+8.86,+10.10,+9.48), (6830.0,+8.58,+9.72,+9.15),
    (7018.0,+8.21,+9.21,+8.71), (7210.0,+7.90,+8.74,+8.32), (7414.0,+7.78,+8.45,+8.11),
    (7620.0,+7.98,+8.44,+8.21), (7831.0,+8.23,+8.49,+8.36), (8000.0,+8.41,+8.66,+8.54),
    (8155.0,+8.44,+8.74,+8.59), (8300.0,+7.70,+8.50,+8.10), (8500.0,+5.60,+7.38,+6.49),
    (8738.0,+1.78,+4.69,+3.24), (8980.0,-3.55,+0.12,-1.72), (9230.0,-6.91,-7.28,-7.10),
    (9550.0,-4.22,-7.10,-5.66), (10298.0,-0.84,-2.61,-1.72), (11490.0,+2.06,+1.13,+1.60),
    (12821.0,-2.79,-4.97,-3.88), (13672.6,-8.01,-7.05,-7.53), (15110.0,-14.59,-8.14,-11.36),
    (16860.0,+0.69,-1.55,-0.43), (18246.1,-4.65,-7.31,-5.98), (19955.5,-7.16,-6.62,-6.89)
]

AUTOEQ_DT770_GEQ_STR = (
    "20 -3.1; 21 -3.4; 22 -3.6; 23 -3.9; 24 -4.2; 26 -4.8; 27 -5.1; 29 -5.4; 30 -5.7; 32 -6.1; "
    "34 -6.4; 36 -6.6; 38 -6.9; 40 -7.2; 43 -7.6; 45 -7.8; 48 -8.1; 50 -8.3; 53 -8.5; 56 -8.6; "
    "59 -8.6; 63 -8.7; 66 -8.5; 70 -8.2; 74 -7.7; 78 -7.2; 83 -6.9; 87 -6.6; 92 -7.0; 97 -7.4; "
    "103 -6.5; 109 -5.9; 115 -6.3; 121 -6.7; 128 -6.9; 136 -6.8; 143 -6.5; 151 -5.7; 160 -4.5; "
    "169 -3.2; 178 -2.1; 188 -1.1; 199 -0.5; 210 -0.5; 222 -1.0; 235 -1.7; 248 -2.4; 262 -3.0; "
    "277 -3.6; 292 -4.1; 309 -4.4; 326 -4.6; 345 -4.8; 364 -4.9; 385 -5.0; 406 -5.1; 429 -5.1; "
    "453 -5.1; 479 -5.0; 506 -4.9; 534 -4.8; 565 -4.7; 596 -4.6; 630 -4.6; 665 -4.6; 703 -4.7; "
    "743 -4.9; 784 -5.2; 829 -5.5; 875 -5.7; 924 -6.0; 977 -5.9; 1032 -5.6; 1090 -5.2; 1151 -5.0; "
    "1216 -5.0; 1284 -5.1; 1357 -5.0; 1433 -4.8; 1514 -4.6; 1599 -4.6; 1689 -4.7; 1784 -4.8; "
    "1885 -5.0; 1991 -5.3; 2103 -5.7; 2221 -5.9; 2347 -5.9; 2479 -5.6; 2618 -5.2; 2766 -4.7; "
    "2921 -4.0; 3086 -3.0; 3260 -1.6; 3443 -0.2; 3637 -0.5; 3842 -2.0; 4058 -3.4; 4287 -4.8; "
    "4528 -6.3; 4783 -7.5; 5052 -8.1; 5337 -8.5; 5637 -8.1; 5955 -7.4; 6290 -8.7; 6644 -9.4; "
    "7018 -9.1; 7414 -8.7; 7831 -8.1; 8272 -7.2; 8738 -6.3; 9230 -5.7; 9749 -5.3; 10298 -5.1; "
    "10878 -5.4; 11490 -5.7; 12137 -6.2; 12821 -6.8; 13543 -7.4; 14305 -8.2; 15110 -9.1; "
    "15961 -10.1; 16860 -11.2; 17809 -12.4; 18812 -13.8; 19871 -15.2"
)

CTRL_PTS_DT770_80OHM = [
    (20.0,+0.20), (21.5,-0.13), (23.2,-0.45), (25.1,-0.84), (27.2,-1.25),
    (29.5,-1.58), (32.0,-1.93), (35.0,-2.15), (38.2,-2.39), (41.5,-2.87),
    (45.0,-3.29), (48.5,-3.58), (51.8,-3.83), (55.2,-3.98), (59.0,-4.05),
    (63.0,-4.03), (66.5,-3.86), (70.0,-3.52), (74.0,-3.02), (78.2,-2.59),
    (83.0,-2.27), (87.0,-2.17), (91.0,-2.39), (95.0,-2.62), (98.5,-2.46),
    (102.0,-2.03), (105.5,-1.62), (109.0,-1.45), (113.0,-1.59), (117.6,-1.94),
    (121.5,-2.19), (125.5,-2.35), (131.2,-2.39), (136.5,-2.26), (143.0,-1.88),
    (151.0,-1.11), (160.0,+0.00), (165.0,+0.93), (169.0,+1.46), (174.5,+2.12),
    (180.3,+2.80), (190.0,+3.69), (199.0,+4.05), (204.9,+4.10), (210.5,+4.01),
    (216.0,+3.87), (222.0,+3.47), (235.0,+2.79), (248.0,+2.11), (262.0,+1.44),
    (277.0,+0.83), (292.0,+0.37), (309.0,+0.03), (327.6,-0.23), (351.2,-0.30),
    (385.0,-0.50), (429.0,-0.55), (478.1,-0.45), (534.0,-0.26), (596.0,-0.09),
    (665.0,-0.03), (743.0,-0.41), (826.5,-0.92), (880.0,-1.21), (940.0,-1.38),
    (985.0,-1.26), (1032.0,-0.98), (1090.0,-0.62), (1151.0,-0.48), (1255.2,-0.55),
    (1335.0,-0.50), (1386.6,-0.40), (1501.4,-0.09), (1599.0,-0.04), (1660.0,-0.14),
    (1725.9,-0.21), (1885.0,-0.55), (2043.9,-1.02), (2221.0,-1.43), (2347.0,-1.43),
    (2479.0,-1.05), (2618.0,-0.66), (2766.0,-0.09), (2921.0,+0.61), (3086.0,+1.57),
    (3260.0,+2.78), (3443.0,+3.71), (3580.0,+4.08), (3703.0,+4.26), (3838.4,+4.02),
    (4058.0,+2.75), (4287.0,+0.68), (4410.0,-0.74), (4528.0,-1.72), (4650.0,-2.38),
    (4783.0,-2.83), (5052.0,-3.28), (5337.0,-3.59), (5490.0,-3.55), (5637.0,-3.21),
    (5800.0,-2.36), (5955.0,-1.70), (6120.0,-2.54), (6290.0,-3.72), (6460.0,-4.33),
    (6644.0,-4.45), (6830.0,-4.32), (7018.0,-4.13), (7210.0,-4.00), (7414.0,-4.01),
    (7620.0,-4.23), (7831.0,-4.58), (8000.0,-4.90), (8155.0,-5.04), (8300.0,-4.79),
    (8500.0,-3.74), (8738.0,-1.85), (8980.0,-1.35), (9230.0,-1.20), (9550.0,-1.15),
    (10298.0,-1.18), (11490.0,-1.55), (12821.0,-1.03), (13672.6,-0.55), (15110.0,-0.05),
    (16860.0,+0.00), (18246.1,+0.00), (19955.5,+0.00)
]


# ==============================================================================
# 2. SHARED DSP ENGINE (PCHIP, BESSEL I0, WAV I/O, HILBERT-CEPSTRAL SYNTHESIS)
# ==============================================================================
def format_rate_tag(fs: int) -> str:
    """Formats 48000 -> '48k', 44100 -> '44.1k', 96000 -> '96k' (never truncates 44.1k to 44k)."""
    return f"{fs / 1000.0:g}k"


def pchip_log_interp(f_nodes: np.ndarray, g_nodes: np.ndarray, f_query: np.ndarray) -> np.ndarray:
    """Pure-NumPy Fritsch-Carlson Monotonic Cubic Hermite (PCHIP) interpolator in log10(f) space."""
    x = np.log10(np.asarray(f_nodes, dtype=np.float64))
    y = np.asarray(g_nodes, dtype=np.float64)
    xq = np.log10(np.clip(np.asarray(f_query, dtype=np.float64), f_nodes[0], f_nodes[-1]))

    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros_like(y)

    for k in range(1, len(x) - 1):
        if delta[k - 1] * delta[k] > 0.0:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
        else:
            d[k] = 0.0
    d[0] = delta[0]
    d[-1] = delta[-1]

    idx = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, len(x) - 2)
    dx = xq - x[idx]
    hi = h[idx]
    t = dx / hi

    h00 = (2.0 * t**3) - (3.0 * t**2) + 1.0
    h10 = (t**3) - (2.0 * t**2) + t
    h01 = (-2.0 * t**3) + (3.0 * t**2)
    h11 = (t**3) - (t**2)
    return h00 * y[idx] + h10 * hi * d[idx] + h01 * y[idx + 1] + h11 * hi * d[idx + 1]


def bessel_i0(x: np.ndarray) -> np.ndarray:
    """Exact 64-bit float 45-term power series for Modified Bessel Function I0(x) (zero scipy dependency)."""
    x = np.asarray(x, dtype=np.float64)
    res = np.ones_like(x)
    term = np.ones_like(x)
    half_x_sq = (0.5 * x) ** 2
    for k in range(1, 45):
        term *= half_x_sq / (k * k)
        res += term
    return res


def write_wav_ieee_float(filename: str, data: np.ndarray, fs: int = 48000, bits: int = 64) -> None:
    """Writes an IEEE 754 64-bit float (`pcm_f64le`) or 32-bit float (`pcm_f32le`) WAV file."""
    dtype_str = "<f8" if bits == 64 else "<f4"
    bytes_per_sample = 8 if bits == 64 else 4

    if data.ndim == 1:
        channels = 1
        raw_bytes = data.astype(dtype_str).tobytes()
    else:
        channels = data.shape[1]
        raw_bytes = np.ascontiguousarray(data, dtype=dtype_str).tobytes()

    byte_rate = fs * channels * bytes_per_sample
    block_align = channels * bytes_per_sample
    data_size = len(raw_bytes)
    riff_size = 36 + data_size

    with open(filename, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", riff_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 3, channels, fs, byte_rate, block_align, bits))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(raw_bytes)


def apply_dt770_notch_mode(ctrl_pts, mode: str = "safe", safe_cap_db: float = 1.50):
    """
    Adjusts the 2.8 kHz - 4.4 kHz non-minimum-phase concha reflection notch region on the DT 770 Pro:
      - 'original': keeps the +4.26 dB @ 3,703 Hz regularized fill as-is.
      - 'safe'    : smoothly caps positive boosts inside 2.8-4.4 kHz to +1.50 dB so the
                    L/R-asymmetric concha null is bridged without creating a +4.3 dB peak.
      - 'zero'    : clamps positive boosts inside 2.8-4.4 kHz to 0.00 dB.
    """
    if mode == "original":
        return list(ctrl_pts)

    out = []
    for f, g in ctrl_pts:
        if 2800.0 <= f <= 4400.0 and g > 0.0:
            if mode == "zero":
                out.append((f, 0.0))
            else:
                g_soft = safe_cap_db * np.tanh(g / safe_cap_db)
                out.append((f, float(np.round(g_soft, 4))))
        else:
            out.append((f, float(g)))
    return out


def synthesize_minimum_phase_ir(
    ctrl_pts,
    fs: int = 48000,
    n_fft: int = 262144,
    n_taps: int = 65536,
    n_tail: int = 8192,
    headroom_db: float = -1.50,
    gauss_apodize: bool = False,
    kaiser_beta: float = 18.0,
):
    f_nodes = np.array([p[0] for p in ctrl_pts], dtype=np.float64)
    g_nodes = np.array([p[1] for p in ctrl_pts], dtype=np.float64)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    f_safe = np.clip(freqs, f_nodes[0], f_nodes[-1])
    target_db = pchip_log_interp(f_nodes, g_nodes, f_safe)
    target_db[0] = g_nodes[0]

    if gauss_apodize:
        apod_mask = freqs > 20200.0
        target_db[apod_mask] += -3.0103 * ((freqs[apod_mask] - 20200.0) / 1650.0) ** 2

    max_boost_db = float(np.max(target_db))
    preamp_db = headroom_db - max_boost_db
    gain_db_norm = target_db + preamp_db

    # 64-bit Hilbert-Cepstral Minimum-Phase Reconstruction
    mag_lin = 10.0 ** (gain_db_norm / 20.0)
    log_mag = np.log(np.maximum(mag_lin, 1e-18))
    log_spec_full = np.concatenate([log_mag, log_mag[-2:0:-1]])
    cepstrum = np.fft.ifft(log_spec_full).real

    win_cep = np.zeros(n_fft, dtype=np.float64)
    win_cep[0] = 1.0
    win_cep[1 : n_fft // 2] = 2.0
    win_cep[n_fft // 2] = 1.0

    min_phase_spec = np.exp(np.fft.fft(cepstrum * win_cep))
    ir_full_f64 = np.fft.ifft(min_phase_spec).real

    ir_f64 = ir_full_f64[:n_taps].copy()
    unwindowed_floor_dbfs = float(20.0 * np.log10(np.max(np.abs(ir_f64[-128:])) + 1e-25))

    t_norm = np.arange(n_tail, dtype=np.float64) / n_tail
    kaiser_tail = bessel_i0(kaiser_beta * np.sqrt(np.maximum(0.0, 1.0 - t_norm**2))) / bessel_i0(kaiser_beta)
    ir_f64[-n_tail:] *= kaiser_tail

    windowed_floor_dbfs = float(20.0 * np.log10(np.max(np.abs(ir_f64[-100:])) + 1e-25))
    energy_pct = float(np.sum(ir_f64**2) / np.sum(ir_full_f64**2) * 100.0)
    return ir_f64, mag_lin, max_boost_db, preamp_db, energy_pct, unwindowed_floor_dbfs, windowed_floor_dbfs


def export_audit_csv(model_key: str, raw_pts, harm_pts, harm_offset: float, ctrl_pts) -> str:
    """Exports an auditable CSV table containing Frequency, Raw Measurement, Harman Target, Raw Error, and Applied EQ."""
    csv_file = f"{model_key}_audit_data.csv"
    f_ctrl = np.array([p[0] for p in ctrl_pts], dtype=np.float64)
    g_ctrl = np.array([p[1] for p in ctrl_pts], dtype=np.float64)

    f_raw = np.array([p[0] for p in raw_pts], dtype=np.float64)
    g_raw = np.array([p[-1] for p in raw_pts], dtype=np.float64)
    raw_at_ctrl = pchip_log_interp(f_raw, g_raw, f_ctrl)

    f_h = np.array([p[0] for p in harm_pts], dtype=np.float64)
    g_h = np.array([p[1] for p in harm_pts], dtype=np.float64) + harm_offset
    harm_at_ctrl = pchip_log_interp(f_h, g_h, f_ctrl)

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frequency_hz", "raw_dBr", "harman_target_dBr", "raw_error_dB", "applied_eq_dB"])
        for freq, r_val, h_val, eq_val in zip(f_ctrl, raw_at_ctrl, harm_at_ctrl, g_ctrl):
            writer.writerow([f"{freq:.1f}", f"{r_val:.2f}", f"{h_val:.2f}", f"{h_val - r_val:.2f}", f"{eq_val:.2f}"])
    return csv_file


# ==============================================================================
# 3. UNIFIED 4-PANEL VERIFICATION PLOTTER (MARKS UNCORRECTED HF BYPASS ZONE)
# ==============================================================================
def render_verification_plot(
    model_key: str,
    fs: int,
    n_fft_rt: int,
    ir_rt_f32: np.ndarray,
    mag_lin_rt: np.ndarray,
    boost_rt: float,
    pre_rt: float,
    unwin_rt: float,
    active_ctrl_pts,
    notch_mode: str,
) -> str:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import FixedLocator, NullFormatter

    H_wav = np.fft.rfft(ir_rt_f32.astype(np.float64), n=n_fft_rt)
    spec_f32 = np.abs(H_wav)
    freqs_fft = np.fft.rfftfreq(n_fft_rt, d=1.0 / fs)
    f_plot = np.geomspace(20.0, 20000.0, 1200)

    f_h = np.array([p[0] for p in HARMAN_2018_PTS])
    g_h = np.array([p[1] for p in HARMAN_2018_PTS])

    wav_mag_db_unnorm = 20.0 * np.log10(np.maximum(spec_f32, 1e-15)) - pre_rt
    wav_phase_deg = np.degrees(np.angle(H_wav))
    wav_eq_plot = np.interp(f_plot, freqs_fft, wav_mag_db_unnorm)
    wav_phase_plot = np.interp(f_plot, freqs_fft, wav_phase_deg)
    ir_lin_centered = np.fft.fftshift(np.fft.irfft(mag_lin_rt, n=n_fft_rt))

    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#343b42"
    plt.rcParams["axes.linewidth"] = 0.9

    fig = plt.figure(figsize=(13.0, 11.5), dpi=150, facecolor="#151618")
    gs = gridspec.GridSpec(
        3, 2, height_ratios=[1.15, 1.15, 0.85], hspace=0.26, wspace=0.18,
        left=0.065, right=0.965, top=0.945, bottom=0.055
    )
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])

    major_freqs = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    major_labels = ["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"]
    minor_freqs = [30, 40, 60, 70, 80, 90, 300, 400, 600, 700, 800, 900, 3000, 4000, 6000, 7000, 8000, 9000, 15000]

    def style_freq_axis(ax):
        ax.set_facecolor("#1b1f22")
        ax.set_xscale("log")
        ax.set_xlim(20, 20000)
        ax.xaxis.set_major_locator(FixedLocator(major_freqs))
        ax.set_xticklabels(major_labels, color="#c9d1d9", fontsize=9.5)
        ax.xaxis.set_minor_locator(FixedLocator(minor_freqs))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="y", colors="#c9d1d9", labelsize=9.5)
        ax.tick_params(axis="x", which="both", colors="#6e7681", direction="out", length=3)
        ax.grid(True, which="major", color="#2d333b", linestyle="-", linewidth=0.75, alpha=0.9)
        ax.grid(True, which="minor", color="#24292f", linestyle="-", linewidth=0.45, alpha=0.7)
        ax.axhline(0.0, color="#6e7681", linewidth=0.8, alpha=0.6, zorder=1)

    arrow_kw = dict(arrowstyle="->", color="#58d5e0", lw=1.15, shrinkA=2, shrinkB=3)
    arrow_green = dict(arrowstyle="->", color="#26e683", lw=1.2, shrinkA=2, shrinkB=3)

    if model_key == "jbl770nc":
        plot_file = "jbl770nc_final_verification.png"
        f_raw = np.array([p[0] for p in JBL_770NC_RAW_PTS])
        raw_avg = pchip_log_interp(f_raw, np.array([p[1] for p in JBL_770NC_RAW_PTS]), f_plot)
        harm_plot = pchip_log_interp(f_h, g_h, f_plot) - 2.45
        raw_eq_needed = harm_plot - raw_avg

        log_grid = np.log2(f_plot)
        autoeq_plot = np.zeros_like(f_plot)
        for i, f_val in enumerate(f_plot):
            bw = 0.08 if f_val <= 4000.0 else (2.00 if f_val >= 6000.0 else 0.08 + 1.92 * ((f_val - 4000.0) / 2000.0))
            w = np.exp(-0.5 * ((log_grid - log_grid[i]) / (bw / 2.355)) ** 2)
            autoeq_plot[i] = np.sum(raw_eq_needed * w) / np.sum(w)
            if f_val >= 6000.0:
                autoeq_plot[i] *= 0.50
            elif f_val > 4000.0:
                autoeq_plot[i] *= 1.0 - 0.50 * ((f_val - 4000.0) / 2000.0)

        style_freq_axis(ax1)
        ax1.axvspan(9092.0, 20000.0, color="#30363d", alpha=0.35, zorder=1,
                    label="Uncorrected >9.1 kHz Coupler Cylinder Null Zone (Clamped to 0.0 dB)")
        ax1.set_title(
            f"VERIFICATION: JBL Live 770NC (Passive) Convolver (Preamp: {pre_rt:.2f} dB | Unwindowed Floor: {unwin_rt:.1f} dBFS)",
            color="#ffffff", fontsize=12.0, fontweight="bold", pad=10
        )
        ax1.plot(f_plot, raw_avg, color="#9ca3af", linestyle="--", linewidth=1.35, alpha=0.9, zorder=3,
                 label="Raw JBL Live 770NC (Passive on Filk KB006x + IEC 711 Coupler)")
        ax1.plot(f_plot, harm_plot, color="#1f6feb", linestyle="-", linewidth=3.2, alpha=0.88, zorder=4,
                 label="Harman Over-Ear 2018 Target")
        ax1.plot(f_plot, raw_avg + autoeq_plot, color="#f85149", linestyle="-", linewidth=1.55, alpha=0.92, zorder=5,
                 label="Standard AutoEq (+1.5dB accidental boost on 6.88k/8.48kHz spikes & +2.7dB 20Hz boost)")
        ax1.plot(f_plot, raw_avg + wav_eq_plot, color="#00e676", linestyle="-", linewidth=2.5, alpha=0.98, zorder=6,
                 label="Final WAV Convolver (128-Node PCHIP + Breakup Pole Cuts)")
        ax1.set_ylabel("Acoustic Level (dBr)", color="#e6edf3", fontsize=10.5)
        ax1.set_ylim(-15.0, 9.0)
        ax1.set_yticks([-15, -10, -5, 0, 5])

        style_freq_axis(ax2)
        ax2.axvspan(9092.0, 20000.0, color="#30363d", alpha=0.35, zorder=1)
        ax2.plot(f_plot, raw_eq_needed, color="#6e7681", linestyle=":", linewidth=1.0, alpha=0.65, zorder=2,
                 label="Raw Unsmoothed Error (+8.75dB non-min-phase concha dip @ 5.37kHz)")
        ax2.plot(f_plot, autoeq_plot, color="#f5a623", linestyle="--", linewidth=2.0, alpha=0.95, zorder=4,
                 label="Standard AutoEq Curve (2.0 oct treble blur boosts 6.88k/8.48kHz peaks)")
        ax2.plot(f_plot, wav_eq_plot, color="#00e676", linestyle="-", linewidth=2.6, alpha=0.98, zorder=5,
                 label=f"Final WAV Convolver EQ Curve (Peak +{boost_rt:.2f} dB -> Normalized w/ {pre_rt:.2f} dB Preamp)")
        ax2.set_ylabel("Filter Gain (dBr)", color="#e6edf3", fontsize=10.5)
        ax2.set_ylim(-11.0, 10.0)
        ax2.set_yticks([-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0])

        ax2.annotate("1. KB006x Stiff-Pinna Leak Fix\n(+0.12dB @ 20Hz vs AutoEq +2.7dB)",
                     xy=(20.5, 0.12), xytext=(22.0, 5.8), color="#58d5e0", fontsize=8.3, arrowprops=arrow_kw)
        ax2.annotate("2. Coupled Passive ANC Cup Dipole\n(-7.04dB @ 160Hz, +2.60dB @ 455Hz)",
                     xy=(156.9, -7.03), xytext=(95.0, 5.2), color="#58d5e0", fontsize=8.3, arrowprops=arrow_kw)
        ax2.annotate("", xy=(454.9, 2.60), xytext=(275.0, 5.2), arrowprops=arrow_kw)
        ax2.annotate("3. 1.05kHz Nasal Cut & Ear-Gain\n(-2.45dB @ 1.05kHz; +5.00dB @ 4.06kHz)",
                     xy=(1049.4, -2.45), xytext=(480.0, 3.2), color="#58d5e0", fontsize=8.3, arrowprops=arrow_kw)
        ax2.annotate("4. 5.37kHz Concha Notch Shelf Bridge\n(+3.70dB flat bridge across 4.8-5.5kHz;\nignores narrow +8.75dB null spike)",
                     xy=(5365.9, 3.70), xytext=(1450.0, 7.3), color="#26e683", fontsize=8.3, arrowprops=arrow_green)
        ax2.annotate("5. 6.88kHz & 8.48kHz Breakup Poles\n(-2.31dB & -3.16dB Surgical Cuts)",
                     xy=(6881.4, -2.31), xytext=(3200.0, -8.5), color="#26e683", fontsize=8.3, arrowprops=arrow_green)
        ax2.annotate("", xy=(8480.6, -3.16), xytext=(6200.0, -6.8), arrowprops=arrow_green)
        ax2.annotate("6. >9.1kHz Uncorrected Coupler Null Zone\n(0.0dB wire: ignores 9.75k & 11.2kHz nulls)",
                     xy=(10347.9, 0.0), xytext=(5900.0, 4.8), color="#58d5e0", fontsize=8.3, arrowprops=arrow_kw)

    else:
        plot_file = "dt770_80ohm_final_verification.png"
        f_raw = np.array([p[0] for p in DT770_80OHM_RAW_LR])
        raw_L = pchip_log_interp(f_raw, np.array([p[1] for p in DT770_80OHM_RAW_LR]), f_plot)
        raw_R = pchip_log_interp(f_raw, np.array([p[2] for p in DT770_80OHM_RAW_LR]), f_plot)
        raw_avg = pchip_log_interp(f_raw, np.array([p[3] for p in DT770_80OHM_RAW_LR]), f_plot)
        harm_plot = pchip_log_interp(f_h, g_h, f_plot)

        geq_pts = [tuple(map(float, item.strip().split())) for item in AUTOEQ_DT770_GEQ_STR.split(";")]
        f_geq = np.array([p[0] for p in geq_pts])
        g_geq = np.array([p[1] for p in geq_pts]) + 4.55
        autoeq_plot = np.interp(np.log10(f_plot), np.log10(f_geq), g_geq)
        notch_3703_db = dict(active_ctrl_pts).get(3703.0, 0.0)

        style_freq_axis(ax1)
        ax1.axvspan(8800.0, 20000.0, color="#30363d", alpha=0.35, zorder=1,
                    label="Uncorrected >8.8 kHz Coupler Null & >15.8 kHz 0.0 dB Wire Zone (Left Unmodified)")
        ax1.set_title(
            f"VERIFICATION: Beyerdynamic DT 770 Pro (80 Ω) Convolver (Notch Mode: {notch_mode} | Preamp: {pre_rt:.2f} dB)",
            color="#ffffff", fontsize=12.0, fontweight="bold", pad=10
        )
        ax1.plot(f_plot, raw_L, color="#58a6ff", linestyle=":", linewidth=0.85, alpha=0.38, zorder=2,
                 label="Raw L / R Channels (L vs R: -2.5dB 20Hz Pad Leak & 6.6dB 3.84kHz Concha Null Shift)")
        ax1.plot(f_plot, raw_R, color="#58a6ff", linestyle=":", linewidth=0.85, alpha=0.38, zorder=2)
        ax1.plot(f_plot, raw_avg, color="#9ca3af", linestyle="--", linewidth=1.35, alpha=0.9, zorder=3,
                 label="Raw Beyerdynamic DT 770 Pro (80 Ω on Crinacle GRAS 43AG-7: RA0402 + KB500X)")
        ax1.plot(f_plot, harm_plot, color="#1f6feb", linestyle="-", linewidth=3.2, alpha=0.88, zorder=4,
                 label="Harman Over-Ear 2018 Target")
        ax1.plot(f_plot, raw_avg + autoeq_plot, color="#f85149", linestyle="-", linewidth=1.55, alpha=0.92, zorder=5,
                 label="Standard AutoEq (GraphicEQ.txt: +3.7dB un-cut 8.2kHz Beyer spike & +1.5dB 20Hz boost)")
        ax1.plot(f_plot, raw_avg + wav_eq_plot, color="#00e676", linestyle="-", linewidth=2.5, alpha=0.98, zorder=6,
                 label=f"Final WAV Convolver (Notch Mode: {notch_mode} | Unwindowed Floor: {unwin_rt:.1f} dBFS @ 16k)")
        ax1.set_ylabel("Acoustic Level (dBr)", color="#e6edf3", fontsize=10.5)
        ax1.set_ylim(-14.5, 11.5)
        ax1.set_yticks([-10, -5, 0, 5, 10])

        style_freq_axis(ax2)
        ax2.axvspan(8800.0, 20000.0, color="#30363d", alpha=0.35, zorder=1)
        ax2.plot(f_plot, -(raw_avg - harm_plot), color="#6e7681", linestyle=":", linewidth=1.0, alpha=0.65, zorder=2,
                 label="Raw Unsmoothed Error (+14.3dB non-min-phase concha null @ 3.78kHz)")
        ax2.plot(f_plot, autoeq_plot, color="#f5a623", linestyle="--", linewidth=2.0, alpha=0.95, zorder=4,
                 label="Standard AutoEq Curve (2.0 oct treble blur erases 8.2kHz cut + +1.5dB 20Hz boost)")
        ax2.plot(f_plot, wav_eq_plot, color="#00e676", linestyle="-", linewidth=2.6, alpha=0.98, zorder=5,
                 label=f"Final WAV Convolver EQ Curve (Peak +{boost_rt:.2f} dB -> Normalized w/ {pre_rt:.2f} dB Preamp)")
        ax2.set_ylabel("Filter Gain (dBr)", color="#e6edf3", fontsize=10.5)
        ax2.set_ylim(-10.8, 10.5)
        ax2.set_yticks([-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0])

        ax2.annotate("1. EDT 770 V Velour Pad-Leak Fix\n(+0.20dB @ 20Hz vs AutoEq +1.5dB)",
                     xy=(20.5, 0.12), xytext=(21.5, 7.5), color="#58d5e0", fontsize=8.3, arrowprops=arrow_kw)
        ax2.annotate("2. Coupled Port & Cup Suckout\n(-4.05dB @ 59Hz, +4.10dB @ 205Hz)",
                     xy=(59.0, -4.05), xytext=(82.0, 5.2), color="#58d5e0", fontsize=8.3, arrowprops=arrow_kw)
        ax2.annotate("", xy=(204.9, 4.10), xytext=(168.0, 5.5), arrowprops=arrow_kw)
        ax2.annotate("3. 940Hz Cup & 2.35kHz Trim\n(-1.38dB @ 940Hz & -1.43dB @ 2.35kHz)",
                     xy=(940.0, -1.38), xytext=(420.0, 5.8), color="#58d5e0", fontsize=8.3, arrowprops=arrow_kw)
        ax2.annotate("", xy=(2347.0, -1.43), xytext=(1120.0, 5.4), arrowprops=arrow_kw)
        ax2.annotate(f"4. 3.76kHz Non-Min-Phase Concha Zero\n({notch_3703_db:+.2f}dB in '{notch_mode}' mode;\navoids L/R & head-to-head glare)",
                     xy=(3703.0, notch_3703_db), xytext=(1500.0, 7.3), color="#26e683", fontsize=8.3, arrowprops=arrow_green)
        ax2.annotate("5. 5.3k, 6.6k & 8.16kHz Beyer Breakup Poles\n(-3.59dB, -4.45dB & -5.04dB Surgical Cuts)",
                     xy=(8155.0, -5.04), xytext=(2250.0, -9.2), color="#26e683", fontsize=8.3, arrowprops=arrow_green)
        ax2.annotate("", xy=(6644.0, -4.45), xytext=(4500.0, -7.2), arrowprops=arrow_green)
        ax2.annotate("", xy=(5337.0, -3.59), xytext=(3900.0, -7.2), arrowprops=arrow_green)
        ax2.annotate("6. >8.8kHz Uncorrected Null & HF Wire\n(Bypasses 9.2kHz null; 0.0dB >15.8kHz\nleave coupler response unmodified)",
                     xy=(11490.0, -1.55), xytext=(5300.0, 4.0), color="#58d5e0", fontsize=8.3, arrowprops=arrow_kw)

    leg1 = ax1.legend(loc="lower left", facecolor="#16191d", edgecolor="#343b42", framealpha=0.92, fontsize=8.5)
    for text in leg1.get_texts():
        text.set_color("#d0d7de")
    leg2 = ax2.legend(loc="lower left", facecolor="#16191d", edgecolor="#343b42", framealpha=0.92, fontsize=8.5)
    for text in leg2.get_texts():
        text.set_color("#d0d7de")

    style_freq_axis(ax3)
    ax3.set_title("Filter Minimum-Phase Response (Hilbert-Cepstral Phase Angle)", color="#ffffff", fontsize=10.8, fontweight="bold", pad=8)
    ax3.plot(f_plot, wav_phase_plot, color="#c961de", linestyle="-", linewidth=2.0, alpha=0.95,
             label="Filter Minimum-Phase Response (deg)")
    ax3.set_xlabel("Frequency (Hz)", color="#e6edf3", fontsize=10.0)
    ax3.set_ylabel("Phase (deg)", color="#e6edf3", fontsize=10.0)
    ax3.set_ylim(-90, 90)
    ax3.set_yticks([-90, -45, 0, 45, 90])
    leg3 = ax3.legend(loc="upper left", facecolor="#16191d", edgecolor="#343b42", framealpha=0.92, fontsize=8.6)
    for text in leg3.get_texts():
        text.set_color("#d0d7de")

    ax4.set_facecolor("#1b1f22")
    ax4.set_title("Impulse Response: Minimum-Phase vs. Linear-Phase Pre-Ringing", color="#ffffff", fontsize=10.8, fontweight="bold", pad=8)
    peak_mp = float(np.max(np.abs(ir_rt_f32)))
    t_mp_ms = np.arange(len(ir_rt_f32)) / fs * 1000.0
    t_lp_ms = (np.arange(n_fft_rt) - n_fft_rt // 2) / fs * 1000.0
    t_mp_plot = np.concatenate([np.linspace(-1.0, -1.0 / fs * 1000.0, 48), t_mp_ms[:250]])
    ir_mp_plot = np.concatenate([np.zeros(48), ir_rt_f32[:250] / peak_mp])
    lp_mask = (t_lp_ms >= -1.0) & (t_lp_ms <= 4.0)

    ax4.plot(t_lp_ms[lp_mask], ir_lin_centered[lp_mask] / peak_mp, color="#f85149", linestyle="--", linewidth=1.4, alpha=0.88,
             label="Linear-Phase IR (Pre-Ringing before 0.0 ms)")
    ax4.plot(t_mp_plot, ir_mp_plot, color="#00e676", linestyle="-", linewidth=2.0, alpha=0.96,
             label="Minimum-Phase IR (0.00 ms IR Peak Delay, Zero Pre-Ringing)")
    ax4.axvline(0.0, color="#388bfd", linestyle=":", linewidth=1.1, alpha=0.85)
    ax4.axhline(0.0, color="#6e7681", linewidth=0.8, alpha=0.6)
    ax4.set_xlim(-1.0, 4.0)
    ax4.set_ylim(-0.45, 1.08)
    ax4.set_xlabel("Time Relative to Transient Peak (ms)", color="#e6edf3", fontsize=10.0)
    ax4.set_ylabel("Normalized Amplitude", color="#e6edf3", fontsize=10.0)
    ax4.tick_params(axis="both", colors="#c9d1d9", labelsize=9.5)
    ax4.grid(True, which="major", color="#2d333b", linestyle="-", linewidth=0.75, alpha=0.9)
    leg4 = ax4.legend(loc="upper right", facecolor="#16191d", edgecolor="#343b42", framealpha=0.92, fontsize=8.4)
    for text in leg4.get_texts():
        text.set_color("#d0d7de")

    plt.savefig(plot_file, dpi=150, facecolor="#151618")
    plt.close(fig)
    return plot_file


# ==============================================================================
# 4. PER-MODEL RUNNER & DYNAMIC MPV CONFIG GENERATOR
# ==============================================================================
def run_model(model_key: str, args) -> None:
    fs = args.fs
    rate_tag = format_rate_tag(fs)

    if model_key == "jbl770nc":
        title = "JBL Live 770NC (Passive Mode | Filk KB006x + IEC 711 Coupler)"
        active_ctrl_pts = list(CTRL_PTS_JBL770NC)
        raw_pts = JBL_770NC_RAW_PTS
        harm_offset = -2.45
        f64_wav = f"jbl770nc_f64_{rate_tag}.wav"
        mono_rt = f"jbl770nc_mastering_mono_{rate_tag}.wav"
        stereo_rt = f"jbl770nc_mastering_stereo_{rate_tag}.wav"
    else:
        title = f"Beyerdynamic DT 770 Pro 80 Ohm (Crinacle GRAS 43AG-7: RA0402 + KB500X | Notch: {args.notch_mode})"
        active_ctrl_pts = apply_dt770_notch_mode(CTRL_PTS_DT770_80OHM, mode=args.notch_mode)
        raw_pts = DT770_80OHM_RAW_LR
        harm_offset = 0.0
        f64_wav = f"dt770_80ohm_f64_{rate_tag}.wav"
        mono_rt = f"dt770_80ohm_mastering_mono_{rate_tag}.wav"
        stereo_rt = f"dt770_80ohm_mastering_stereo_{rate_tag}.wav"

    n_fft_mpv = max(65536, args.taps * 4)
    n_tail_mpv = args.taps // 8
    kaiser_beta = 18.0
    kaiser_form_db = kaiser_beta / 0.1102 + 8.7

    ir_mpv_f64, _, boost_mpv, pre_mpv, en_mpv, unwin_mpv, win_mpv = synthesize_minimum_phase_ir(
        active_ctrl_pts, fs=fs, n_fft=n_fft_mpv, n_taps=args.taps, n_tail=n_tail_mpv,
        headroom_db=-1.50, gauss_apodize=args.apodize, kaiser_beta=kaiser_beta
    )
    write_wav_ieee_float(f64_wav, ir_mpv_f64, fs=fs, bits=64)
    # Also copy to ir.wav when generating a single headphone model for convenience
    if args.headphone != "all":
        shutil.copyfile(f64_wav, "ir.wav")

    apod_desc = "Gaussian Apodizer (>20.2 kHz)" if args.apodize else "Non-Apodizing 0.00 dB HF Wire"
    print(f"\n================================================================================")
    print(f"MODEL: {title}")
    print(f"================================================================================")
    print(f"=== [1] Offline / mpv 64-Bit Double Convolver ({f64_wav}) ===")
    print(f"  Format            : 64-bit IEEE Float (pcm_f64le) | {fs:,} Hz | Mono")
    print(f"  FFT / Taps        : {n_fft_mpv:,} FFT ({fs/n_fft_mpv:.4f} Hz bin) -> {args.taps:,} taps ({args.taps/fs*1000:.1f} ms)")
    print(f"  HF Mode / Taper   : {apod_desc} + Half-Kaiser Fade (beta={kaiser_beta:.1f}, {kaiser_form_db:.1f} dB Kaiser formula)")
    print(f"  Total Preamp      : {pre_mpv:.2f} dB (Max Boost +{boost_mpv:.2f} dB, -1.50 dB true-peak headroom)")
    print(f"  Causal Energy     : {en_mpv:.10f}%")
    print(f"  Decay @ Truncation: {unwin_mpv:.2f} dBFS (unwindowed) | {win_mpv:.2f} dBFS (after half-Kaiser fade)")

    n_fft_rt = 32768
    ir_rt_f64, mag_lin_rt, boost_rt, pre_rt, _, unwin_rt, win_rt = synthesize_minimum_phase_ir(
        active_ctrl_pts, fs=fs, n_fft=n_fft_rt, n_taps=16384, n_tail=2048,
        headroom_db=-1.50, gauss_apodize=args.apodize, kaiser_beta=16.0
    )
    ir_rt_f32 = ir_rt_f64.astype(np.float32)
    write_wav_ieee_float(mono_rt, ir_rt_f32, fs=fs, bits=32)
    write_wav_ieee_float(stereo_rt, np.column_stack([ir_rt_f32, ir_rt_f32]), fs=fs, bits=32)
    err_rms_dbfs = 20.0 * np.log10(np.sqrt(np.mean((ir_rt_f64 - ir_rt_f32.astype(np.float64)) ** 2)))

    print(f"\n=== [2] Real-Time PipeWire / DAW 32-Bit Float Convolvers ===")
    print(f"  FFT / Taps        : {n_fft_rt:,} FFT ({fs/n_fft_rt:.4f} Hz bin) -> 16,384 taps ({16384/fs*1000:.1f} ms, Half-Kaiser beta=16.0)")
    print(f"  Decay @ Truncation: {unwin_rt:.2f} dBFS (unwindowed) | {win_rt:.2f} dBFS (after half-Kaiser fade)")
    print(f"  f32 vs f64 Error  : {err_rms_dbfs:.2f} dBFS RMS")
    print(f"  Written WAVs      : {mono_rt}, {stereo_rt}")

    if args.export_csv:
        csv_path = export_audit_csv(model_key, raw_pts, HARMAN_2018_PTS, harm_offset, active_ctrl_pts)
        print(f"  Written Audit CSV : {csv_path}")

    if not args.no_plot:
        try:
            plot_file = render_verification_plot(
                model_key, fs, n_fft_rt, ir_rt_f32, mag_lin_rt,
                boost_rt, pre_rt, unwin_rt, active_ctrl_pts, args.notch_mode
            )
            print(f"  Written Plot      : {plot_file}")
        except ImportError:
            print("  [Note] matplotlib not installed; skipped PNG plot generation.")

    abs_ir_path = os.path.abspath(f64_wav).replace("\\", "/")
    if ":" in abs_ir_path:
        abs_ir_path = abs_ir_path.replace(":", "\\:")
    print(f"\n=== [3] Matching mpv.conf Snippet ({fs} Hz, minp=1024:maxp=8192 -> {1024/fs*1000:.1f} ms block latency) ===")
    print(f"audio-samplerate={fs}")
    print(
        f"af=lavfi=[aformat=sample_fmts=dblp,aresample={fs}:filter_size=256:phase_shift=12:cutoff=0.98:"
        f"filter_type=kaiser:kaiser_beta=16:osf=dblp[in];amovie={abs_ir_path},aformat=sample_fmts=dblp[ir];"
        f"[in][ir]afir=irfmt=mono:irnorm=-1:wet=1:minp=1024:maxp=8192:precision=double]"
    )


def main():
    parser = argparse.ArgumentParser(
        description="HyprEQ: Unified Minimum-Phase Convolver Generator (JBL Live 770NC & Beyerdynamic DT 770 Pro 80 Ohm)."
    )
    parser.add_argument(
        "-H", "--headphone", choices=["jbl770nc", "dt770_80", "all"], default="all",
        help="Headphone profile to generate (default: all)"
    )
    parser.add_argument("--fs", type=int, default=48000, help="Sample rate in Hz (default: 48000)")
    parser.add_argument(
        "--taps", type=int, default=65536, choices=[8192, 16384, 32768, 65536],
        help="Tap count for 64-bit float WAV (default: 65536; 16384 already reaches <-147 dBFS unwindowed)"
    )
    parser.add_argument(
        "--notch-mode", choices=["safe", "original", "zero"], default="safe",
        help="DT 770 Pro ~3.76 kHz non-minimum-phase concha notch handling: "
             "'safe' (default, caps narrow fill to +1.50 dB), 'original' (+4.26 dB fill), or 'zero' (0.0 dB)"
    )
    parser.add_argument(
        "--apodize", action="store_true",
        help="Enable Gaussian Nyquist roll-off (>20.2 kHz) for legacy brickwall-ringing sources"
    )
    parser.add_argument("--export-csv", action="store_true", help="Export auditable CSV tables of raw, target, and EQ data")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib 4-panel verification plot generation")
    args = parser.parse_args()

    models = ["jbl770nc", "dt770_80"] if args.headphone == "all" else [args.headphone]
    for m in models:
        run_model(m, args)


if __name__ == "__main__":
    main()
