
from models.efficientnet_wrapper import (
    EfficientNetV2_77_classes, 
    EfficientNetV2_AFIB_5Y, 
    EfficientNetV2_LVEF_Equal_Under_40, 
    EfficientNetV2_LVEF_Under_50
)
from models.wcr_ecg_transformer import (
    WCR_77_classes, 
    WCR_AFIB_5Y, 
    WCR_LVEF_Equal_Under_40, 
    WCR_LVEF_Under_50
)
from models.bert_classifier import BertClassifier_En_Fr

DIAGNOSIS_TO_FILE_COLUMNS = {
    'ecg_machine_diagnosis': '77_classes_ecg_file_name',
    'afib_5y': 'afib_ecg_file_name',
    'lvef_40': 'lvef_40_ecg_file_name',
    'lvef_50': 'lvef_50_ecg_file_name'
}

MODEL_MAPPING = {
    'ecg_machine_diagnosis': {
        'efficientnet': EfficientNetV2_77_classes.name,
        'wcr': WCR_77_classes.name,
        'bert': BertClassifier_En_Fr.name
    },
    'afib_5y': {
        'efficientnet': EfficientNetV2_AFIB_5Y.name,
        'wcr': WCR_AFIB_5Y.name,
        'bert': None
    },
    'lvef_40': {
        'efficientnet': EfficientNetV2_LVEF_Equal_Under_40.name,
        'wcr': WCR_LVEF_Equal_Under_40.name,
        'bert': None
    },
    'lvef_50': {
        'efficientnet': EfficientNetV2_LVEF_Under_50.name,
        'wcr': WCR_LVEF_Under_50.name,
        'bert': None
    }
}

ECG_CATEGORIES = {
    "Rhythm Disorders": [
        "Ventricular tachycardia",
        "Bradycardia",
        "Brugada",
        "Wolff-Parkinson-White (Pre-excitation syndrome)",
        "Atrial flutter",
        "Ectopic atrial rhythm (< 100 BPM)",
        "Atrial tachycardia (>= 100 BPM)",
        "Sinusal",
        "Ventricular Rhythm",
        "Supraventricular tachycardia",
        "Junctional rhythm",
        "Regular",
        "Regularly irregular",
        "Irregularly irregular",
        "Afib",
        "Premature ventricular complex",
        "Premature atrial complex"
    ],
    "Conduction Disorder": [
        "Left anterior fascicular block",
        "Delta wave",
        "2nd degree AV block - mobitz 2",
        "Left bundle branch block",
        "Right bundle branch block",
        "Left axis deviation",
        "Atrial paced",
        "Right axis deviation",
        "Left posterior fascicular block",
        "1st degree AV block",
        "Right superior axis",
        "Nonspecific intraventricular conduction delay",
        "Third Degree AV Block",
        "2nd degree AV block - mobitz 1",
        "Prolonged QT",
        "U wave",
        "LV pacing",
        "Ventricular paced"
    ],
    "Enlargement of the heart chambers": [
        "Bi-atrial enlargement",
        "Left atrial enlargement",
        "Right atrial enlargement",
        "Left ventricular hypertrophy",
        "Right ventricular hypertrophy"
    ],
    "Pericarditis": [
        "Acute pericarditis"
    ],
    "Infarction or ischemia": [
        "Q wave (septal- V1-V2)",
        "ST elevation (anterior - V3-V4)",
        "Q wave (posterior - V7-V9)",
        "Q wave (inferior - II, III, aVF)",
        "Q wave (anterior - V3-V4)",
        "ST elevation (lateral - I, aVL, V5-V6)",
        "Q wave (lateral- I, aVL, V5-V6)",
        "ST depression (lateral - I, avL, V5-V6)",
        "Acute MI",
        "ST elevation (septal - V1-V2)",
        "ST elevation (inferior - II, III, aVF)",
        "ST elevation (posterior - V7-V8-V9)",
        "ST depression (inferior - II, III, aVF)",
        "ST depression (anterior - V3-V4)"
    ],
    "Other diagnoses": [
        "ST downslopping",
        "ST depression (septal- V1-V2)",
        "R/S ratio in V1-V2 >1",
        "RV1 + SV6 > 11 mm",
        "Polymorph",
        "rSR' in V1-V2",
        "QRS complex negative in III",
        "qRS in V5-V6-I, aVL",
        "QS complex in V1-V2-V3",
        "R complex in V5-V6",
        "RaVL > 11 mm",
        "T wave inversion (septal- V1-V2)",
        "SV1 + RV5 or RV6 > 35 mm",
        "T wave inversion (inferior - II, III, aVF)",
        "Monomorph",
        "T wave inversion (anterior - V3-V4)",
        "T wave inversion (lateral -I, aVL, V5-V6)",
        "Low voltage",
        "Lead misplacement",
        "ST depression (anterior - V3-V4)",
        "Early repolarization",
        "ST upslopping",
        "no_qrs"
    ]
}

ECG_PATTERNS = [
    "Sinusal",
    "Regular",
    "Monomorph",
    "QS complex in V1-V2-V3",
    "R complex in V5-V6",
    "T wave inversion (inferior - II, III, aVF)",
    "Left bundle branch block",
    "RaVL > 11 mm",
    "SV1 + RV5 or RV6 > 35 mm",
    "T wave inversion (lateral -I, aVL, V5-V6)",
    "T wave inversion (anterior - V3-V4)",
    "Left axis deviation",
    "Left ventricular hypertrophy",
    "Bradycardia",
    "Q wave (inferior - II, III, aVF)",
    "Afib",
    "Irregularly irregular",
    "Atrial tachycardia (>= 100 BPM)",
    "Nonspecific intraventricular conduction delay",
    "Premature ventricular complex",
    "Polymorph",
    "T wave inversion (septal- V1-V2)",
    "Right bundle branch block",
    "Ventricular paced",
    "ST elevation (anterior - V3-V4)",
    "ST elevation (septal - V1-V2)",
    "1st degree AV block",
    "Premature atrial complex",
    "Atrial flutter",
    "rSR' in V1-V2",
    "qRS in V5-V6-I, aVL",
    "Left anterior fascicular block",
    "Right axis deviation",
    "2nd degree AV block - mobitz 1",
    "ST depression (inferior - II, III, aVF)",
    "Acute pericarditis",
    "ST elevation (inferior - II, III, aVF)",
    "Low voltage",
    "Regularly irregular",
    "Junctional rhythm",
    "Left atrial enlargement",
    "ST elevation (lateral - I, aVL, V5-V6)",
    "Atrial paced",
    "Right ventricular hypertrophy",
    "Delta wave",
    "Wolff-Parkinson-White (Pre-excitation syndrome)",
    "Prolonged QT",
    "ST depression (anterior - V3-V4)",
    "QRS complex negative in III",
    "Q wave (lateral- I, aVL, V5-V6)",
    "Supraventricular tachycardia",
    "ST downslopping",
    "ST depression (lateral - I, avL, V5-V6)",
    "2nd degree AV block - mobitz 2",
    "U wave",
    "R/S ratio in V1-V2 >1",
    "RV1 + SV6 > 11 mm",
    "Left posterior fascicular block",
    "Right atrial enlargement",
    "ST depression (septal- V1-V2)",
    "Q wave (septal- V1-V2)",
    "Q wave (anterior - V3-V4)",
    "ST upslopping",
    "Right superior axis",
    "Ventricular tachycardia",
    "ST elevation (posterior - V7-V8-V9)",
    "Ectopic atrial rhythm (< 100 BPM)",
    "Lead misplacement",
    "Third Degree AV Block",
    "Acute MI",
    "Early repolarization",
    "Q wave (posterior - V7-V9)",
    "Bi-atrial enlargement",
    "LV pacing",
    "Brugada",
    "Ventricular Rhythm",
    "no_qrs"
]

BERT_THRESHOLDS = {
    "Rhythm Disorders": {
        "macro_threshold": 0.34,
        "micro_threshold": 0.45
    },
    "Conduction Disorder": {
        "macro_threshold": 0.43,
        "micro_threshold": 0.41
    },
    "Enlargement of the heart chambers": {
        "macro_threshold": 0.38,
        "micro_threshold": 0.38
    },
    "Pericarditis": {
        "macro_threshold": 0.38,
        "micro_threshold": 0.38
    },
    "Infarction or ischemia": {
        "macro_threshold": 0.4,
        "micro_threshold": 0.4
    },
    "Other diagnoses": {
        "macro_threshold": 0.52,
        "micro_threshold": 0.56
    },
    "Sinusal": {
        "threshold": 0.43
    },
    "Regular": {
        "threshold": 0.48
    },
    "Monomorph": {
        "threshold": 0.51
    },
    "QS complex in V1-V2-V3": {
        "threshold": 0.57
    },
    "R complex in V5-V6": {
        "threshold": 0.4
    },
    "T wave inversion (inferior - II, III, aVF)": {
        "threshold": 0.6
    },
    "Left bundle branch block": {
        "threshold": 0.31
    },
    "RaVL > 11 mm": {
        "threshold": 0.65
    },
    "SV1 + RV5 or RV6 > 35 mm": {
        "threshold": 0.48
    },
    "T wave inversion (lateral -I, aVL, V5-V6)": {
        "threshold": 0.59
    },
    "T wave inversion (anterior - V3-V4)": {
        "threshold": 0.58
    },
    "Left axis deviation": {
        "threshold": 0.46
    },
    "Left ventricular hypertrophy": {
        "threshold": 0.38
    },
    "Bradycardia": {
        "threshold": 0.57
    },
    "Q wave (inferior - II, III, aVF)": {
        "threshold": 0.46
    },
    "Afib": {
        "threshold": 0.46
    },
    "Irregularly irregular": {
        "threshold": 0.58
    },
    "Atrial tachycardia (>= 100 BPM)": {
        "threshold": 0.39
    },
    "Nonspecific intraventricular conduction delay": {
        "threshold": 0.34
    },
    "Premature ventricular complex": {
        "threshold": 0.34
    },
    "Polymorph": {
        "threshold": 0.61
    },
    "T wave inversion (septal- V1-V2)": {
        "threshold": 0.65
    },
    "Right bundle branch block": {
        "threshold": 0.38
    },
    "Ventricular paced": {
        "threshold": 0.34
    },
    "ST elevation (anterior - V3-V4)": {
        "threshold": 0.46
    },
    "ST elevation (septal - V1-V2)": {
        "threshold": 0.48
    },
    "1st degree AV block": {
        "threshold": 0.31
    },
    "Premature atrial complex": {
        "threshold": 0.33
    },
    "Atrial flutter": {
        "threshold": 0.44
    },
    "rSR' in V1-V2": {
        "threshold": 0.56
    },
    "qRS in V5-V6-I, aVL": {
        "threshold": 0.63
    },
    "Left anterior fascicular block": {
        "threshold": 0.45
    },
    "Right axis deviation": {
        "threshold": 0.49
    },
    "2nd degree AV block - mobitz 1": {
        "threshold": 0.51
    },
    "ST depression (inferior - II, III, aVF)": {
        "threshold": 0.51
    },
    "Acute pericarditis": {
        "threshold": 0.38
    },
    "ST elevation (inferior - II, III, aVF)": {
        "threshold": 0.36
    },
    "Low voltage": {
        "threshold": 0.5
    },
    "Regularly irregular": {
        "threshold": 0.58
    },
    "Junctional rhythm": {
        "threshold": 0.43
    },
    "Left atrial enlargement": {
        "threshold": 0.52
    },
    "ST elevation (lateral - I, aVL, V5-V6)": {
        "threshold": 0.46
    },
    "Atrial paced": {
        "threshold": 0.42
    },
    "Right ventricular hypertrophy": {
        "threshold": 0.38
    },
    "Delta wave": {
        "threshold": 0.3
    },
    "Wolff-Parkinson-White (Pre-excitation syndrome)": {
        "threshold": 0.28
    },
    "Prolonged QT": {
        "threshold": 0.4
    },
    "ST depression (anterior - V3-V4)": {
        "threshold": 0.48
    },
    "QRS complex negative in III": {
        "threshold": 0.56
    },
    "Q wave (lateral- I, aVL, V5-V6)": {
        "threshold": 0.51
    },
    "Supraventricular tachycardia": {
        "threshold": 0.42
    },
    "ST downslopping": {
        "threshold": 0.37
    },
    "ST depression (lateral - I, avL, V5-V6)": {
        "threshold": 0.51
    },
    "2nd degree AV block - mobitz 2": {
        "threshold": 0.37
    },
    "U wave": {
        "threshold": 0.26
    },
    "R/S ratio in V1-V2 >1": {
        "threshold": 0.52
    },
    "RV1 + SV6 > 11 mm": {
        "threshold": 0.53
    },
    "Left posterior fascicular block": {
        "threshold": 0.35
    },
    "Right atrial enlargement": {
        "threshold": 0.26
    },
    "ST depression (septal- V1-V2)": {
        "threshold": 0.41
    },
    "Q wave (septal- V1-V2)": {
        "threshold": 0.51
    },
    "Q wave (anterior - V3-V4)": {
        "threshold": 0.37
    },
    "ST upslopping": {
        "threshold": 0.39
    },
    "Right superior axis": {
        "threshold": 0.43
    },
    "Ventricular tachycardia": {
        "threshold": 0.35
    },
    "ST elevation (posterior - V7-V8-V9)": {
        "threshold": 0.4
    },
    "Ectopic atrial rhythm (< 100 BPM)": {
        "threshold": 0.4
    },
    "Lead misplacement": {
        "threshold": 0.32
    },
    "Third Degree AV Block": {
        "threshold": 0.37
    },
    "Acute MI": {
        "threshold": 0.38
    },
    "Early repolarization": {
        "threshold": 0.4
    },
    "Q wave (posterior - V7-V9)": {
        "threshold": 0.34
    },
    "Bi-atrial enlargement": {
        "threshold": 0.29
    },
    "LV pacing": {
        "threshold": 0.28
    },
    "Brugada": {
        "threshold": 0.22
    },
    "Ventricular Rhythm": {
        "threshold": 0.33
    },
    "no_qrs": {
        "threshold": 0.27
    }
}

WCR_THRESHOLDS = {
    "Rhythm Disorders": {
        "macro_threshold": 0.12457835674285889,
        "micro_threshold": 0.12457835674285889
    },
    "Conduction Disorder": {
        "macro_threshold": 0.026223255321383476,
        "micro_threshold": 0.026223255321383476
    },
    "Enlargement of the heart chambers": {
        "macro_threshold": 0.023761164397001266,
        "micro_threshold": 0.023761164397001266
    },
    "Pericarditis": {
        "macro_threshold": 0.004548099357634783,
        "micro_threshold": 0.004548099357634783
    },
    "Infarction or ischemia": {
        "macro_threshold": 0.019382962957024574,
        "micro_threshold": 0.019382962957024574
    },
    "Other diagnoses": {
        "macro_threshold": 0.07094576954841614,
        "micro_threshold": 0.07094576954841614
    },
    "Sinusal": {
        "threshold": 0.8075482249259949
    },
    "Regular": {
        "threshold": 0.8469119071960449
    },
    "Monomorph": {
        "threshold": 0.9181361794471741
    },
    "QS complex in V1-V2-V3": {
        "threshold": 0.016304679214954376
    },
    "R complex in V5-V6": {
        "threshold": 0.0392032228410244
    },
    "T wave inversion (inferior - II, III, aVF)": {
        "threshold": 0.15714240074157715
    },
    "Left bundle branch block": {
        "threshold": 0.018835967406630516
    },
    "RaVL > 11 mm": {
        "threshold": 0.03399272263050079
    },
    "SV1 + RV5 or RV6 > 35 mm": {
        "threshold": 0.0222533717751503
    },
    "T wave inversion (lateral -I, aVL, V5-V6)": {
        "threshold": 0.13455051183700562
    },
    "T wave inversion (anterior - V3-V4)": {
        "threshold": 0.0724065974354744
    },
    "Left axis deviation": {
        "threshold": 0.15760254859924316
    },
    "Left ventricular hypertrophy": {
        "threshold": 0.08567972481250763
    },
    "Bradycardia": {
        "threshold": 0.19554725289344788
    },
    "Q wave (inferior - II, III, aVF)": {
        "threshold": 0.09673208743333817
    },
    "Afib": {
        "threshold": 0.1037951335310936
    },
    "Irregularly irregular": {
        "threshold": 0.10453741252422333
    },
    "Atrial tachycardia (>= 100 BPM)": {
        "threshold": 0.04391631856560707
    },
    "Nonspecific intraventricular conduction delay": {
        "threshold": 0.029311716556549072
    },
    "Premature ventricular complex": {
        "threshold": 0.060315560549497604
    },
    "Polymorph": {
        "threshold": 0.01651196926832199
    },
    "T wave inversion (septal- V1-V2)": {
        "threshold": 0.07128537446260452
    },
    "Right bundle branch block": {
        "threshold": 0.060605697333812714
    },
    "Ventricular paced": {
        "threshold": 0.04323868080973625
    },
    "ST elevation (anterior - V3-V4)": {
        "threshold": 0.009488987736403942
    },
    "ST elevation (septal - V1-V2)": {
        "threshold": 0.007370146457105875
    },
    "1st degree AV block": {
        "threshold": 0.054165828973054886
    },
    "Premature atrial complex": {
        "threshold": 0.02900737151503563
    },
    "Atrial flutter": {
        "threshold": 0.012458192184567451
    },
    "rSR' in V1-V2": {
        "threshold": 0.024231072515249252
    },
    "qRS in V5-V6-I, aVL": {
        "threshold": 0.016562914475798607
    },
    "Left anterior fascicular block": {
        "threshold": 0.037277620285749435
    },
    "Right axis deviation": {
        "threshold": 0.010496498085558414
    },
    "2nd degree AV block - mobitz 1": {
        "threshold": 0.006294497288763523
    },
    "ST depression (inferior - II, III, aVF)": {
        "threshold": 0.010292482562363148
    },
    "Acute pericarditis": {
        "threshold": 0.004548099357634783
    },
    "ST elevation (inferior - II, III, aVF)": {
        "threshold": 0.00899659376591444
    },
    "Low voltage": {
        "threshold": 0.05578191205859184
    },
    "Regularly irregular": {
        "threshold": 0.051944803446531296
    },
    "Junctional rhythm": {
        "threshold": 0.01158860418945551
    },
    "Left atrial enlargement": {
        "threshold": 0.025674456730484962
    },
    "ST elevation (lateral - I, aVL, V5-V6)": {
        "threshold": 0.006438542157411575
    },
    "Atrial paced": {
        "threshold": 0.016819339245557785
    },
    "Right ventricular hypertrophy": {
        "threshold": 0.0056317038834095
    },
    "Delta wave": {
        "threshold": 0.0004455529560800642
    },
    "Wolff-Parkinson-White (Pre-excitation syndrome)": {
        "threshold": 0.0005034709465689957
    },
    "Prolonged QT": {
        "threshold": 0.046002209186553955
    },
    "ST depression (anterior - V3-V4)": {
        "threshold": 0.006790333427488804
    },
    "QRS complex negative in III": {
        "threshold": 0.09670284390449524
    },
    "Q wave (lateral- I, aVL, V5-V6)": {
        "threshold": 0.01590864546597004
    },
    "Supraventricular tachycardia": {
        "threshold": 0.007044760510325432
    },
    "ST downslopping": {
        "threshold": 0.08333136141300201
    },
    "ST depression (lateral - I, avL, V5-V6)": {
        "threshold": 0.019376173615455627
    },
    "2nd degree AV block - mobitz 2": {
        "threshold": 0.00933077186346054
    },
    "U wave": {
        "threshold": 0.000755671935621649
    },
    "R/S ratio in V1-V2 >1": {
        "threshold": 0.01122095063328743
    },
    "RV1 + SV6 > 11 mm": {
        "threshold": 0.005522996187210083
    },
    "Left posterior fascicular block": {
        "threshold": 0.0036026940215379
    },
    "Right atrial enlargement": {
        "threshold": 0.002089146990329027
    },
    "ST depression (septal- V1-V2)": {
        "threshold": 0.001671362086199224
    },
    "Q wave (septal- V1-V2)": {
        "threshold": 0.03893815353512764
    },
    "Q wave (anterior - V3-V4)": {
        "threshold": 0.05494214594364166
    },
    "ST upslopping": {
        "threshold": 0.009845602326095104
    },
    "Right superior axis": {
        "threshold": 0.0002490754995960742
    },
    "Ventricular tachycardia": {
        "threshold": 0.0002493064384907484
    },
    "ST elevation (posterior - V7-V8-V9)": {
        "threshold": 0.0013480085181072354
    },
    "Ectopic atrial rhythm (< 100 BPM)": {
        "threshold": 0.004058691672980785
    },
    "Lead misplacement": {
        "threshold": 2.4045009922701865e-05
    },
    "Third Degree AV Block": {
        "threshold": 0.002749016275629401
    },
    "Acute MI": {
        "threshold": 0.013021176680922508
    },
    "Early repolarization": {
        "threshold": 0.004803062882274389
    },
    "Q wave (posterior - V7-V9)": {
        "threshold": 0.002231500344350934
    },
    "Bi-atrial enlargement": {
        "threshold": 0.0013420595787465572
    },
    "LV pacing": {
        "threshold": 0.0006026728078722954
    },
    "Brugada": {
        "threshold": None
    },
    "Ventricular Rhythm": {
        "threshold": 0.010783462785184383
    },
    "no_qrs": {
        "threshold": 0.00020396339823491871
    }
}

WCR_COLUMN_CONVERSION = [15, 23, 16, 1, 57, 63, 73, 41, 39, 36, 2, 29, 30, 65, 34, 12, 55, 56, 21, 8, 42, 71,
                         37, 50, 13, 38, 46, 24, 49, 9, 66, 26, 40, 4, 22, 0, 11, 74, 64, 7, 76, 58, 33, 70, 17, 6, 28,
                         69, 44, 61, 32, 72, 45, 25, 75, 18, 14, 5, 3, 31, 27, 67, 62, 10, 43, 51, 52, 47, 19, 68, 53, 48, 60, 20, 59, 54, 35]

PTBXL_POWER_RATIO = 3.003154

class Mode:
    FULL_RUN = "full_run"
    ANALYSIS = "analysis"
    PREPROCESSING = "preprocessing"