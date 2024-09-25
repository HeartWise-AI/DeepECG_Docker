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


class Mode:
    FULL_RUN = "full_run"
    ANALYSIS = "analysis"
    PREPROCESSING = "preprocessing"