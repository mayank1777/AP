import pandas as pd

PATIENT_PARA_FILE = '/home/aasmaan/simglucose/simglucose/params/vpatient_params.csv'

try:
    patient_params = pd.read_csv(PATIENT_PARA_FILE , encoding='utf-8')
    print(patient_params.head())
except Exception as e:
    print(f"Error reading the file: {e}")
