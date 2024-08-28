
import pandas as pd


# df_general = pd.read_parquet('ECG_MHI_cat_data.parquet')

df_test = pd.read_parquet("/workspace/Llava-ECG/hw_QA_generator/parquet_files/ECG_data_validatedByMD_test.parquet")

print(df_test.columns.tolist())



