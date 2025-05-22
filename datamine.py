import pandas as pd

file_path = "C:/TDPython/AeroportProject/customer_responses_2025.xlsx"


# Загрузка данных
df = pd.read_excel(file_path)

# Посмотреть первые строки таблицы
print(df.head())

