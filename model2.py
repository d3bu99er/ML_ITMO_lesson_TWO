import pandas as pd
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier  # раскомментируйте для RF
from scipy.sparse import hstack


def shannon_entropy(s: str) -> float:
    """Вычисляет энтропию Шеннона для строки."""
    if not s or len(s) == 0:
        return 0.0
    entropy = 0.0
    for char in set(s):
        p = s.count(char) / len(s)
        if p > 0:
            entropy += -p * math.log(p, 2)
    return entropy


def extract_all_features(series):
    """Извлекает ручные признаки: сигнатуры шеллов + признаки обфускации."""
    s = series.astype(str)
    df = pd.DataFrame()

    # --- Сигнатурные признаки (реверс/бинд шеллы) ---
    df['has_dev_tcp'] = s.str.contains(r'/dev/tcp', case=False, na=False)
    df['has_nc'] = s.str.contains(r'\bnc\b', case=False, na=False)
    df['has_socat'] = s.str.contains(r'\bsocat\b', case=False, na=False)
    df['has_mkfifo'] = s.str.contains(r'mkfifo', case=False, na=False)
    df['has_bash_i'] = s.str.contains(r'bash\s+-i', case=False, na=False)
    df['has_sh_pipe'] = s.str.contains(r'\|.*sh\b', case=False, na=False)
    df['has_perl'] = s.str.contains(r'\bperl\b', case=False, na=False)
    df['has_python'] = s.str.contains(r'\bpython[23]?\b', case=False, na=False)
    df['has_php'] = s.str.contains(r'\bphp\b', case=False, na=False)
    df['has_ruby'] = s.str.contains(r'\bruby\b', case=False, na=False)
    df['has_telnet'] = s.str.contains(r'\btelnet\b', case=False, na=False)
    df['has_awk'] = s.str.contains(r'\bawk\b', case=False, na=False)
    df['has_gawk'] = s.str.contains(r'\bgawk\b', case=False, na=False)
    df['has_eval'] = s.str.contains(r'eval\s*\(', case=False, na=False)
    df['has_backtick'] = s.str.contains(r'`.*`', case=False, na=False)
    df['has_dollar_paren'] = s.str.contains(r'\$\([^)]*\)', case=False, na=False)
    df['has_base64_cmd'] = s.str.contains(r'base64\s+[-\w]*\s*-\w*d', case=False, na=False)
    df['has_echo_pipe'] = s.str.contains(r'echo\s+["\'][A-Za-z0-9+/=]{20,}["\']\s*\|', case=False, na=False)

    # --- Признаки обфускации ---
    df['len'] = s.str.len()
    df['entropy'] = s.apply(shannon_entropy)
    base64_like = (
        s.str.fullmatch(r'[A-Za-z0-9+/]*={0,2}') &
        (s.str.len() % 4 == 0) &
        (s.str.len() >= 20)
    )
    df['is_base64_like'] = base64_like.fillna(False)
    df['only_base64_chars'] = s.str.fullmatch(r'[A-Za-z0-9+/=]*').fillna(False)
    df['no_spaces_or_ops'] = ~s.str.contains(r'[\s|;&$><`()\\]', na=False)
    df['high_entropy'] = df['entropy'] > 4.5
    df['long_obfuscated'] = (df['len'] > 60) & df['high_entropy']

    return df.fillna(0).astype(float)


# -----------------------------
# 1. Загрузка данных
# -----------------------------
train_df = pd.read_parquet('train_cleaned.parquet')
test_df = pd.read_parquet('test.parquet')

# -----------------------------
# 2. Обработка пропущенных значений
# -----------------------------
train_df['shell'] = train_df['shell'].fillna('')
test_df['shell'] = test_df['shell'].fillna('')

# -----------------------------
# 3. Векторизация текста (символьные n-граммы)
# -----------------------------
vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(2, 6),
    max_features=15000,
    lowercase=False
)

X_train_text = vectorizer.fit_transform(train_df['shell'])
X_test_text = vectorizer.transform(test_df['shell'])

# -----------------------------
# 4. Извлечение ручных признаков
# -----------------------------
X_train_manual = extract_all_features(train_df['shell'])
X_test_manual = extract_all_features(test_df['shell'])

# -----------------------------
# 5. Объединение признаков
# -----------------------------
X_train = hstack([X_train_text, X_train_manual.values])
X_test = hstack([X_test_text, X_test_manual.values])

# -----------------------------
# 6. Обучение модели
# -----------------------------
# Вариант 1: LogisticRegression
#model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

# Вариант 2: RandomForest
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    class_weight='balanced',
    min_samples_split=20,
    min_samples_leaf=10,
    max_features=0.4,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, train_df['label'])

# -----------------------------
# 7. Определение id для submission
# -----------------------------
if 'id' in test_df.columns:
    ids = test_df['id']
elif 'Id' in test_df.columns:
    ids = test_df['Id']
elif 'ID' in test_df.columns:
    ids = test_df['ID']
else:
    print("Колонка 'id' не найдена. Используем индекс как id.")
    ids = test_df.index

# -----------------------------
# 8. Предсказание и сохранение
# -----------------------------
test_pred = model.predict(X_test)

submission = pd.DataFrame({
    'id': ids,
    'label': test_pred
})

submission.to_csv('submission.csv', index=False)
print("Решение сохранено в submission.csv")
