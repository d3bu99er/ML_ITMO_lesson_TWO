import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

# -----------------------------
# 1. Загрузка данных
# -----------------------------
train_df = pd.read_parquet('train.parquet')
test_df = pd.read_parquet('test.parquet')

# -----------------------------
# 2. Обработка пропущенных значений
# -----------------------------
train_df['shell'] = train_df['shell'].fillna('')
test_df['shell'] = test_df['shell'].fillna('')

# -----------------------------
# 3. Извлечение ручных признаков
# -----------------------------
def extract_manual_features(series):
    s = series.astype(str)  # дополнительно гарантируем строку
    features = pd.DataFrame({
        'has_dev_tcp': s.str.contains(r'/dev/tcp', case=False, na=False),
        'has_nc': s.str.contains(r'\bnc\b', case=False, na=False),
        'has_socat': s.str.contains(r'\bsocat\b', case=False, na=False),
        'has_mkfifo': s.str.contains(r'mkfifo', case=False, na=False),
        'has_bash_i': s.str.contains(r'bash\s+-i', case=False, na=False),
        'has_sh_pipe': s.str.contains(r'\|.*sh\b', case=False, na=False),
        'has_perl': s.str.contains(r'\bperl\b', case=False, na=False),
        'has_python': s.str.contains(r'\bpython[23]?\b', case=False, na=False),
        'has_php': s.str.contains(r'\bphp\b', case=False, na=False),
        'has_ruby': s.str.contains(r'\bruby\b', case=False, na=False),
        'has_telnet': s.str.contains(r'\btelnet\b', case=False, na=False),
        'has_awk': s.str.contains(r'\bawk\b', case=False, na=False),
        'has_gawk': s.str.contains(r'\bgawk\b', case=False, na=False),
        'has_base64': s.str.contains(r'base64', case=False, na=False),
        'has_eval': s.str.contains(r'eval\s*\(', case=False, na=False),
        'has_backtick': s.str.contains(r'`.*`', case=False, na=False),
        'has_dollar_paren': s.str.contains(r'\$\([^)]*\)', case=False, na=False),
        'len': s.str.len(),
    }).astype(int)
    return features

# -----------------------------
# 4. Векторизация текста
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
# 5. Ручные признаки
# -----------------------------
X_train_manual = extract_manual_features(train_df['shell'])
X_test_manual = extract_manual_features(test_df['shell'])

# -----------------------------
# 6. Объединение признаков
# -----------------------------
X_train = hstack([X_train_text, X_train_manual.values])
X_test = hstack([X_test_text, X_test_manual.values])

# -----------------------------
# 7. Обучение модели
# -----------------------------
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train, train_df['label'])

# -----------------------------
# 8. Предсказание и сохранение
# -----------------------------
test_pred = model.predict(X_test)

submission = pd.DataFrame({
    'id': test_df['ID'],
    'label': test_pred
})

submission.to_csv('submission.csv', index=False)
print("✅ Решение сохранено в submission.csv")
