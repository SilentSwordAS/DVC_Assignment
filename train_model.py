from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys

model = DecisionTreeClassifier(random_state=1, max_depth=3)
data = pd.read_csv(f'{sys.argv[1]}.csv')
le = LabelEncoder()

X_train,X_test = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data['species']
)

print(len(data))

model.fit((X_train.drop(['species'], axis=1)), le.fit_transform(X_train['species']))
joblib.dump(model, f'{sys.argv[1]}_model.joblib')
