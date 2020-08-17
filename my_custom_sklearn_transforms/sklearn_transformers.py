from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')
    
class SetIndex(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.set_index(self.columns, inplace=True)
    
class Balanceamento(BaseEstimator, TransformerMixin):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def fit(self, X, y):
        return self

    def transform(self, X, y):
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(sampling_strategy='all')
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        self.X, self.y = smote.fit_resample(X,y)
        # Retornamos um novo dataframe sem as colunas indesejadas
        return self.X, self.y
    
