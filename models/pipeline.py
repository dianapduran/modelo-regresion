from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import ElasticNet

def build_pipeline():
    num_features = ['edad', 'horas_trabajadas']
    cat_features = ['nivel_educativo']
    
    num_transformer = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False))
    ])
    
    cat_transformer = OneHotEncoder(drop='first')
    
    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', ElasticNet())
    ])
    
    return pipeline
