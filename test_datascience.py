# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
'''
https://machinelearningmastery.com/feature-selection-for-regression-data/
https://towardsdatascience.com/multiple-linear-regression-model-using-python-machine-learning-d00c78f1172a
https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
https://www.kaggle.com/mnoori/feature-selection-for-mlr-with-python
'''


# %%
'''
MAE (Mean absolute error) represents 
the difference between the original 
and predicted values extracted by 
averaged the absolute difference over 
the data set.
MAE = 1/n(sum(abs(y_test-y_pred)))
'''

# %% [markdown]
# Importamos las librerias necesarias.
# Necesitaremos sklearn que es una librería de machine learning que cuenta con algoritmos como regresión, clasificación, maquinas de soporte vectorial, entre otras.

# %%
import pandas as pd
import numpy as np
import os
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

import matplotlib

from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from matplotlib import pyplot
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

# %% [markdown]
# Importamos el dataset y con la ayuda de os.chdir 
# especificamos nuetro directorio de trabajo

# %%
#os.chdir("/home/nacho/Documents/ares_materials")
df1 = pd.read_csv('dataset1.csv', index_col=0)
print(df1.shape)
df1.head(5)


# %%
# exploratory data analysis


# %%
df1.info()
#No hay datos nulos


# %%
df1.isnull().sum()


# %%
df1.describe().T


# %%
#ColorMap
fig, ax = plt.subplots()
fig.set_size_inches(20, 11)
ax.xaxis.tick_top()
sns.heatmap(df1, cmap="Blues", vmin= 0.9, vmax=1.65,
           linewidth=0.3, cbar_kws={"shrink": .8})


# %%
sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11, 11)
sns.boxplot(data=df1, orient="h", palette="Set2")
plt.xticks(rotation=45)
sns.despine()


# %%
#Columnas que tienen una desviacion estandar mas alta que el promedio de las demas
df_std = df1.describe().loc['std',:][df1.describe().loc['std',:] > df1.describe().loc['std',:].mean()]
print(df_std)
df_std = df_std.to_frame()
columns_std = list(df_std.index)
columns_std.append('out')
df1[columns_std].head()


# %%
sns.pairplot(df1[columns_std], corner=True, diag_kind="kde", kind="reg", plot_kws={'line_kws':{'color':'red'}})
#solo dos columnas que tienen alto std tienen baja correlacion


# %%
corr = df1.corr()
#sort_corr= corr['out'].abs().sort_values(ascending=False)
sort_corr= corr['out'].sort_values(ascending=False)
corr_matrix = corr[sort_corr.index].corr()
matrix = np.triu(corr_matrix)
fig, ax = plt.subplots()
fig.set_size_inches(15, 15)
sns.heatmap(corr_matrix, annot=False, mask=matrix,cmap='coolwarm')
#los valores en color suave tienen baja correlacion


# %%
#Valor minimo de correlacion
c = 0.5
#c = abs(corr['out']).mean()
#c = 0.25


# %%
#columnas que tienen una correlacion mayor 
high_corr= abs(corr['out'])[abs(corr['out']) > c]
#high_corr= corr['out'][corr['out'] > corr.out.mean()]
high_corr = high_corr.drop(labels=['out'])
print(high_corr.sort_values(ascending=False))
columns_high_corr = list(high_corr.sort_values(ascending=False).to_frame().index)
#plot
sns.pairplot(df1, x_vars=columns_high_corr, y_vars=["out"], height=5, aspect=.8, kind="reg")


# %%
#columnas que tienen una correlacion menor 
low_corr= abs(corr['out'])[abs(corr['out']) < c]
#low_corr= corr['out'][corr['out'] < corr.out.mean()]
print(low_corr.sort_values())
df_low_corr = low_corr.sort_values().to_frame()
columns_low_corr = list(df_low_corr.index)


# %%
#se eliminan las columnas con poca correlacion
df_highcorr = df1.drop(df1[columns_low_corr],axis=1)
df_highcorr.shape


# %%
#Multivariate Regression


# %%
#Dividimos nuestros datos en "x" y "y" excluyendo la columna out


# %%
#X = df1.loc[:, df1.columns != 'out']
#y = df1.loc[:, df1.columns == 'out'].values.ravel()
X = df_highcorr.loc[:, df_highcorr.columns != 'out']
y = df_highcorr.loc[:, df_highcorr.columns == 'out'].values.ravel()


# %%
Dividimos el dataset en datos de entrenamiento y datos que nos servirán para el testing (en este caso el 33% de los datos).


# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# %%
#Tune the Number of Selected Features

# define the evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define the pipeline to evaluate
model = LinearRegression()
fs = SelectKBest(score_func=mutual_info_regression)
pipeline = Pipeline(steps=[('sel',fs), ('lr', model)])
# define the grid
grid = dict()
grid['sel__k'] = [i for i in range(X.shape[1]-20, X.shape[1]+1)]
# define the grid search
search = GridSearchCV(pipeline, grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=cv)
# perform the search
results = search.fit(X, y)
# summarize best
print('Best MAE: %.5f' % results.best_score_)
print('Best Config: %s' % results.best_params_)
# summarize all
means = results.cv_results_['mean_test_score']
params = results.cv_results_['params']
for mean, param in zip(means, params):
    print(">%.5f with: %r" % (mean, param))


# %%
# feature selection
def select_features(X_train, y_train, X_test):
	# configure to select a subset of features
	fs = SelectKBest(score_func=mutual_info_regression, k=results.best_params_['sel__k'])
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)

# %% [markdown]
# Model Selection: realizamos el entrenamiento utilizando varios modelos de regresión diferentes

# %%
classifiers = [
    svm.SVR(),
    linear_model.Ridge(),
    linear_model.SGDRegressor(),
    linear_model.BayesianRidge(),
    linear_model.LassoLars(),
    linear_model.LassoCV(),
    linear_model.ARDRegression(),
    linear_model.PassiveAggressiveRegressor(),
    linear_model.TheilSenRegressor(),
    linear_model.LinearRegression()
    ]

# %% [markdown]
# Definimos las llaves del diccionario y lo creamos junto con una lista vacia

# %%
dict_list = ['name', 'y_pred', 'mae', 'model']
class_dict = {}
class_list = []

# %% [markdown]
# Iteramos todos los modelos y almacenamos el atributo name y cada modelo calcula los modelos más óptimos para los pesos utilizando los valores de entrada y salida de los datos de entrenamiento.
# %% [markdown]
# Utilizamos mean absolute error (MAE) para encontrar cual de todos los modelos nos da el error mas pequeño.

# %%
for i, classifier in enumerate(classifiers):
    class_dict = {}
    class_dict[dict_list[0]] = (classifiers[i].__class__.__name__)
    clf = classifier
    pipeline = Pipeline(steps=[
    #('scaler', MinMaxScaler()),
    #('scaler', RobustScaler()),
    ('scaler', StandardScaler()),
    ('model', clf)
    ])

    pipeline.fit(X_train_fs, y_train)
    y_pred = pipeline.predict(X_test_fs)

    #pipeline.fit(X_train, y_train)
    #y_pred = pipeline.predict(X_test)

    class_dict[dict_list[1]] = y_pred
    mae = metrics.mean_absolute_error(y_test, y_pred)
    class_dict[dict_list[2]] = mae
    class_dict[dict_list[3]] = clf
    class_list.append(class_dict)

# %% [markdown]
# Analizamos los datos 'mae' del diccionario y encontramos el más pequeño, en este caso fue.

# %%
minl = []
for dicts in class_list:
    #print(dicts['name'])
    minl.append(dicts['mae'])

min_mae = min(minl)

def return_best_mae(class_list, min_mae):
    for dicts in class_list:
        if dicts['mae'] == min_mae:
            return dicts

best_mae = return_best_mae(class_list, min_mae)
print(best_mae)


# %%
'''
WITH HIGH CORR DATA AND FEATURE SELECTION
{'name': 'BayesianRidge', 'y_pred': array([1.54516922, 1.52425139, 1.55374143, 1.54541806, 1.54516922,
       1.55588899, 1.52931748, 1.53386998, 1.55374143, 1.52922606,
       1.52516904, 1.555227  , 1.52149844, 1.54231889, 1.555227  ,
       1.54218316]), 'mae': 0.003681995423639478, 'model': BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, alpha_init=None,
              compute_score=False, copy_X=True, fit_intercept=True,
              lambda_1=1e-06, lambda_2=1e-06, lambda_init=None, n_iter=300,
              normalize=False, tol=0.001, verbose=False)}
'''


# %%
'''
WITH HIGH CORRELATION DATA AND NO FEATURE SELECTION
{'name': 'BayesianRidge', 'y_pred': array([1.54528197, 1.52248676, 1.55455364, 1.54557025, 1.54528197,
       1.5565687 , 1.52645666, 1.53282866, 1.55455364, 1.52780615,
       1.52341837, 1.55584439, 1.51969194, 1.54197984, 1.55584439,
       1.54182259]), 'mae': 0.0037847423337281777, 'model': BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, alpha_init=None,
              compute_score=False, copy_X=True, fit_intercept=True,
              lambda_1=1e-06, lambda_2=1e-06, lambda_init=None, n_iter=300,
              normalize=False, tol=0.001, verbose=False)}
'''


# %%
'''
WITH HIGH CORR DATA AND FEATURE SELECTION
{'name': 'LinearRegression', 'y_pred': array([1.54369903, 1.52056455, 1.55432933, 1.54360333, 1.54369903,
       1.55712336, 1.52616507, 1.53423661, 1.55432933, 1.52827235,
       1.52306229, 1.55585731, 1.51307132, 1.5447952 , 1.55585731,
       1.5448474 ]), 'mae': 0.0039378484031668826, 'model': LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)}
'''


# %%
'''
WITH ALL DATA (no correlation or feature selection)
{'name': 'BayesianRidge', 'y_pred': array([1.5448769 , 1.52247832, 1.55469369, 1.54512147, 1.5448769 ,
       1.55666446, 1.52603901, 1.53295677, 1.55469369, 1.52793525,
       1.52354839, 1.55612351, 1.51926809, 1.54207539, 1.55612351,
       1.54194198]), 'mae': 0.0037418050873889858, 'model': BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, alpha_init=None,
              compute_score=False, copy_X=True, fit_intercept=True,
              lambda_1=1e-06, lambda_2=1e-06, lambda_init=None, n_iter=300,
              normalize=False, tol=0.001, verbose=False)}
'''

# %% [markdown]
# Graficamos los resultados

# %%
#Residual Analysis

y_train_price = best_mae['model'].predict(X_train_fs)
#y_train_price = best_mae['model'].predict(X_train)

#plot
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)        
plt.xlabel('Errors', fontsize = 18)    


# %%
#Feature Importance

fs_index = fs.get_support(indices=True)
fs_columns = X.iloc[:,fs_index].columns
coef = pd.Series(best_mae['model'].coef_, index = fs_columns)

#coef = pd.Series(best_mae['model'].coef_, index = X.columns)

imp_coef = coef.sort_values()
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title(f"Feature importance using {best_mae['name']}\nwith MAE: {best_mae['mae']}")


# %%
high_corr.sort_values(ascending=False)


# %%
plt.plot(y_test - best_mae['y_pred'],marker='o',linestyle='')
plt.title("Prediction Errors")
plt.legend()
plt.show() 


# %%



