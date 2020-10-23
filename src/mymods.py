from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, f1_score, classification_report, plot_confusion_matrix, make_scorer, recall_score, accuracy_score, precision_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer, make_column_transformer
from xgboost import XGBClassifier 
import xgboost as xgb
import scipy



def model_output(model, X_t, X_val, y_t, y_val):
    '''Can be used on final test and train validation''
    input:   model, X_t, X_val, y_t, y_val
    or 
    input:   model, X_train, X_test, y_train, y_test
    '''
    
    model.fit(X_t, y_t)
    y_hat = model.predict(X_val)
    
    print(f'''The Cross Val accuracy is: {cross_val_score(estimator = model, X = X_t,y = y_t, cv = 3, scoring = 'accuracy').mean()}''')
    print(f'The test Accuracy is: {accuracy_score(y_val, y_hat)}')
    print(confusion_matrix(y_val, y_hat))
    print(classification_report(y_val, y_hat))

        
    return

def plot_feature_importances(model):
    n_features = X_resample.shape[1]
    plt.figure(figsize=(8,15))
    plt.barh(range(n_features), model.feature_importances_, align='center') 
    plt.yticks(np.arange(n_features), X_resample.columns.values) 
    plt.title('Random forest feature importance')
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    plt.savefig(f'../report/figures/{str(model)}Feature_importance.png',dpi=300, bbox_inches='tight') 