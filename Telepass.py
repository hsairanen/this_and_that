
#%%

import os
import pandas as pd
from scipy.stats import ttest_ind
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import math
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scorecardpy as sc
from optbinning import OptimalBinning
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Set seed
np.random.seed(42)

wd = "C:\\Users\\stahei\\Documents"

#%%

# Read data
df = pd.read_excel(wd+"\\Telepass.xlsx", sheet_name=2)

# Clean response variable
df['response']=0
df.loc[df['issued']==True,'response']=1

#%%

# Create some variables from dates

df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')
df['age'] = (pd.Timestamp.now() - df['birth_date']).dt.days / 365.25
df = df.drop(columns=['birth_date'])

df['premium_cancellation'] = pd.to_datetime(df['premium_cancellation'], errors='coerce')
df['premium_subscription'] = pd.to_datetime(df['premium_subscription'], errors='coerce')

df['premium_age'] = (df['premium_cancellation']-df['premium_subscription']).dt.days / 365.25
df = df.drop(columns=['premium_cancellation'])
df = df.drop(columns=['premium_subscription'])


#%%

# Split into train and test 
X_train, X_test, y_train, y_test = train_test_split(
    df, df["response"], test_size=0.3, random_state=42
    )


#%%

# Fix datatype and remove columns not needed

variable_names = X_train.columns
variable_names = variable_names[~variable_names.isin(['quotation_id','client_id'])]

# Remove columns with datetime
datetime_cols = X_train.select_dtypes(include=['datetime']).columns
variable_names = variable_names[~variable_names.isin(datetime_cols)]


#%%

######################################################
# LOGISTIC REGRESSION
######################################################

# Woe binning of variables

user_splits_dict = {}
binning_results = {}
error_vars = []

for var in variable_names:
    try:
        if var in user_splits_dict:
            optb = OptimalBinning(
                name=var,
                dtype="numerical",
                user_splits=user_splits_dict[var],
                solver="mip",
                monotonic_trend=None,
                special_codes=[np.nan]
            )
        else:
            if X_train[var].dtype.name in ['category', 'object']:
                optb = OptimalBinning(
                    name=var,
                    dtype="categorical",
                    solver="mip",
                    #categorical_missing_treatment="missing"
                    special_codes=[np.nan]
                )
            else:
                optb = OptimalBinning(
                    name=var,
                    dtype="numerical",
                    solver="mip",
                    monotonic_trend="auto",
                    max_n_bins=7,
                    special_codes=[np.nan]
                )
    
        # Fit and transform
        optb.fit(X_train[var], y_train)
        binning_results[var] = optb

        X_train[f"{var}_woe"] = optb.transform(X_train[var], metric="woe")
        X_test[f"{var}_woe"] = optb.transform(X_test[var], metric="woe")

        X_train[f"{var}_bins"] = optb.transform(X_train[var], metric="bins")
        X_test[f"{var}_bins"] = optb.transform(X_test[var], metric="bins")

    except Exception as e:
        print(f"Failed on {var}: {e}")
        error_vars.append(var)

#%%

# Sort variables by total IV
iv_dict = {}
error_vars = []

for var in variable_names:
    try:
        opt = binning_results[var]
        bin_table = opt.binning_table.build()
        iv_dict[var] = bin_table['IV'].iloc[-1] 
    except Exception as e:
        error_vars.append(var)

sorted_iv = sorted(iv_dict.items(), key=lambda x: x[1], reverse=True)
sorted_iv = pd.DataFrame(sorted_iv, columns=['variable','iv'])
sorted_iv.to_csv('iv_values.csv', sep=';')


#%%

# Choose variables based on the total IV value
chosenVars = sorted_iv.loc[sorted_iv['iv']>0.05,'variable']
chosenVars = list(chosenVars)
chosenVars_woe = [f"{col}_woe" for col in chosenVars]

#%%
y = y_train

# Exclude:
exclude = ["price_full_woe", "price_sale_woe"]
chosenVars_woe = [x for x in chosenVars_woe if x not in exclude]

X_selected = X_train[chosenVars_woe]
X_selected = sm.add_constant(X_selected)  # add intercept term

logit_model = sm.Logit(y, X_selected)
result = logit_model.fit(disp=0)

#%%

summary_df = pd.DataFrame({
    "Coefficient": result.params,
    "Std.Error": result.bse,
    "z.value": result.tvalues,
    "p.value": result.pvalues,
    #"CI.lower": result.conf_int()[0],
    #"CI.upper": result.conf_int()[1]
})

print("\nDetailed coefficient table:")
print(summary_df)

#%%

vif_data = pd.DataFrame()
vif_data["feature"] = X_selected.columns
vif_data["VIF"] = [variance_inflation_factor(X_selected.values, i) for i in range(X_selected.shape[1])]
print(vif_data)

#%%

# Get predicted probabilities
y_pred_prob = result.predict(X_selected)

# Calculate AUC and Gini
auc = roc_auc_score(y, y_pred_prob)
gini = auc * 2 - 1

print(auc)

#%%

X_test_selected = X_test[chosenVars_woe]
X_test_selected = sm.add_constant(X_test_selected)  # Add intercept term

y_test_pred_prob = result.predict(X_test_selected)

# Calculate AUC
auc = roc_auc_score(y_test, y_test_pred_prob)

print(auc)


#%%

######################################################
# DECISION TREE
######################################################

# Replace "_woe" with "" in chosenVars_woe  
chosenVars = [var.replace("_woe", "") for var in chosenVars_woe]

X_train_tree = X_train[chosenVars]

# Convert categorical columns to dummy variables
X_train_tree = pd.get_dummies(X_train_tree, drop_first=True)

# Create and train the decision tree model
model = DecisionTreeClassifier(
    criterion="gini",   
    max_depth=3,
    random_state=42
)
model.fit(X_train_tree, y_train)

#%%

plt.figure(figsize=(16, 8))
plot_tree(
    model,
    feature_names=X_train_tree.columns,   # remove if X_train_tree is a NumPy array
    class_names=[str(c) for c in model.classes_],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Classifier")
plt.show()

#%%

y_train_pred = model.predict(X_train_tree)

train_auc = roc_auc_score(y_train, y_train_pred)
print(train_auc)


#%%

X_test_tree = X_test[chosenVars]

# Convert categorical columns to dummy variables
X_test_tree = pd.get_dummies(X_test_tree, drop_first=True)

# Match the training columns exactly
X_test_tree = X_test_tree.reindex(columns=X_train_tree.columns, fill_value=0)

y_test_pred = model.predict(X_test_tree)

test_auc = roc_auc_score(y_test, y_test_pred)

print(test_auc)

#%%

######################################################
# RANDOM FOREST
######################################################

X_train_forest = X_train_tree.copy()

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=20,   # number of trees
    max_depth=150,    # let trees grow fully
    random_state=42
)

# Train model
model.fit(X_train_forest, y_train)

#%%

# Make predictions
y_pred = model.predict(X_train_forest)

train_auc = roc_auc_score(y_train, y_pred)
print(train_auc)

#%%

X_test_forest = X_test_tree.copy()

# Make predictions
y_pred_test = model.predict(X_test_forest)

test_auc = roc_auc_score(y_test, y_pred_test)
print(test_auc)
