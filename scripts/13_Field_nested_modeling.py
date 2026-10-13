from project_paths import DATA_DIR, RESULTS_DIR
from project_paths import require_review, read_field_dataset
require_review("field_training_method")
# ============================================================
# PFAS Field ML
# Nested Group CV
# Four models + Hyperparameter optimization
# Generate OOF predictions
# ============================================================


import os
import json
import numpy as np
import pandas as pd


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    GroupKFold,
    GridSearchCV
)

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)


from xgboost import XGBRegressor

from lightgbm import LGBMRegressor



# ============================================================
# 1. Paths
# ============================================================


ROOT = (
str(RESULTS_DIR)
)


INPUT_FILE = str(DATA_DIR / 'Machine-learning dataset.xlsx')


OUTPUT_DIR = os.path.join(
    ROOT,
    "Field_Nested_ML_results"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



# ============================================================
# 2. Read dataset
# ============================================================


df = read_field_dataset(INPUT_FILE)


print(
    "Original:",
    df.shape
)



# strict dataset

df = df[
    df["Data quality tier"]
    ==
    "A_strict_matched"
].copy()



df.reset_index(
    drop=True,
    inplace=True
)


print(
    "Strict:",
    df.shape
)



# ============================================================
# 3. Define variables
# ============================================================


cat_cols = [

    "Plant tissue group",

    "Morphotype"

]



num_cols = [

    "Soil pH",

    "SOM",

    "Measured soil concentration of the corresponding PFAS",

    "Chi3v",

    "MinPartialCharge",

    "TPSA",

    "ALogP",

    "GATS3c",

    "SpMin8_Bhi"

]



target = (
    "Log PFASs concentration"
)



X = df[
    cat_cols + num_cols
]


y = df[target]



# ============================================================
# 4. Group definition
# ============================================================


# 推荐最终论文使用 soil/site group
# 如果你的表没有 Soil_sample_ID
# 改成 Study_ID + Site_ID


groups = (

df["Study_ID"]
.astype(str)

+
"__"

+

df["PFASs Name"]
.astype(str)

)



print(
"Groups:",
groups.nunique()
)



# ============================================================
# 5. Preprocessor
# ============================================================


preprocessor = ColumnTransformer(

[
(
"cat",

OneHotEncoder(
handle_unknown="ignore",
sparse_output=False
),

cat_cols

),

(
"num",

"passthrough",

num_cols

)

]

)



# ============================================================
# 6. Models
# ============================================================


models = {



"RF":

(

RandomForestRegressor(
random_state=42
),

{

"model__n_estimators":[300,500],

"model__max_depth":[None,10,20],

"model__min_samples_leaf":[1,3]

}

),




"XGBoost":

(

XGBRegressor(

objective="reg:squarederror",

random_state=42,

n_jobs=-1

),

{

"model__n_estimators":[300,500],

"model__learning_rate":[0.03,0.05],

"model__max_depth":[3,5],

"model__subsample":[0.8,1]

}

),




"LightGBM":

(

LGBMRegressor(

random_state=42,

verbose=-1

),

{

"model__n_estimators":[300,500],

"model__learning_rate":[0.03,0.05],

"model__num_leaves":[15,31]

}

),




"GB":

(

GradientBoostingRegressor(

random_state=42

),

{

# Ordinary GradientBoostingRegressor, confirmed by the author.
# Historical saved GB metrics have not been regenerated with this repair.
"model__n_estimators":[200,300],

"model__learning_rate":[0.03,0.05],

"model__max_leaf_nodes":[15,31]

}

)

}



# ============================================================
# 7. Nested Group CV
# ============================================================


outer_cv = GroupKFold(
n_splits=5
)


inner_cv = GroupKFold(
n_splits=4
)



all_results=[]



model_oof={}



for model_name,(model,param_grid) in models.items():


    print("\n===================")

    print(model_name)

    print("===================")



    oof_pred=np.zeros(
        len(df)
    )


    fold_results=[]



    for fold,(train_idx,test_idx) in enumerate(

        outer_cv.split(
            X,
            y,
            groups
        ),

        start=1

    ):


        print(
        "Outer fold:",
        fold
        )


        pipe = Pipeline(

        [

        (
        "prep",
        preprocessor
        ),

        (
        "model",
        model
        )

        ]

        )



        search = GridSearchCV(

            pipe,

            param_grid,

            cv=inner_cv,

            scoring="r2",

            n_jobs=-1

        )



        search.fit(

            X.iloc[train_idx],

            y.iloc[train_idx],

            groups=groups.iloc[train_idx]

        )



        best_model = search.best_estimator_



        pred = best_model.predict(

            X.iloc[test_idx]

        )



        oof_pred[test_idx]=pred



        fold_results.append(

        {

        "fold":fold,

        "R2":

        r2_score(

        y.iloc[test_idx],

        pred

        ),

        "RMSE":

        np.sqrt(

        mean_squared_error(

        y.iloc[test_idx],

        pred

        )

        )

        }

        )



    # performance

    r2 = r2_score(
        y,
        oof_pred
    )


    rmse = np.sqrt(

        mean_squared_error(

        y,

        oof_pred

        )

    )


    mae = mean_absolute_error(

        y,

        oof_pred

    )



    print(
    model_name,
    r2,
    rmse,
    mae
    )



    model_oof[model_name]=oof_pred



    all_results.append(

    {

    "Model":model_name,

    "OOF_R2":r2,

    "OOF_RMSE":rmse,

    "OOF_MAE":mae,

    "Fold_results":fold_results

    }

    )



# ============================================================
# 8. Select best model
# ============================================================


results_df=pd.DataFrame(
[
{
"Model":x["Model"],
"OOF_R2":x["OOF_R2"],
"OOF_RMSE":x["OOF_RMSE"],
"OOF_MAE":x["OOF_MAE"]
}

for x in all_results
]
)


results_df.sort_values(

"OOF_R2",

ascending=False,

inplace=True

)



print(results_df)



results_df.to_excel(

os.path.join(
OUTPUT_DIR,
"Model_comparison.xlsx"
),

index=False

)



best_model_name = (

results_df.iloc[0]["Model"]

)



print(

"Best model:",

best_model_name

)



# ============================================================
# 9. Save OOF prediction
# ============================================================


oof_file=pd.DataFrame(

{

"Observed":

y,

"Predicted":

model_oof[best_model_name],

"Study_ID":

df["Study_ID"],

"PFAS":

df["PFASs Name"]

}

)



oof_file.to_csv(

os.path.join(

OUTPUT_DIR,

"Field_Nested_OOF_predictions.csv"

),

index=False

)



# ============================================================
# 10. Save summary
# ============================================================


summary={

"Dataset":

"Field strict matched",

"N":

len(df),

"Best_model":

best_model_name,

"Results":

results_df.to_dict(
orient="records"
)

}



with open(

os.path.join(

OUTPUT_DIR,

"summary.json"

),

"w",

encoding="utf-8"

) as f:


    json.dump(

    summary,

    f,

    indent=4,

    ensure_ascii=False

    )



print(
"\nFinished!"
)

print(
OUTPUT_DIR
)
