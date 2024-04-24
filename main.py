from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = FastAPI(
    title = "Deploy Obesity detection",
    version = "0.0.1"
    )


# ----------------------------------------
# LOAD MODEL
#-----------------------------------------
model = joblib.load("model/logistic_regression_model_v01.pkl")

@app.post("/api/v1/predict_Obesity", tags=["Obesity"])
async def predict(
    Gender: float,
    Age: float,
    Height: float,
    Weight: float,
    family_history_with_overweight: float,
    FAVC: float,
    FCVC: float,
    NCP: float,
    SMOKE: float,
    CH2O: float,
    SCC: float,
    FAF: float,
    TUE: float,
    CAEC_encoded: float,
    MTRANS_encoded: float,
    CALC_encoded: float,
    ):
    obesity_dictionary = {
        'Gender': Gender,
        'Age': Age,
        'Height': Height,
        'Weight': Weight,
        'family_history_with_overweight': family_history_with_overweight,
        'FAVC': FAVC,
        'FCVC': FCVC,
        'NCP': NCP,
        'SMOKE': SMOKE,
        'CH2O': CH2O,
        'SCC': SCC,
        'FAF': FAF,
        'TUE': TUE,
        'CAEC_encoded': CAEC_encoded,
        'MTRANS_encoded': MTRANS_encoded,
        'CALC_encoded': CALC_encoded
    }

    try:
        df = pd.DataFrame(obesity_dictionary, index=[0])
        prediction = model.predict(df)
        prediction_result = int(prediction[0])
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=prediction_result
        )
    except Exception as e:
        raise HTTPException(
            detail=str(e),
            status_code=status.HTTP_400_BAD_REQUEST
        )




