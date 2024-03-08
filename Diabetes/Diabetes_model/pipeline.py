import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from Diabetes_model.config.core import config

print(config.models_cfg)

Diabetes_pipe=Pipeline([
    
    # scale
    ("scaler", StandardScaler()),
    ('model_xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0, n_estimators=config.models_cfg.n_estimators,  max_depth=config.models_cfg.max_depth,  
                                      learning_rate=config.models_cfg.learning_rate, subsample= config.models_cfg.subsample, colsample_bytree =config.models_cfg.colsample_bytree, scale_pos_weight = config.models_cfg.scale_pos_weight))
     
    ])