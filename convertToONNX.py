import pickle
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType


model_name= "spam_pipeline.pkl"

# load your trained pipeline/model
with open(model_name, "rb") as f:
    model = pickle.load(f)

# define input type (text input)
initial_type = [("input", StringTensorType([None, 1]))]

# convert to ONNX
onnx_model = convert_sklearn(model, initial_types=initial_type)

# save ONNX model
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
