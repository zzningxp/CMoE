from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    'model', type=str,
    help='Model to load; '
)
args = parser.parse_args()

quant_method = "gptq"
model_id = args.model
quant_path = args.model.rstrip("/").split("/")[-1] + "-" + quant_method

calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train"
).select(range(512))["text"]

quant_config = QuantizeConfig(
    bits=4, 
    group_size=128,
    quant_method=quant_method,
    # vram_strategy="balanced",
)

model = GPTQModel.load(model_id, quant_config, trust_remote_code=True)

# increase `batch_size` to match GPU/VRAM specs to speed up quantization
model.quantize(calibration_dataset, batch_size=1)

print(f"Saving quantized model to {quant_path}")
model.save(quant_path)