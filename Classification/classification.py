import pandas as pd 
import csv
import sys
import os
sys.path.append("/opt/pytorch-v2.7.1/lib/python3.12/site-packages/")
import transformers
import torch
from tqdm import tqdm
from pathlib import Path
from utils.prompts import Prompts
from utils.prompt_ita import Prompts_ita
from utils.prompt_es import Prompts_es
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.clean_output import extract_demographics

dir_processed_dataset = ""
dir_predictions_baseline = ""

Path(dir_processed_dataset).mkdir(exist_ok=True)
Path(dir_predictions_baseline).mkdir(parents=True, exist_ok=True)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

#SET THE LANGUAGE
lang = "multi"

if lang == "it":
  prompts = Prompts_ita()
elif lang == "es":  
    prompts = Prompts_es()
else:   
  prompts = Prompts()

# obtain subset

df = pd.read_csv("")

excluded_ids = ["1536383583849525248", "it_1131", "en_1254", "en_1013", "es_286", "es_341"]
df = df[~df["id"].isin(excluded_ids)]

if lang == "it":
  df = df[df["annotator_Lang"] == "it"]
elif lang == "es":  
  df = df[df["annotator_Lang"] == "es"]

#  process subset
df = df.copy()
demographic_traits=False
reclaim=False
batch_size=32


df["prompt"] = df.apply(lambda row: prompts.get_prompt(row, demographic_traits=demographic_traits, reclaim=reclaim), axis=1)


model_id = ""
model_name = model_id.split("/")[1].split("-")[0]

reclaim_str = "reclaim" if reclaim else "offensive"

df.to_csv(f'{dir_processed_dataset}/processed_{model_name}_es_{reclaim_str}_baseline.csv',index=False)
prediction_filename = f'predictions_{model_name}_es_{reclaim_str}_baseline.csv'


# create the file 
print(f"Creating file: {prediction_filename}")


prediction_path = f"{dir_predictions_baseline}/{prediction_filename}"

processed_ids = set()
if os.path.exists(prediction_path):
    print("Resuming from existing results...")
    prev = pd.read_csv(prediction_path)
    processed_ids = set(prev["postId"].astype(str))
else:
    print("No previous results found — starting fresh.")

# Filter dataframe to only unprocessed rows
df = df[~df["id"].astype(str).isin(processed_ids)]
print("Remaining items to process:", len(df))

# Open predictions file in append or write mode
mode = "a" if processed_ids else "w"
file = open(prediction_path, mode)

if reclaim:
  writer = csv.DictWriter(file,fieldnames=["reclaimed","postId", "annId","output"])
else: 
  writer = csv.DictWriter(file,fieldnames=["offensive","postId", "annId","output"]) 

# Write header only when creating new file
if mode == "w":
    writer.writeheader()


#-------------

# classify
transformers.set_seed(42)

tokenizer = AutoTokenizer.from_pretrained(model_id,
    token="",
    padding_side='left')
# Set pad token if not already set
if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
  model_id, 
  token="",
  torch_dtype=torch.bfloat16,
  device_map="auto",
  trust_remote_code=True
  )

device=next(model.parameters()).device
num_batches = len(df) // batch_size + (1 if len(df) % batch_size > 0 else 0)

for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
  start_idx = batch_idx * batch_size
  end_idx = min((batch_idx + 1) * batch_size, len(df))
  df_batch = df.iloc[start_idx:end_idx]
  
  encoded = tokenizer(
    df_batch['prompt'].tolist(),
    return_tensors="pt",
    padding = True,
    truncation=True,
    max_length=2048
  )
  input_ids = encoded.input_ids.to(device)
  attention_mask = encoded.attention_mask.to(device)

  with torch.no_grad():
    outputs = model.generate(
      input_ids,
      attention_mask = attention_mask,
      do_sample=False,
      max_new_tokens=100,
      pad_token_id=tokenizer.eos_token_id
    )
  
  for i, (_, item) in enumerate(df_batch.iterrows()):
    input_length = input_ids[i].shape[0]
    new_tokens = outputs[i][input_length:]
    gen_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Extract demographics and write row
    demographics = extract_demographics(item.prompt)
    row_data = {
            'postId': item.id,
            'annId': item.annotator_ID,
            'output': gen_output
        }
    if reclaim:
      row_data['reclaimed'] = item.reapp 
    else:
      row_data['offensive'] = item.offensive
            
    writer.writerow(row_data)