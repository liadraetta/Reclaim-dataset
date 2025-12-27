import re 
import ast 

def extract_demographics(prompt):
    """
    Extract demographics dictionary from prompt text.
    Returns the dictionary if found, otherwise returns "None".
    """
    # Pattern to match "You are characterized by the following demographics:" followed by a dictionary
    pattern = r"You are characterized by the following demographics:\s*(\{[^}]*\})"
    
    match = re.search(pattern, prompt)
    
    if match:
        try:
            # Extract the dictionary string
            dict_str = match.group(1)
            # Safely evaluate the dictionary string
            demographics = ast.literal_eval(dict_str)
            return demographics
        except (ValueError, SyntaxError):
            # If the dictionary string is malformed, return "None"
            return "None"
    else:
        return "None"


def extract_prediction(output):
    text = output.lower().strip().splitlines()
    for i in text:
        if 'the sentence is offensive' in i:
            return 1, i
        elif 'the sentence is not offensive' in i:
            return 0, i
    return -1, None

def extract_output(df, output_col):
    list_preds = []
    list_parsed_outputs = []
    for idx,row in df.iterrows():
        pred, parsed_output = extract_prediction(row[output_col])
        list_preds.append(pred)
        list_parsed_outputs.append(parsed_output)

    df['prediction'] = list_preds
    df['parsed_output'] = list_parsed_outputs

    return df 
