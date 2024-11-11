import pandas as pd
import glob
import json
result={}
addresses=glob.glob('*/*/*.json')
for address in addresses:
    print(address)
    model_name=address.split('/')[0]
    if model_name not in result.keys():
        result[model_name]={}
    task=address.split('/')[-1].replace('.json','')
    try:
        with open(address, "r") as json_file:
            data = json.load(json_file)
            main_score = data.get('scores',{}).get("test", {})[0].get("main_score")
            if main_score is not None:
                result[model_name][task]=main_score
    except Exception as e:
        # print(f"Error reading {address}: {e}") 
        pass   
result=pd.DataFrame(result)
result.to_csv('res.csv')
print(f'stored in res.csv')