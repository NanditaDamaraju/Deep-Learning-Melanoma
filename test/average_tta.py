import numpy as np
import pandas as pd

data = pd.read_csv("simulation.csv",names=['name','prob'])
data['id'] = data.apply(lambda row: row['name'][:12],axis=1)
new = data.groupby(['id']).mean()
new.to_csv('final_submission.csv',header=False)
