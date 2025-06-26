import numpy as np
import pandas as pd
import re
import glob
from scipy.stats import shapiro
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder

# files = glob.glob('flatten_csv_incorrect/*.csv')
# data = []
# for f in files :
#     doc = pd.read_csv(f)
#     doc = doc.drop(['Label', 'Flow ID', 'Src IP','Src Port','Dst IP','Dst Port','Timestamp'], axis=1)
#     doc = doc.apply(pd.to_numeric, errors='coerce').dropna()
#     data.append(doc)
# data = pd.concat(data)

def sanitize_filename(column_name):
    # Remove invalid file name characters
    sanitized_name = re.sub(r'[<>:"/\\|?*\s]', '', column_name)
    # Replace spaces with underscores
    sanitized_name = sanitized_name.replace(' ', '_')
    # Shorten the name if it's too long
    if len(sanitized_name) > 50:
        sanitized_name = sanitized_name[:50]
    return sanitized_name

def test_normality(data, column_name):
    stat, p = shapiro(data[column_name])
    # print(f'Feature: {column_name}')
    # print('Statistics=%.3f, p=%.3f' % (stat, p))
    # Interpret
    alpha = 0.05
    if p > alpha:
        #print(f'{column_name}: Sample looks Gaussian (fail to reject H0)')
        return True
    else:
        #print(f'{column_name}: Sample does not look Gaussian (reject H0)')
        return False

@dataclass
class NormalizeSolution :
    is_normal: bool
    min_val: float
    mean_after_min: float
    std_after_min: float
    category: bool
    cat_enc: LabelEncoder
    raw: bool = False

    def apply(self, col) :
        if self.category :
            return self.cat_enc.transform(col)
            #return col.astype('category')
        if self.raw :
            return col
        if self.is_normal :
            v = (col - self.mean_after_min) / (self.std_after_min + 1e-6)
            if v.isnull().values.any() :
                #breakpoint()
                raise Exception
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            return v
        else :
            col = np.log1p(np.maximum(col - self.min_val, 0))
            v = (col - self.mean_after_min) / (self.std_after_min + 1e-6)
            if v.isnull().values.any() :
                #breakpoint()
                raise Exception
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            return v
    
    def to_json(self) :
        ret = {
            'is_normal': self.is_normal,
            'min_val': float(self.min_val),
            'mean_after_min': float(self.mean_after_min),
            'std_after_min': float(self.std_after_min),
            'category': self.category,
        }
        if self.category :
            cat_enc = []
            for i in range(len(self.cat_enc.classes_)) :
                cat_enc.append([int(self.cat_enc.classes_[i]), i])
            ret['cat_enc'] = cat_enc
        return ret

# Function to save plot for each feature
def create_solution(data, column_name, method = 'all'):
    print('creating normalize solution for', column_name)
    if column_name in ['Feat 0', 'Feat 1'] :
        le = LabelEncoder()
        le.fit_transform(data[column_name])
        return NormalizeSolution(False, 0, 0, 0, True, le)
        
    if method == 'all' :
        n = test_normality(data, column_name)
        if not n :
            mv = np.min(data[column_name])
            data[column_name] = np.log1p(data[column_name] - mv)
            return NormalizeSolution(False, mv, np.mean(data[column_name]), np.std(data[column_name]), False, None)
        else :
            m = np.mean(data[column_name])
            return NormalizeSolution(True, 0, m, np.std(data[column_name]), False, None)
    elif method == 'nolog' :
        m = np.mean(data[column_name])
        return NormalizeSolution(True, 0, m, np.std(data[column_name]), False, None)
    elif method == 'none' :
        return NormalizeSolution(True, 0, 0, 0, False, None, raw = True)


# # Iterate over each column in the DataFrame
# per_col_solution = {}
# for column in data.columns:
#     sol = create_solution(data, column)
#     per_col_solution[column] = sol


# for f in files :
#     doc = pd.read_csv(f)
#     doc = doc.drop(['Label', 'Flow ID', 'Src IP','Src Port','Dst IP','Dst Port','Timestamp'], axis=1)
#     doc = doc.apply(pd.to_numeric, errors='coerce').dropna()
#     for column in doc.columns :
#         doc[column] = per_col_solution[column].apply(doc[column])
#     f = f.replace('flatten', 'normalized')
#     doc.to_csv(f)

