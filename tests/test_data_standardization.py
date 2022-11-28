
import numpy as np
import pandas as pd
import time
from MPWeakSupervision.dataset import data_standardization
from sklearn.preprocessing import StandardScaler, normalize

def test_data_standardization_all():
    # pandas can be super slow if not being careful, so check that we're not doing silly things
    # when normalizing large arrays

    # e.g. 500k cells with 500 features each
    data_array = np.random.rand(500000, 500)
    column_subset_index = [i for i in range(20, 500)]
    column_names = [f"col_{i}" for i in range(500)]
    column_subset_names = [f"col_{i}" for i in range(20, 500)]

    data = data_array
    begin_time = time.time()
    data_array = StandardScaler().fit_transform(data)
    end_time = time.time()
    print(f"standardize numpy data runtime: {end_time - begin_time}")    

    begin_time = time.time()
    data_subset = data[:, column_subset_index]
    end_time = time.time()
    print(f"accessing subset of numpy data runtime: {end_time - begin_time}")        

    
    data = data_array
    begin_time = time.time()
    data_array[:, column_subset_index] = StandardScaler().fit_transform(data[:, column_subset_index])
    end_time = time.time()
    print(f"standardize numpy data subset runtime: {end_time - begin_time}")


    data = pd.DataFrame(data_array, columns = column_names)
    begin_time = time.time()
    data = data_standardization(data)
    end_time = time.time()
    print(f"standardize pandas data runtime: {end_time - begin_time}")

    data = pd.DataFrame(data_array, columns = column_names)     
    begin_time = time.time()
    data = data_standardization(data, columns = column_subset_names)
    end_time = time.time()
    print(f"standardize pandas data subset runtime: {end_time - begin_time}")
