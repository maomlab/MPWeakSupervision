
import numpy as np
import pandas as pd
from MPWeakSupervision.dataset import make_bags


def test_make_bags():
    np.random.seed(0)
    data = pd.DataFrame({'well_id': np.random.randint(low=1, high=6, size = 100)})
    
    bags = make_bags(data, bag_size = 3)
    assert len(bags) == 34
    assert len(bags[0]) == 3

    bags = make_bags(data, bag_size = 110, sample = False)
    assert len(bags) == 1
    assert len(bags[0]) == 100

    bags = make_bags(data, bag_size = 110, sample = True)    
    assert len(bags) == 1
    assert len(bags[0]) == 110

    bags = make_bags(data, bag_size = 3, group_by = 'well_id')
    assert len(bags) == 36
    assert len(bags[3]) == 3

    bags = make_bags(data, bag_size = 110, group_by = 'well_id', sample = False)
    assert len(bags) == 5
    assert len(bags[0]) == 22
    
    bags = make_bags(data, bag_size = 110, group_by = 'well_id', sample = True)    
    assert len(bags) == 5
    assert len(bags[0]) == 110

