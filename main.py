import os
from feature_extractor import naive_pixel_extractor
from get_examples import get_examples, view_test_result_id3
from id3 import id3

if __name__=="__main__":
    extractor = naive_pixel_extractor()
    ex = get_examples(extractor)
    
    print('Train...')
    tree=id3().train(ex,False,0)

    print('Test')
    view_test_result_id3(tree, extractor)
