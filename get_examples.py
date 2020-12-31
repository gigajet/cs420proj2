import os
from feature_extractor import naive_pixel_extractor
from id3 import id3

"""
Nếu cấu trúc thư mục train thay đổi, thì sửa file này lại
Sửa hàm get_examples.
"""
# Dùng feature_extractor.extract()
def get_examples (feature_extractor):
    HAPPY_DIR='train/0/' # Trailing / is necessary
    DISGUST_DIR='train/1/'

    examples=[]
    for img_file in os.listdir(HAPPY_DIR):
        ex=feature_extractor.extract(HAPPY_DIR+img_file)
        ex.append(0)
        examples.append(ex)
    for img_file in os.listdir(DISGUST_DIR):
        ex=feature_extractor.extract(DISGUST_DIR+img_file)
        ex.append(1)
        examples.append(ex)
    return examples


"""
Nếu cấu trúc thư mục test (hoặc bộ test) thay đổi, thì sửa hàm này lại
"""
# Dùng feature_extractor.extract()
# feature_extractor cần giống với trong get_examples
def view_test_result_id3 (id3_tree, feature_extractor):
    HAPPY_DIR='test/0/' # Trailing / is necessary
    DISGUST_DIR='test/1/'

    print('Testing happy...')
    pos=0;neg=0;inc=0
    for img_file in os.listdir(HAPPY_DIR):
        ex=feature_extractor.extract(HAPPY_DIR+img_file)
        res = id3_tree.classify(ex)
        if res == 1:
            pos+=1
        elif res == -1:
            neg+=1
        else:
            inc+=1
    print('hpy dis inc = tot: %d %d %d = %d' % (pos,neg,inc,pos+neg+inc))

    print('Testing disgusting...')
    pos=0;neg=0;inc=0
    for img_file in os.listdir(DISGUST_DIR):
        ex=feature_extractor.extract(DISGUST_DIR+img_file)
        res = id3_tree.classify(ex)
        if res == 1:
            pos+=1
        elif res == -1:
            neg+=1
        else:
            inc+=1
    print('hpy dis inc = tot: %d %d %d = %d' % (pos,neg,inc,pos+neg+inc))
