"""
Những class con inherit class feature_extractor cần cài method extract
Method extract(file):
Input: đường dẫn file ảnh
Output: list các giá trị là feature của ảnh, không cần cột kết quả

Ví dụ nếu ta muốn có một feature_extractor lấy 2 giá trị là
pixel tại (0,0) và pixel tại (1,1), ta làm như sau:
    1. Tạo 1 class inherit feature_extractor, tạm gọi là ext2pixel
    2. Cài đặt @staticmethod ext2pixel.extract(file)
        2.1 đọc file lấy 2 pixel nói trên, ra hai số a và b đều 0..255
        2.2 Trả về [a,b] (list có 2 phần tử tương ứng với 2 feature)

Yêu cầu của hàm extract():
    1. Số feature trả về phải cố định (mọi lần gọi đều trả về sồ feature như nhau)
    2. Thứ tự các feature phải cố định (cho mọi lần gọi)
    3. [Kĩ thuật] Giá trị feature phải hỗ trợ toán tử == cho 2 giá trị giống nhau.
"""

import cv2
import numpy as np

class feature_extractor:
    # Override hàm này
    @staticmethod
    def extract (file):
        return []

# Extractor lấy giá trị 48x48 pixel làm 2304 feature
class naive_pixel_extractor(feature_extractor):
    @staticmethod
    def extract(file):
        img=cv2.imread(file)
        rows, cols, chan = img.shape
        feat=[]
        for i in range(rows):
            for j in range(cols):
                feat.append(img[i,j,0])
        return feat
