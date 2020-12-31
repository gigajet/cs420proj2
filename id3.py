"""
Nhớ là: Đầu ra của ID3 là classify nhị phân.

Đọc dữ liệu rồi extract features xong, mình mong có được gì?
Có lẽ là một loại bảng biểu như cô thể hiện.

Hai phần tử sẽ so sánh bằng nhau hay không bằng toán tử ==, bất chấp kiểu dữ liệu.
(Chuẩn hóa True là 1 và False là 0)

Input bảng biểu đó vô ID3,
ID3 sẽ tự train thành 1 cây, có thể dùng để phân loại.

Về cấu trúc dữ liệu, cần thiết kế 2 loại CTDL: cho examples và ID3.

(i) CTDL cho examples khá đơn giản, có vẻ sẽ là mảng 2 chiều.
    1. Số dòng là số lượng example.
    2. Số cột là số feature.
    3. Cột cuối cùng, là feature đích, bắt buộc phải mang giá trị True / False.
    4. Cột cuối cùng LUÔN được ngầm định là feature đích.
    5. Thứ tự feature (thứ tự các cột) là CỐ ĐỊNH trong suốt quá trình train cũng như classify
    6. Khi classify, cần đưa một mảng các giá trị có cùng thứ tự với
        thứ tự feature trong example (không cần cột cuối).
(ii) Đặc tả chức năng (features) của ID3 trường hợp đặc biệt:
    1. Làm gì khi không có branch: Inconclusive hay Default_Value (xem bài cuối Review Exercise 7)
    2. Majority vote giữa 2 tập bằng nhau luôn cho kết quả Inconclusive
    3. Split dựa trên minimum entropy.
(iii) Vấn đề chi tiết cài đặt:
    1. Lưu feature nào đã dùng hay chưa ra làm sao?
    2. Tách & Đẩy các example xuống nút lá như thế nào?
    3. Mỗi nút cần lưu: nó là lá hoặc không, nếu không, nó split trên feature nào.
    4. Mỗi nút cần lưu con trỏ tới các nút con.
    5. Các nút lá cần lưu các examples tồn đọng.
"""
import copy
import math

class id3:

    def __init__ (self):
        self.trained = False

    def train (self, dataset, create_default_value=False, positive_label=None):
        self.trained = True
        self.dataset = copy.deepcopy (dataset)
        self.num_feature = len(self.dataset[0]) - 1
        if positive_label is None:
            self.positive_label = True
        else:
            self.positive_label = copy.deepcopy(positive_label)

        if create_default_value == True:
            self.default_value = {}
            # For each attribute A and value V, store # positive & negative
            for i in range(len(self.dataset)):
                res = self.dataset[i][self.num_feature]
                for A in range(self.num_feature):
                    V = self.dataset[i][A]
                    if (A,V) not in self.default_value:
                        self.default_value[(A,V)]=(0,0)
                    pos, neg = self.default_value[(A,V)]
                    if res==self.positive_label:
                        pos += 1
                    else:
                        neg += 1
                    self.default_value[(A,V)]=(pos,neg)   
        else:
            self.default_value = None

        self.root = self.learn (self.dataset, set(range(self.num_feature)))
        return self
    
    class Node:
        def __init__ (self, splitFeature=None, examples=None):
            self.splitFeature = splitFeature    #-1 nếu là lá
            self.conclusion = 0 #1 cho +, -1 cho - và 0 cho Inconclusive
            self.examples = examples
            self.child = {}
        def is_leaf (self):
            return self.splitFeature == -1

    @staticmethod
    def majority_vote (pos, neg):
        if pos>neg:
            return 1
        elif pos<neg:
            return -1
        else:
            return 0
    #enddef majority_vote

    def learn (self, examples, available_attributes):

        def avg_entropy (feature, examples):
            n = len(examples)
            dic = {}
            for i in range(n):
                V=examples[i][feature]
                res=examples[i][self.num_feature]
                if V not in dic:
                    dic[V]=(0,0)
                pos,neg=dic[V]
                if res==self.positive_label:
                    pos += 1
                else:
                    neg += 1
                dic[V]=(pos,neg)
            ans = 0
            for V in dic.keys():
                pos, neg = dic[V]
                m=pos+neg

                def entropy (pos, neg):
                    ans=0; m=pos+neg
                    if (pos>0):
                        ans += -(pos/m)*math.log2(pos/m)
                    if (neg>0):
                        ans += -(neg/m)*math.log2(neg/m)
                    return ans
                #enddef entropy

                term = (m/n) * entropy(pos,neg)
                ans += term
            return ans
        #enddef avg_entropy

        def stat (examples):
            n=len(examples)
            pos=0;neg=0
            for i in range(n):
                res=examples[i][self.num_feature]
                if res==self.positive_label:
                    pos += 1
                else:
                    neg += 1
            return pos,neg
        #enddef stat


        cur_node = id3.Node(-1, examples)
        tot_pos, tot_neg = stat(examples)
        if tot_neg == 0:    # +
            cur_node.conclusion = 1
        elif tot_pos == 0:
            cur_node.conclusion = -1
        elif len(available_attributes) == 0:
            cur_node.conclusion = self.majority_vote(tot_pos, tot_neg)
        else:
            minEntropy=1.01; splitFeature=-1
            for A in available_attributes:
                tmp = avg_entropy(A, examples)
                if tmp < minEntropy:
                    minEntropy=tmp; splitFeature=A

            
            cur_node.splitFeature = splitFeature
            cur_node.conclusion = 0

            dic = {}
            # For each value V of splitFeature, store the examples
            while len(examples)>0:
                ex=examples.pop()
                V=ex[splitFeature]
                if V not in dic:
                    dic[V] = []
                dic[V].append(ex)

            # If feature have only one value, it is pointless to split
            if len(dic) == 1:
                cur_node.splitFeature = -1
                V=list(dic.keys())[0]
                cur_node.examples = dic[V]
                pos,neg = stat(cur_node.examples)
                cur_node.conclusion = self.majority_vote(pos,neg)
            
            # Now, split
            new_attrib = available_attributes.copy()
            new_attrib.remove(splitFeature)
            for V in dic:
                cur_node.child[V] = self.learn(dic[V], new_attrib)
        return cur_node

    def classify (self, case):
        cur_node = self.root
        while cur_node.splitFeature != -1:
            A=cur_node.splitFeature;V=case[A]
            if V in cur_node.child:
                cur_node = cur_node.child[V]
            else:
                if self.default_value is not None:
                    pos,neg=self.default_value[(A,V)]
                    return self.majority_vote(pos,neg)
                else:
                    return 0 #Inconclusive
        return cur_node.conclusion

    def classify_label (self, case, positive='1', negative='-1', inconclusive='0'):
        res = self.classify(case)
        if res > 0:
            return positive
        elif res < 0:
            return negative
        else:
            return inconclusive