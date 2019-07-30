import numpy as np


# 是否还存在误分类点
def isHasMisclassification(x, y, w, b):
    misclassification = False
    ct = 0
    misclassification_index = 0
    for i in range(0, len(y)):
        if y[i]*(np.dot(w, x[i]) + b) <= 0:
            ct += 1
            misclassification_index = i
    if ct > 0:
        misclassification = True
    return  misclassification, misclassification_index


# 更新参数
def update(x, y, w, b, i):
    w = w + y[i] * x[i]
    b = b + y[i]
    return w, b


#更新迭代
def optimization(x, y, w, b):
    misclassification, misclassification_index = isHasMisclassification(x, y, w, b)
    while misclassification:
        print("误分类的点：", misclassification_index)
        w, b = update(x, y, w, b, misclassification_index)
        print("采用误分类点 {} 更新后的权重为：w是{}， b是{} ".format(misclassification_index, w, b))
        misclassification, misclassification_index = isHasMisclassification(x, y, w, b)
    return w, b


if __name__ == "__main__":
    # 输入分类点 x为分类点坐标，y为分类点的属性
    x = np.array([[3, 3], [4, 3], [1, 1]])
    y = [1, 1, -1]

    # w, b为分离超平面的参数
    w = [0, 0]
    b = 0
    yita = 1

    w, b = optimization(x, y, w, b)
    print("超平面的w为 {}，b为 {}。".format(w, b))