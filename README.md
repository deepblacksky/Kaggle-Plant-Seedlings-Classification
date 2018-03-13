# Plant-Seedlings-Classification
本项目是kaggle 比赛 [Plant Seedlings Classification](https://www.kaggle.com/c/plant-seedlings-classification)

本次比赛是对12种植物的幼苗进行分类，具体描述请参考比赛链接

本次项目采用基于tensorflow的keras框架

## step
- 在kaggle上下载比赛所用的训练集以及测试集等，解压至`data`目录
- `split.py` 将训练集按照比例分成训练集和验证集
- 利用keras在imagenet的预处理模型进行搭建网络，并重新重建分类层，即fine-tuning. 
其中 `mobilenet_classifer.py` `xception_classifer.py` 等便是基于此构建。
- 为了提高模型的泛化能力，提高准确率，使用图片增广。改写keras中`keras.preprocessing.image.ImageDataGenerator` 的某些函数。比如图片标准化，首先可以利用`mean_and_var.py`求得数据集中所有图片的均值和方差。此种标准化图片方法可提高准确率。
- 训练参数设置。在fine-tuning预训练网络时，在机器允许的前提下适当提高batch_size和减少frozen网络的层数。合适的learning decay策略可以提高准确率，另外early stop也可以提高准确率。
- 模型ensemble在比赛中是重要手段。当单模型的能力有限时，模型融合可以将准确率提高1-5个百分点，具体可见`ensembling.py`

## update
 - 对植物种子图片进行前景提取处理。由于训练集是单纯的种子和土壤的图片，如下：
 ![](https://www.github.com/deepblacksky/Plant-Seedlings-Classification/raw/master/images/1.png) \
然后只把绿色的部分提取出来，其他部分则删除掉,
```
def create_mask_for_plant(image):
    """产生植物图片的掩码
    """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

def segment_plant(image):
    """按照mask分割植物图片
    """
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask=mask)

    return output

def sharpen_image(image):
    """锐化图片
    """
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)

    return image_sharp

image = cv2.imread(os.path.join(sub_path, image_path))
image_segmented = segment_plant(image)
image_sharpen = sharpen_image(image_segmented)
``` 
处理过后如下：

![](https://www.github.com/deepblacksky/Plant-Seedlings-Classification/raw/master/images/2.png) \
这样清洗过后的数据更加简洁，可以提高一点准确率，另外一个好处是可以大幅减少运算量，加快模型收敛。

- 新的训练策略。在`new_xception_classifer.py`和`new_densnet121_classifer.py` 中，采用了新的训练策略。在预训练模型基础上自己添加分类网络，训练时先将特征提却网络全部固定住，用大约40epochs只训练分类网络，之后再用150epochs训练整个全部网络。这样可以避免新加入的分类网络在一开始参数随机时对特征提取网络造成较大破坏，这样训练策略同样可以提升准确率。

- 其他的tips
    - 增加输入网络的图片的size
    - 在预测时对测试集采用和训练集相同的图像增广
    - 模型融合是采用多种策略，vote，max，average等

## *reference:*
- https://keras.io/
- https://www.kaggle.com/c/plant-seedlings-classification/discussion

