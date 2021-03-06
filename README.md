# Malaria-Detection

Dataset can be found here https://ceb.nlm.nih.gov/repositories/malaria-datasets/

The dataset contains 27,588 cell images belonging to 2 classes:
```
1.Parasitized: Implying that the region contains malaria.

2.Uninfected: Meaning there is no evidence of malaria in the region.
```
The number of images per class is equally distributed with 13,794 images per each respective class.



# Configuration file
```
Config.py file stores all my constant variables
```

80% of the data is for training and the remaining 20% will be for testing.
10% for validation 

# Resnet.py
```
Pre-Trained Resnet model is used for this image classification problem.(resnet.py)
```

# Installing Dependencies

Supported Python version 3 or higher

* [Keras](https://keras.io/) - ``` pip install keras ```
* [Numpy](http://www.numpy.org/) & [Scikit-learn](https://scikit-learn.org/stable/) - ``` pip install -U scikit-learn ```
* [Matplotlib](https://matplotlib.org/) - ``` pip install matplotlib ```
* [imutils](https://github.com/jrosebr1/imutils) ``` pip install --upgrade imutils ```

# Result

At the end of the 50th epoch we are obtaining:
```
96.50% accuracy on the training data
96.78% accuracy on the validation data
97% accuracy on the testing data
```
