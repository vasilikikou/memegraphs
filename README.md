## MemeGraphs: Linking memes to knowledge graphs for hateful memes classification

This repository contains the code and MemeGraphs input 
data used in our paper, which has been accepted for 
presentation at the 17th International Conference on Document 
Analysis and Recognition (ICDAR 2023).

### Abstract

Memes are a popular form of communicating trends and ideas in social
media and on the internet in general, combining the modalities of images
and text. They can express humor and sarcasm but can also have offensive content.
Analyzing and classifying memes automatically is challenging since their
interpretation relies on the understanding of visual elements, language, and background
knowledge. Thus, it is important to meaningfully represent these sources
and the interaction between them in order to classify a meme as a whole. In this
work, we propose to use scene graphs, that express images in terms of objects
and their visual relations, and knowledge graphs as structured representations for
meme classification with a Transformer-based architecture. We compare our approach
with ImgBERT, a multimodal model that uses only learned (instead of
structured) representations of the meme, and observe consistent improvements.
We further provide a dataset with human graph annotations that we compare to
automatically generated graphs and entity linking. Analysis shows that automatic
methods link more entities than human annotators and that automatically generated
graphs are better suited for hatefulness classification in memes.


### Run MemeGraphs

The code was run on Google Colab Pro.
In the `run_memegraphs.ipynb` notebook you can see how to download 
the data and run the code.


### Data

The dataset used for these experiments is 
[MultiOFF](https://aclanthology.org/2020.trac-1.6/), which is 
available [here](https://drive.google.com/drive/folders/1hKLOtpVmF45IoBmJPwojgq6XraLtHmV6?usp=sharing).
In our experiments we use the training, validation and test data 
provided by the authors of MultiOFF.

The additional MemeGraphs input used for the *MemeGraphs[SceneGr]*,
*MemeGraphs[Know]* and *MemeGraphs[SceneGr+Know]* models can be found 
in the **data** folder.

### Citation

*To be added.*

### Acknowledgements

This research was funded by the Deutsche Forschungsgemeinschaft 
(DFG, German Research Foundation) - RO 5127/2-1 and the Vienna 
Science and Technology Fund (WWTF)[10.47379/VRG19008].
