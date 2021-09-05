Wallstreetbets Reddit Analysis
================
Rodrigo Durso
3/31/2021

## Wallstreetbets Reddit Analysis

The purpose of this document is to analyze the posts in the subreddit
wallstreetbets from January 2019 to February 2021. The dataset is
available at <https://www.kaggle.com/unanimad/reddit-rwallstreetbets>.
The dataset contains several variables, but for the purpose of this
document we are taking into account the following ones:

-   title: the title of the post.
-   created\_utc: date of the post publication.

We start by taking a sample of 10,000 observations. Next, we convert the
vector containing the title of the post to a corpus, which is just a
collection of text documents.

Finally, we modify the corpus by stemming and removing punctuations,
white spaces and stop words. As an example, we compare a resulting text
following the text transformation:

-   Weekend Discuss Thread Weekend October

With the text prior to the transformation:

-   Weekend Discussion Thread for the Weekend of October 16, 2020

We then convert the corpus to a Term Document Matrix, which is a
representation of our text documents in a matrix of numbers. The below
table shows a subset of the Term Document Matrix.

| buy | dumb | puts | spi | collegiate | almost | call | casual | comeback | dont |
|----:|-----:|-----:|----:|-----------:|-------:|-----:|-------:|---------:|-----:|
|   1 |    1 |    1 |   1 |          0 |      0 |    0 |      0 |        0 |    0 |
|   0 |    0 |    0 |   0 |          1 |      0 |    0 |      0 |        0 |    0 |
|   0 |    0 |    0 |   0 |          0 |      1 |    1 |      1 |        1 |    1 |
|   0 |    0 |    0 |   0 |          0 |      0 |    0 |      0 |        0 |    0 |
|   0 |    0 |    0 |   0 |          0 |      0 |    0 |      0 |        0 |    0 |
|   0 |    0 |    0 |   0 |          0 |      0 |    0 |      0 |        0 |    0 |
|   0 |    0 |    0 |   0 |          0 |      0 |    0 |      0 |        0 |    0 |
|   0 |    0 |    0 |   0 |          0 |      0 |    0 |      0 |        0 |    0 |
|   1 |    0 |    0 |   0 |          0 |      0 |    0 |      0 |        0 |    0 |
|   0 |    0 |    0 |   0 |          0 |      0 |    0 |      0 |        0 |    0 |
|   0 |    0 |    0 |   0 |          0 |      0 |    0 |      0 |        0 |    0 |
|   0 |    0 |    0 |   0 |          0 |      0 |    0 |      0 |        0 |    0 |
|   0 |    0 |    0 |   0 |          0 |      0 |    0 |      0 |        0 |    0 |
|   0 |    0 |    0 |   0 |          0 |      0 |    0 |      0 |        0 |    0 |
|   0 |    0 |    0 |   0 |          0 |      0 |    0 |      0 |        0 |    0 |

We can observe below that we are working with a highly dimensional
dataset with over 9,000 dimensions.

| Var1             |  Freq |
|:-----------------|------:|
| No. of Documents | 10000 |
| Dimensions       |  9278 |

As a result, we remove the rare words to reduce the dimensionality in
order to make matrix operations feasible. There are other problems
related to multidimensionality in document analysis, in particular
distance based clustering, but that is outside of the scope of this
document.

We check the words with the highest frequency in order to and plot them
on a graph.

![](GME_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

We now focus on posts containing the word GME. The below graph shows the
evolution of the number of posts containing the word GME over time.

![](GME_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

We finish our review by plotting a network to highlight which words are
more often linked to GME. We could also have used a clustering
algorithm, but we have opted to a graph network for visualization
purposes. We can observe below that GME is often linked to the words
“AMC”, buy" and “hold”.

![](GME_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->
