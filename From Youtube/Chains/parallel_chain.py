from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableParallel

llm = ChatOllama(model = "llama3.2")

prompt1 = PromptTemplate(
    template = "Give me simple and short notes on \n {text}",
    input_variables = ["text"]
)

prompt2 = PromptTemplate(
    template = "Give 5 mcq question quiz using the notes \n {text}",
    input_variables = ["text"]
)

prompt3 = PromptTemplate(
    template = "Given notes and quiz to you, just merge them and give me a singe document \n {notes} \n {quiz}",
    input_variables = ["notes", "quiz"]
)
parser = StrOutputParser()

chain1 = prompt1 | llm | parser
chain2 = prompt2 | llm | parser

parallel_chain = RunnableParallel({
    "notes": chain1,
    "quiz": chain2
})

text = """
The KMeans algorithm clusters data by trying to separate samples in n groups of equal variance, minimizing a criterion known as the inertia or within-cluster sum-of-squares (see below). This algorithm requires the number of clusters to be specified. It scales well to large numbers of samples and has been used across a large range of application areas in many different fields.

The k-means algorithm divides a set of samples into disjoint clusters 
, each described by the mean of the samples in the cluster. The means are commonly called the cluster “centroids”; note that they are not, in general, points from 
, although they live in the same space.

The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum-of-squares criterion:

Inertia can be recognized as a measure of how internally coherent clusters are. It suffers from various drawbacks:

Inertia makes the assumption that clusters are convex and isotropic, which is not always the case. It responds poorly to elongated clusters, or manifolds with irregular shapes.

Inertia is not a normalized metric: we just know that lower values are better and zero is optimal. But in very high-dimensional spaces, Euclidean distances tend to become inflated (this is an instance of the so-called “curse of dimensionality”). Running a dimensionality reduction algorithm such as Principal component analysis (PCA) prior to k-means clustering can alleviate this problem and speed up the computations.

../_images/sphx_glr_plot_kmeans_assumptions_002.png
For more detailed descriptions of the issues shown above and how to address them, refer to the examples Demonstration of k-means assumptions and Selecting the number of clusters with silhouette analysis on KMeans clustering.

K-means is often referred to as Lloyd’s algorithm. In basic terms, the algorithm has three steps. The first step chooses the initial centroids, with the most basic method being to choose samples from the dataset 
. After initialization, K-means consists of looping between the two other steps. The first step assigns each sample to its nearest centroid. The second step creates new centroids by taking the mean value of all of the samples assigned to each previous centroid. The difference between the old and the new centroids are computed and the algorithm repeats these last two steps until this value is less than a threshold. In other words, it repeats until the centroids do not move significantly.

../_images/sphx_glr_plot_kmeans_digits_001.png
K-means is equivalent to the expectation-maximization algorithm with a small, all-equal, diagonal covariance matrix.

The algorithm can also be understood through the concept of Voronoi diagrams. First the Voronoi diagram of the points is calculated using the current centroids. Each segment in the Voronoi diagram becomes a separate cluster. Secondly, the centroids are updated to the mean of each segment. The algorithm then repeats this until a stopping criterion is fulfilled. Usually, the algorithm stops when the relative decrease in the objective function between iterations is less than the given tolerance value. This is not the case in this implementation: iteration stops when centroids move less than the tolerance.

Given enough time, K-means will always converge, however this may be to a local minimum. This is highly dependent on the initialization of the centroids. As a result, the computation is often done several times, with different initializations of the centroids. One method to help address this issue is the k-means++ initialization scheme, which has been implemented in scikit-learn (use the init='k-means++' parameter). This initializes the centroids to be (generally) distant from each other, leading to probably better results than random initialization, as shown in the reference. For detailed examples of comparing different initialization schemes, refer to A demo of K-Means clustering on the handwritten digits data and Empirical evaluation of the impact of k-means initialization.

K-means++ can also be called independently to select seeds for other clustering algorithms, see sklearn.cluster.kmeans_plusplus for details and example usage.

The algorithm supports sample weights, which can be given by a parameter sample_weight. This allows to assign more weight to some samples when computing cluster centers and values of inertia. For example, assigning a weight of 2 to a sample is equivalent to adding a duplicate of that sample to the dataset 
."""

merge_chain = prompt3 | llm | parser

chain = parallel_chain | merge_chain

result = chain.invoke({"text": text})

print(result)
chain.get_graph().print_ascii()

