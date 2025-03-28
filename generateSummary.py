import base64
import os
from google import genai
from google.genai import types


def summary(code_text):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""As an AI language model, your task is to analyze machine learning scripts and convert them into a structured JSON format. For each script provided, you should:

1. **Identify Key Phases:** Determine the main stages in the code, such as \"Data Loading,\" \"Preprocessing,\" \"Model Training,\" and \"Evaluation.\"

2. **Extract Code Snippets:** For each identified phase, extract all relevant code snippets. If a phase is complex, divide it into sub-phases and update the JSON structure accordingly.

3. **Provide Non-Technical Descriptions:** Offer simple, non-technical explanations for each phase to ensure clarity for individuals without coding expertise.

4. **Output Structured JSON:** Return the breakdown in JSON format, adhering to the structure outlined below. Ensure that no text appears outside the JSON body.

**JSON Structure:**


```json
{
  \"phases\": [
    {
      \"phase\": \"Phase Name\",
      \"description\": \"Simple explanation of the phase.\",
      \"code\": [
        \"Relevant code snippet 1\",
        \"Relevant code snippet 2\",
        ...
      ],
      \"sub_phases\": [
        {
          \"sub_phase\": \"Sub-Phase Name\",
          \"description\": \"Simple explanation of the sub-phase.\",
          \"code\": [
            \"Relevant code snippet 1\",
            \"Relevant code snippet 2\",
            ...
          ]
        },
        ...
      ]
    },
    ...
  ]
}
```


**Guidelines:**

- **Meaningful Naming:** Assign clear and descriptive names to each phase and sub-phase.

- **Comprehensive Code Extraction:** Include all pertinent code snippets for each phase or sub-phase.

- **Clarity in Descriptions:** Craft explanations that are straightforward and accessible to non-coders.

- **Structured Output:** Ensure the JSON output strictly follows the specified format without any extraneous text.

By following these instructions, you will effectively transform complex AI/ML code into an organized and easily understandable JSON representation. """),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""Okay, I understand. I am ready to process machine learning scripts and output them in the specified JSON format, including phase and sub-phase identification, code extraction, and non-technical descriptions.

Let's wait for the input script!
"""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""```json
{
  \"phases\": [
    {
      \"phase\": \"Data Loading and Preparation\",
      \"description\": \"Loads the customer data from a file and selects the relevant columns for analysis.\",
      \"code\": [
        \"import numpy as np\",
        \"import matplotlib.pyplot as plt\",
        \"import pandas as pd\",
        \"dataset = pd.read_csv('Mall_Customers.csv')\",
        \"X = dataset.iloc[:, [3, 4]].values\"
      ]
    },
    {
      \"phase\": \"Finding Optimal Number of Clusters (Elbow Method)\",
      \"description\": \"Determines the best number of customer groups by plotting the WCSS (Within-Cluster Sum of Squares) for different numbers of clusters and identifying the 'elbow' point.\",
      \"code\": [
        \"from sklearn.cluster import KMeans\",
        \"wcss = []\",
        \"for i in range(1, 11):\",
        \"    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)\",
        \"    kmeans.fit(X)\",
        \"    wcss.append(kmeans.inertia_)\",
        \"plt.plot(range(1, 11), wcss)\",
        \"plt.title('The Elbow Method')\",
        \"plt.xlabel('Number of clusters')\",
        \"plt.ylabel('WCSS')\",
        \"plt.show()\"
      ]
    },
    {
      \"phase\": \"Model Training\",
      \"description\": \"Trains the K-Means clustering model using the chosen number of clusters (5 in this case).  The model learns to group customers based on their income and spending score.\",
      \"code\": [
        \"kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)\",
        \"y_kmeans = kmeans.fit_predict(X)\"
      ]
    },
    {
      \"phase\": \"Visualization of Clusters\",
      \"description\": \"Displays the identified customer groups (clusters) on a graph, showing the relationship between income and spending score for each group.  The centroids (centers) of each cluster are also shown.\",
      \"code\": [
        \"plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')\",
        \"plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')\",
        \"plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')\",
        \"plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')\",
        \"plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')\",
        \"plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')\",
        \"plt.title('Clusters of customers')\",
        \"plt.xlabel('Annual Income (k$)')\",
        \"plt.ylabel('Spending Score (1-100)')\",
        \"plt.legend()\",
        \"plt.show()\"
      ]
    }
  ]
}
```"""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=code_text),
            ],
        ),
    ]
  
    generate_content_config = types.GenerateContentConfig(
          response_mime_type="text/plain",
    )

    response = client.models.generate_content(
      model=model,
      contents=contents,
      config=generate_content_config,
    )
    
    return  response.text
    

