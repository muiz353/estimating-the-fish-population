# Estimating the fish population in the five Nordic countries

Final project for the Building AI course

## Summary

My project is about estimating the fish population in the five Nordic countries (Denmark, Finland, Iceland, Norway, and Sweden) based on the number of fishermen in each country. The project utilizes linear regression methods to find the relationship between the independent variable (number of fishermen) and the dependent variable (fish population). This project can assist researchers, managers, and policymakers in monitoring and protecting fisheries resources in the Nordic region. 


## Background
The problem addressed by my idea is the lack of accurate and up-to-date data on fish populations in the Nordic region. This issue frequently arises due to the difficulty in directly calculating or measuring fish populations. The problem is significant as fish populations impact ecosystem balance, community well-being, and the national economy. My personal motivation is to learn more about fisheries and the environment, as well as how AI can assist in addressing challenges related to these fields.

## How is it used?
The process of using my solution is as follows:

Firstly, I collect data on the number of fishermen and fish populations in the five Nordic countries from various sources, such as government reports, international organizations, or scientific research.
Secondly, I clean and process the data, including removing invalid data, filling in missing data, or normalizing data.
Thirdly, I use linear regression methods to find the equation of the line that best fits the data by minimizing the square of the errors between actual and predicted values.
Fourthly, I use this linear equation to estimate fish populations in each country based on the number of fishermen, or conversely, estimate the required number of fishermen to achieve a desired fish population.
Fifthly, I evaluate the performance and accuracy of my model using metrics such as the coefficient of determination, mean absolute error, or mean squared error.
Sixthly, I present my results and findings in the form of tables, graphs, or reports.

The situations where my solution is needed are when there is a requirement to know or predict fish populations in the Nordic region, whether for research, management, or policy purposes. Users of my solution are researchers, managers, or policymakers interested in fisheries and the environment. Considerations include the availability and quality of data, the validity and reliability of the model, as well as the ethics and social impact of my solution.
Images will make your README look nice!
Once you upload an image to your repository, you can link link to it like this (replace the URL with file path, if you've uploaded an image to Github.)
![Cat](https://upload.wikimedia.org/wikipedia/commons/5/5e/Sleeping_cat_on_her_back.jpg)

def main():
   countries = ['Denmark', 'Finland', 'Iceland', 'Norway', 'Sweden']
   pop = [5615000, 5439000, 324000, 5080000, 9609000]   # not actually needed in this exercise...
   fishers = [1891, 2652, 3800, 11611, 1757]

   totPop = sum(pop)
   totFish = sum(fishers)

   # write your solution here

   for i in range(len(countries)):
      print("%s %.2f%%" % (countries[i], 100.0))    # current just prints 100%

main()

## Data sources and AI methods
[Twitter API](https://developer.twitter.com/en/docs)

| Syntax      | Description |
| ----------- | ----------- |
| Header      | Title       |
| Paragraph   | Text        |

## Challenges

My project does not address several issues, such as:
The presence of other factors influencing fish populations besides the number of fishermen, such as environmental conditions, climate change, migration patterns, or natural predators.
The existence of variations or differences in the types, sizes, or quality of fish caught by fishermen in each country, which can affect the economic value or benefits of the catch.
The possibility of errors, biases, or manipulation in the collection or reporting of data by involved parties, which can impact the accuracy or reliability of the model.
Some limitations and ethical considerations to be noted when implementing such a solution are:
The availability and quality of sufficient and relevant data to train and test the model, as well as mechanisms to verify and validate this data.
The validity and reliability of the model reflecting the actual relationships between the variables under investigation, as well as mechanisms to evaluate and enhance the performance and accuracy of the model.
Ethics and social impact of the solution that can influence the interests and well-being of various stakeholders, such as fishermen, researchers, managers, policymakers, or the general public, along with mechanisms to anticipate and address conflicts or issues that may arise.


My project can evolve and become something more by doing several things, including:
Conducting further research on other factors influencing fish populations, as well as exploring methods to measure or model these factors.
Developing a more complex or sophisticated model capable of handling more than one independent or dependent variable, as well as capturing non-linear relationships or interactions between these variables.
Collaborating or forming partnerships with various stakeholders related to fisheries and the environment, such as international organizations, research institutions, or fishing communities.


I would like to express my gratitude to all parties who have assisted and supported my project, both directly and indirectly. In particular, I would like to thank:
The Food and Agriculture Organization of the United Nations (FAO), Nordic Council of Ministers, International Council for the Exploration of the Sea (ICES), and the European Commission for providing data and information on the number of fishermen and fish populations in the five Nordic countries, as well as the facilities and assistance needed to access this data.
The research team from the University of Helsinki, who provided guidance and scientific input on the methods and techniques of linear regression, as well as shared their experiences and findings from their research.
Instructors and fellow participants from the Building AI course, who offered advice, input, feedback, and recognition to my project, serving as examples and inspiration for me in learning and creating AI projects.
Scikit-learn, a Python library that provides various tools for machine learning, which I used to create and train my linear regression model. I appreciate the creators and contributors of scikit-learn, and I include the appropriate license and links. Scikit-learn is created by Fabian Pedregosa et al. / BSD 3-Clause License.
