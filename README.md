# Titanic Survival Prediction Model

This project predicts the survival chances of passengers aboard the Titanic based on various features such as age, sex, passenger class, and more. The model uses machine learning algorithms to predict survival outcomes, and it is deployed using Qwack for efficient management of the machine learning lifecycle (MLOps).

## Table of Contents
- [Problem Statement](#problem-statement)
- [Assumptions](#assumptions)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [How to Run the Model](#how-to-run-the-model)
- [API Documentation](#api-documentation)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [MLOps and DevOps with Qwack](#mlops-and-devops-with-qwack)
- [Contributing](#contributing, N/A)
- [License](#license, e.g. by MIT in this case by Johnson)

## Problem Statement
The Titanic dataset contains information about passengers who were aboard the Titanic. The goal of this project is to predict whether a passenger survived or not based on various features, such as:
- Age
- Sex
- Passenger Class
- Fare
- Embarked port
- and more...

## Assumptions
- The dataset contains historical passenger data from the Titanic and is used as a representation of typical survival factors.
- The machine learning model trained on this data will generalize well to other similar datasets.
- The model's performance may degrade over time (concept drift), and retraining may be necessary.

## Technologies Used
- **Machine Learning Libraries**: 
  - Scikit-learn
  - Pandas
  - NumPy
- **Deployment**: 
  - Qwack (for MLOps, model versioning, deployment, and monitoring)
- **Web Framework**:
  - Flask (for building the API)
- **Containerization**:
  - Docker (to containerize the application for portability)
- **Version Control**:
  - Git/GitHub (for version control and collaboration)

## Setup Instructions

### Prerequisites:
Ensure that you have the following installed:
- Python 3.x
- Git
- Docker (if running locally in a container)

### 1. Clone the Repository:
Clone the repository to your local machine.

```with bash
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction

```your config
pip install -r requirements.txt
export QWAK_TOKEN="your_actual_token"

```testing e.g. your local or cloud or docker
python app.py

curl --location --request POST 'http://localhost:5000/predict' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $QWAK_TOKEN' \
--data '{"columns":["Sex","SibSp","Embarked","Cabin","Ticket","PassengerId","Parch","Fare","Name","Pclass","Age"],"index":[0],"data":[["male",0,"S","","",0,0,7.25,"Johnson",3,22]]}'

docker build -t titanic-survival .
docker run -p 5000:5000 titanic-survival
