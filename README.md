# Water-Potability Prediction
“Potable water” simply means water that is safe to drink, and it is becoming scarcer in the world.
Increasing use is stressing freshwater resources worldwide, and a seemingly endless list of
contaminants can turn once potable water into a health hazard or simply make it unacceptable
aesthetically.
Of the more than 2 billion people who lack potable water at home, 844 million don’t have even
basic drinking water service, including 263 million who must travel 30 minutes per trip to collect
water. About 159 million drink untreated surface water. Unsafe drinking water is a major cause
of diarrheal disease, which kills about 800,000 children under the age of 5 a year, usually in
developing countries, but 90 countries are expected to fail to reach the goal of universal coverage
by 2030.

## Dataset

The dataset provided to me has 9 attributes which shall be taken as input from the user. Using our
prediction model over the provided inputs, we are going to predict whether the water is potable or
not with regards to the input provided. The following were the attributes in the given dataset.

- pH: pH of the given water sample
- HARDNESS: Hardness of the water sample
- SOLIDS: Solid contents of the water
- CHLORAMINES: Amount of Chloramines in the water
- SULFATE: Amount of Sulfates in water
- CONDUCTIVITY: Conductivity of the given water sample
- ORGANIC CARBON: The carbon content in the water
- TRIHALOMETHANES: Amount of halogen-methane in the water sample
- TURBIDITY: Turbidity of the given water sample.


## Graphical Depiction of the attributes

<img src="https://github.com/2000utkarsh/Water-Potability/blob/master/app/imgs/Screenshot%202021-09-20%20at%208.21.58%20PM.png" width="600" height="400" />

<img src="https://github.com/2000utkarsh/Water-Potability/blob/master/app/imgs/Screenshot%202021-09-20%20at%208.22.42%20PM.png" width="600" height="400" />

<img src="https://github.com/2000utkarsh/Water-Potability/blob/master/app/imgs/Screenshot%202021-09-20%20at%208.22.46%20PM.png" width="600" height="400" />

<img src="https://github.com/2000utkarsh/Water-Potability/blob/master/app/imgs/Screenshot%202021-09-20%20at%208.22.50%20PM.png" width="600" height="400" />


## Models Used:

• Decision Tree Classifier

<img src="https://github.com/2000utkarsh/Water-Potability/blob/master/app/imgs/Screenshot%202021-09-20%20at%208.23.04%20PM.png" width="500" height="600" />

• Logistic Regression

<img src="https://github.com/2000utkarsh/Water-Potability/blob/master/app/imgs/Screenshot%202021-09-20%20at%208.23.18%20PM.png" width="500" height="600" />

• Random Forest Classifier

<img src="https://github.com/2000utkarsh/Water-Potability/blob/master/app/imgs/Screenshot%202021-09-20%20at%208.23.23%20PM.png" width="500" height="600" />

• K-Neighbor Classifier

<img src="https://github.com/2000utkarsh/Water-Potability/blob/master/app/imgs/Screenshot%202021-09-20%20at%208.23.31%20PM.png" width="500" height="600" />

• SVC

<img src="https://github.com/2000utkarsh/Water-Potability/blob/master/app/imgs/Screenshot%202021-09-20%20at%208.23.37%20PM.png" width="500" height="600" />

• Multinomial NB

<img src="https://github.com/2000utkarsh/Water-Potability/blob/master/app/imgs/Screenshot%202021-09-20%20at%208.23.49%20PM.png" width="500" height="600" />

## Approach:
During the very first phase of the model, I cleaned the data and made it more sensible in numeric form, i.e. converted all the categorical data to numerical data and removed the NULL and missing values.
In the Second Phase, I did some visualizations over the given dataset which gave me a great view and hints for the implementation of the model.
In the third phase, I implemented various methods over the given dataset, the ones I had mentioned previously. The model gave prediction taking each and every model into consideration and then took out the average value of all the models result and then gave the result which were based on the provided input by the user.
In the final phase, I developed the UI for the model and then linked my model with the created UI for to make it ready for deployment in Django.


## User Interface:



<img src="https://github.com/2000utkarsh/Water-Potability/blob/master/app/imgs/Screenshot%202021-09-20%20at%208.23.57%20PM.png" width="500" height="400" />

<img src="https://github.com/2000utkarsh/Water-Potability/blob/master/app/imgs/Screenshot%202021-09-20%20at%208.24.03%20PM.png" width="500" height="400" />


## Output Screen
<img src="https://github.com/2000utkarsh/Water-Potability/blob/master/app/imgs/Screenshot%202021-09-20%20at%208.24.07%20PM.png" width="500" height="400" />




## Future Scope

In near future, when the water is getting scarce and the remaining water is getting increasingly polluted, it is very essential to check if the water is potable or not. Now, it takes a high deal of money for the testing of the water, but if such a model is deployed where upon receiving inputs it can predict if the water is potable or not, it would be highly efficient for the users. Hence, our model has a huge scope in future as well.
















